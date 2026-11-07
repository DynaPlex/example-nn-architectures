"""Stage 1 of the convergence-speed experiment: give every arm its own learning rate.

WHY THIS COMES FIRST. The claim under test is "SAT reaches a given policy quality
in fewer epochs than the original attention nets". All three arms currently share
`lr=1e-4`, and for a SPEED claim that is close to fatal: the learning rate IS the
convergence-speed knob. "SAT converges in 40 epochs and attention_net needs 85"
invites the obvious reply that attention_net at 3e-4 needs 30. Note that 1e-4 is
not a neutral default either -- `dynaplex/nn.py:366-369` records it as the value
tuned for the Akkerman paper's FLAT MLP, not for any of these three.

So each arm gets the same grid on the same samples with the same seeds, and the
learning curves in stage 2 are run at each arm's own best rate. That converts the
claim from "SAT is faster at someone else's learning rate" into "SAT is faster
when each architecture is given its best rate from a common budget", which is the
version that survives a referee.

SELECTION RULE, fixed before looking: for each arm, pick the lr with the lowest
MEAN POLICY COST at the final rung across seeds, breaking ties toward the smaller
lr. Cost, not validation regret -- the `default`-budget run established that
validation regret does not determine closed-loop cost (pointer seed 0: regret
2.311, cost 28.150; tokens seed 1: regret 2.661, cost 2.303). The per-epoch
validation curves are recorded anyway because they are free, but they do not drive
the choice.

Reuses the collections already sitting in the `default` budget's workdirs, so this
costs training time only.

Usage:
    python run_lr_screen.py --seeds 2            # ~6.5 h on 16 cores
    python run_lr_screen.py --epochs 20 --seeds 1   # a shorter look
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import time

import numpy as np
import torch

import dynaplex as dp
from dynaplex.sample_set import SampleSet

from benchmark_config import (
    ALL_ARMS,
    BENCHMARK,
    BUDGETS,
    COMPARER_SEED,
    DEFAULT_ARMS,
    RESULTS_ROOT,
    WORKDIR_ROOT,
)
from learning_curve import train_with_snapshots
from mdp import MyopicPolicy, VehicleRoutingMDP
from run_benchmark import degeneracy, workdir_for

LR_GRID = (3e-5, 1e-4, 3e-4, 1e-3)
SCREEN_TAG = "lr_screen"


def samples_for(arm_name: str, source_budget) -> str:
    """The arm's already-collected gen-0 sample set, from the default budget."""
    wd = workdir_for(WORKDIR_ROOT, source_budget, arm_name, 0)
    path = os.path.join(wd, "samples_gen0")
    if not os.path.isdir(path):
        raise SystemExit(
            f"no collected samples at {path}. Run "
            f"`python run_benchmark.py --budget default --seeds 1` first, or point "
            f"--source at a budget whose collections exist.")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40,
                    help="rung at which each (arm, lr) is scored")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--lrs", default=",".join(str(x) for x in LR_GRID))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--source", default="default",
                    help="budget whose collected samples to reuse")
    args = ap.parse_args()

    workers = args.workers or multiprocessing.cpu_count()
    torch.set_num_threads(min(workers, 32))
    source = BUDGETS[args.source]
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    lrs = [float(x) for x in args.lrs.split(",")]
    seeds = list(range(args.seeds))
    outdir = os.path.join(RESULTS_ROOT, SCREEN_TAG)
    os.makedirs(outdir, exist_ok=True)

    print(f"=== lr screen | arms={arm_names} | lrs={lrs} | seeds={seeds} "
          f"| rung={args.epochs} | workers={workers}")
    print(f"    reusing collections from budget {source.name} ({source.tag})")
    print(f"    {len(arm_names) * len(lrs) * len(seeds)} runs x {args.epochs} epochs")

    trained: dict[str, dict] = {}
    for arm_name in arm_names:
        arm = ALL_ARMS[arm_name]
        ss = SampleSet.load(samples_for(arm_name, source))
        mdp = VehicleRoutingMDP(action_encoding=arm.encoding)
        for lr in lrs:
            for seed in seeds:
                key = f"{arm_name}_lr{lr:g}_s{seed}"
                rp = os.path.join(outdir, f"{key}.json")
                if os.path.exists(rp):
                    print(f"--- {key}: exists, skipping")
                    continue
                print(f"\n--- {key}")
                # Seed the GLOBAL rng before building: `Net.build` draws weight
                # init from it (dynaplex/nn.py:404 vs :408), so without this the
                # screen would compare initialisations, not learning rates.
                torch.manual_seed(seed)
                t0 = time.time()
                points, hist = train_with_snapshots(
                    ss, network=arm.net(), ladder=[args.epochs],
                    features=arm.featurizer, lr=lr, seed=seed, verbose=False)
                elapsed = time.time() - t0
                pt = points[-1]
                share, _ = degeneracy(pt.agent, mdp, arm.featurizer)
                print(f"    {elapsed:.0f}s  best_val {pt.best_val_loss:.4f} "
                      f"@ epoch {pt.best_epoch}  regret {pt.val_regret:.4f}  "
                      f"dominance {share:.1%}")
                trained.setdefault(arm_name, {})[key] = (pt.agent, {
                    "arm": arm_name, "lr": lr, "seed": seed,
                    "rung_epochs": args.epochs,
                    "train_seconds": elapsed,
                    "seconds_per_epoch": elapsed / args.epochs,
                    "best_val_loss": pt.best_val_loss,
                    "best_epoch": pt.best_epoch,
                    "val_regret": pt.val_regret,
                    "degeneracy_share": share,
                    "history_train_loss": hist.train_loss,
                    "history_val_loss": hist.val_loss,
                    "history_val_regret": hist.val_regret,
                    "encoding": arm.encoding.name,
                    "source_budget": source.name,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                })

    if not trained:
        print("\nnothing new to evaluate.")
        return

    # One comparer call per encoding, all that encoding's screened policies at once
    # plus the anchors, paired on common random numbers -- same protocol as
    # run_benchmark, so myopic's returns remain the splice probe.
    by_enc: dict[str, list] = {}
    for arm_name, d in trained.items():
        by_enc.setdefault(ALL_ARMS[arm_name].encoding.name, []).extend(d.items())

    for enc_name, entries in by_enc.items():
        mdp = VehicleRoutingMDP(
            action_encoding=next(a.encoding for a in ALL_ARMS.values()
                                 if a.encoding.name == enc_name))
        print(f"\n=== evaluating {enc_name}: {len(entries)} policies + 2 anchors")
        comparer = dp.PolicyComparer(mdp, seed=COMPARER_SEED, workers=workers,
                                     **source.comparer)
        policies = {"myopic": MyopicPolicy(mdp), "random": dp.RandomPolicy(mdp)}
        for key, (agent, _) in entries:
            policies[key] = agent
        t0 = time.time()
        res = comparer.compare(policies, benchmark=BENCHMARK)
        print(f"    ({time.time() - t0:.0f}s)\n{res}")
        anchors = {n: {"mean": float(res[n].mean),
                       "returns": [float(x) for x in res[n].returns]}
                   for n in ("myopic", "random")}
        for key, (_, meta) in entries:
            a = res[key]
            meta.update({"mean": float(a.mean), "error": float(a.error),
                         "delta_mean": float(a.delta_mean),
                         "returns": [float(x) for x in a.returns],
                         "anchors": anchors,
                         "comparer": source.comparer,
                         "comparer_seed": COMPARER_SEED})
            with open(os.path.join(outdir, f"{key}.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1)

    print(f"\ndone -> {outdir}")
    print("Now: python analyze_lr_screen.py")


if __name__ == "__main__":
    main()
