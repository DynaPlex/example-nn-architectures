"""Stage 2: does SAT reach a given policy quality in fewer epochs than the
original attention nets?

THE CLAIM. Not "SAT ends up better" -- the `long`-budget probe showed
`attention_net` reaches 2.303 given enough epochs, better than any SAT seed
measured at 40. The claim is about the RATE: SAT gets to a given quality sooner.

THE DESIGN. One training run per (arm, seed) to `max(ladder)` epochs, snapshotting
the deployable policy at every rung, then every snapshot scored in one paired
comparer call. `learning_curve.train_with_snapshots` is asserted bitwise equal to
`train_network(epochs=E, patience=0)` at each rung, so a rung is exactly the
policy a dedicated E-epoch run would have produced -- at roughly half the compute
of retraining per rung.

WHAT MAKES IT FAIR. Each arm runs at ITS OWN learning rate, chosen by
`run_lr_screen.py` from a common grid on common samples. Without that the
comparison measures how well `lr=1e-4` suits each architecture, and the screen
showed that matters enormously: `tokens` averages 13.21 at 1e-4 and 3.01 at 1e-3.
The learning rate is the convergence-speed knob, so for a SPEED claim it cannot be
held at someone else's value.

Note the selection rule is deliberately speed-oriented -- lr is picked for best
cost at a 40-epoch budget, not at the final rung. That is the same rule for every
arm, but it does mean a rate that wins slowly is not preferred, which is the
intent. The per-arm curves below show whether the chosen rate also holds up at the
top of the ladder.

Usage:
    python run_learning_curve.py --seeds 5 --lrs sat=1e-3,tokens=1e-3,pointer=3e-4
    python analyze_learning_curve.py
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

LADDER = (5, 10, 20, 40, 80, 160)
CURVE_TAG = "learning_curve"


def parse_lrs(spec: str) -> dict[str, float]:
    out = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        name, _, val = part.partition("=")
        out[name.strip()] = float(val)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--lrs", required=True,
                    help="per-arm learning rates from run_lr_screen.py, e.g. "
                         "sat=1e-3,tokens=1e-3,pointer=3e-4")
    ap.add_argument("--ladder", default=",".join(str(e) for e in LADDER))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--source", default="default",
                    help="budget whose collected samples to reuse")
    ap.add_argument("--tag", default=CURVE_TAG)
    args = ap.parse_args()

    workers = args.workers or multiprocessing.cpu_count()
    torch.set_num_threads(min(workers, 32))
    source = BUDGETS[args.source]
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    lrs = parse_lrs(args.lrs)
    missing = [a for a in arm_names if a not in lrs]
    if missing:
        ap.error(f"no learning rate given for {missing}; run run_lr_screen.py first")
    ladder = sorted(int(e) for e in args.ladder.split(","))
    seeds = list(range(args.seeds))
    outdir = os.path.join(RESULTS_ROOT, args.tag)
    os.makedirs(outdir, exist_ok=True)

    print(f"=== learning curves | arms={arm_names} | seeds={seeds}")
    print(f"    ladder {ladder}   lrs " + ", ".join(f"{a}={lrs[a]:g}" for a in arm_names))
    print(f"    {len(arm_names) * len(seeds)} runs x {max(ladder)} epochs, "
          f"{len(arm_names) * len(seeds) * len(ladder)} policies to score")

    pending: dict[str, list] = {}
    for arm_name in arm_names:
        arm = ALL_ARMS[arm_name]
        wd = workdir_for(WORKDIR_ROOT, source, arm_name, 0)
        spath = os.path.join(wd, "samples_gen0")
        if not os.path.isdir(spath):
            raise SystemExit(f"no collected samples at {spath}")
        ss = SampleSet.load(spath)
        mdp = VehicleRoutingMDP(action_encoding=arm.encoding)
        for seed in seeds:
            done = all(os.path.exists(os.path.join(
                outdir, f"{arm_name}_s{seed}_e{e}.json")) for e in ladder)
            if done:
                print(f"--- {arm_name} s{seed}: all rungs exist, skipping")
                continue
            print(f"\n--- {arm_name} s{seed}  lr={lrs[arm_name]:g}")
            # Global seed before build: Net.build draws init from ambient rng
            # (dynaplex/nn.py:404 vs :408).
            torch.manual_seed(seed)
            t0 = time.time()
            points, hist = train_with_snapshots(
                ss, network=arm.net(), ladder=ladder, features=arm.featurizer,
                lr=lrs[arm_name], seed=seed, verbose=True)
            elapsed = time.time() - t0
            print(f"    {elapsed:.0f}s total ({elapsed / max(ladder):.1f}s/epoch)")
            for p in points:
                share, _ = degeneracy(p.agent, mdp, arm.featurizer)
                key = f"{arm_name}_s{seed}_e{p.epochs}"
                pending.setdefault(arm.encoding.name, []).append((key, p.agent, {
                    "arm": arm_name, "seed": seed, "rung_epochs": p.epochs,
                    "lr": lrs[arm_name], "ladder": ladder,
                    "encoding": arm.encoding.name,
                    "best_val_loss": p.best_val_loss, "best_epoch": p.best_epoch,
                    "val_regret": p.val_regret,
                    "degeneracy_share": share,
                    "run_train_seconds": elapsed,
                    "seconds_per_epoch": elapsed / max(ladder),
                    # wall-clock actually spent to reach this rung
                    "seconds_to_rung": elapsed / max(ladder) * p.epochs,
                    "history_val_loss": hist.val_loss[:p.epochs],
                    "history_val_regret": hist.val_regret[:p.epochs],
                    "history_train_loss": hist.train_loss[:p.epochs],
                    "source_budget": source.name,
                    "comparer": source.comparer,
                    "comparer_seed": COMPARER_SEED,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                }))

    if not pending:
        print("\nnothing new to evaluate.")
        return

    for enc_name, entries in pending.items():
        mdp = VehicleRoutingMDP(
            action_encoding=next(a.encoding for a in ALL_ARMS.values()
                                 if a.encoding.name == enc_name))
        print(f"\n=== evaluating {enc_name}: {len(entries)} policies + 2 anchors")
        comparer = dp.PolicyComparer(mdp, seed=COMPARER_SEED, workers=workers,
                                     **source.comparer)
        policies = {"myopic": MyopicPolicy(mdp), "random": dp.RandomPolicy(mdp)}
        for key, agent, _ in entries:
            policies[key] = agent
        t0 = time.time()
        res = comparer.compare(policies, benchmark=BENCHMARK)
        print(f"    ({time.time() - t0:.0f}s)\n{res}")
        anchors = {n: {"mean": float(res[n].mean),
                       "returns": [float(x) for x in res[n].returns]}
                   for n in ("myopic", "random")}
        for key, _, meta in entries:
            a = res[key]
            meta.update({"mean": float(a.mean), "error": float(a.error),
                         "delta_mean": float(a.delta_mean),
                         "returns": [float(x) for x in a.returns],
                         "anchors": anchors})
            with open(os.path.join(outdir, f"{key}.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1)

    print(f"\ndone -> {outdir}\nNow: python analyze_learning_curve.py --tag {args.tag}")


if __name__ == "__main__":
    main()
