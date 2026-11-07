"""The head-to-head benchmark: SAT/AGILE vs the two original attention nets.

WHY THIS EXISTS. `README.md` compares SAT's 2.826 against `attention_net`'s
3.324 and myopic's 3.096. Three things make that comparison unreadable, none of
which depend on whether its conclusion is right:

  1. THE BUDGETS DIFFER. 2.826 is n=10000/m=500/h=100, 40 epochs, 128 comparer
     trajectories (run_sat.py:81-83). 3.324 is m=2000, 200 epochs, 256
     trajectories (the paper budget in train.py). Four knobs move at once.
  2. THE STATISTICAL UNIT IS WRONG. The quoted +/-0.021 is the comparer's
     TRAJECTORY standard error. But the same SAT arm has scored 2.435, 2.663 and
     2.826 across three runs -- seed sd ~ 0.2, about TEN TIMES the evaluation SE.
     The thing being compared is an architecture, so the unit has to be the
     training seed. Every baseline number on record is a single seed, and
     `attention_net`'s seed variance has never been measured at all.
  3. `attention_pointer_net` HAS NO PUBLISHED NUMBER anywhere in the repo.

So: all three arms, re-run at ONE budget with ONE recipe, over many seeds, with
the raw per-trajectory returns persisted so the statistics are recomputable
without re-running anything. `analyze_benchmark.py` does the testing; this script
only produces evidence and refuses to interpret it.

WHAT IS HELD IDENTICAL: the MDP, the DCL budget, the training recipe, the
transformer size knobs, the collection seed, the comparer config and seed. What
varies: the (featurizer, network, encoding) triple, and the training seed.

Usage:
    python run_benchmark.py --budget quick --seeds 2    # ~5 min, wiring only
    python run_benchmark.py --seeds 8                   # the real run, overnight
    python run_benchmark.py --seeds 12                  # extends the above; the
                                                        # first 8 are not recomputed
    python analyze_benchmark.py
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import time

import numpy as np
import torch

import dynaplex as dp
from sat_compat import make_context
from dynaplex.modelling import (
    allocate_arrays,
    feature_spec,
    featurizer_template,
    holder_from_featurizer,
)
from dynaplex.modelling.features import ValidityWriter

from benchmark_config import (
    ALL_ARMS,
    BENCHMARK,
    BUDGETS,
    COMPARER_SEED,
    DCL_SEED,
    DEFAULT_ARMS,
    DEGENERACY_LIMIT,
    RESULTS_ROOT,
    WORKDIR_ROOT,
    Arm,
    Budget,
)
from mdp import MyopicPolicy, VehicleRoutingMDP

SCHEMA = "benchmark-result-v1"


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def degeneracy(agent, mdp, featurizer: type, n_states: int = 60,
               seed: int = 5) -> tuple[float, list[int]]:
    """(dominant-action share, histogram) of an agent's greedy choices.

    Adapted from `run_sat_ablation.py:84-121`, generalised over the featurizer,
    and with the probe's stepping policy FIXED -- see below.

    An undertrained net has near-flat logits, so its argmax collapses onto one
    arbitrary action and it behaves as a CONSTANT policy -- cost then climbs to
    holding_cost*max_orders regardless of architecture. Two such arms can even
    collapse onto the SAME action and evaluate byte-identically, which looks like
    a harness bug but is really "the budget was too small to compare anything".

    THE PROBE MUST WALK A REPRESENTATIVE STATE DISTRIBUTION, which the original
    did not. It stepped with a hardcoded action 0: under DIRECTIONAL that is
    "stay", so the fleet never moved, and under CELL it is cell (0,0), which is
    additionally INADMISSIBLE unless the vehicle is beside the depot (mdp.py:203).
    Either way the 60 probe states covered only ~5 distinct focal positions, so
    the dominance figure described the probe rather than the network -- visible in
    the tell that eight independently initialised nets all scored exactly 91.7%.
    Stepping with a uniformly random ADMISSIBLE action instead matches
    `dp.RandomPolicy`, which is the same distribution DCL collects generation 0
    from, so the states the probe scores are the states the net was trained on.

    NOTE the deepcopy: the MDP mutates `State` IN PLACE, so `states.append(state)`
    inside a stepping loop yields N aliases of the final state
    (MODULE_MAP.md:255-261).
    """
    ctx = make_context(mdp, seed, eval=True)
    rng = np.random.default_rng(seed)
    state = mdp.get_initial_state(ctx)
    step_mask = np.zeros((1, mdp.num_actions), dtype=np.bool_)
    step_vw = ValidityWriter(step_mask, 0)
    states = []
    while len(states) < n_states:
        states.append(copy.deepcopy(state))
        step_vw.reset(0)
        step_mask[0, :] = False
        mdp.write_action_validity(state, step_vw)
        admissible = np.flatnonzero(step_mask[0])
        assert admissible.size, "no admissible action in an AWAIT_ACTION state"
        mdp.modify_state_with_action(state, ctx, int(rng.choice(admissible)))
        while state.category.name == "AWAIT_EVENT":
            mdp.modify_state_with_event(state, ctx)

    spec = feature_spec(featurizer, mdp)
    arrays = allocate_arrays(spec, len(states))
    fz = featurizer_template(featurizer, mdp)
    fz = fz.bind(holder_from_featurizer(featurizer)(**arrays))
    mask = np.zeros((len(states), mdp.num_actions), dtype=np.bool_)
    vw = ValidityWriter(mask, 0)
    for i, st in enumerate(states):
        fz.reset(i)
        fz.write_features(st)
        fz.finish()
        vw.reset(i)
        mdp.write_action_validity(st, vw)

    acts = agent.act(arrays, mask)
    hist = np.bincount(acts, minlength=mdp.num_actions)
    return float(hist.max() / hist.sum()), hist.tolist()


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def sample_label_digest(workdir: str) -> dict | None:
    """A cheap fingerprint of a stored gen-0 sample set: how many samples, and a
    hash of the labels and the validity masks.

    Used by validity gate 3. `sat` and `tokens` collect under the same MDP with
    the same DCL seed, so they should see the SAME states with the SAME labels --
    only the features differ. If the digests match, those two arms are matched at
    the DATA level and sampling noise cancels between them; if not, they are only
    budget-matched and the write-up has to say so. (`pointer` is CELL-encoded, so
    its labels are cell indices and its digest is expected to differ.)
    """
    path = os.path.join(workdir, "samples_gen0")
    if not os.path.isdir(path):
        return None
    from dynaplex.sample_set import SampleSet
    s = SampleSet.load(path, mmap=True)
    return {
        "n": int(len(s)),
        "num_actions": int(s.mask.shape[1]),
        "y_sha": _sha(s.y),
        "mask_sha": _sha(s.mask),
    }


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def workdir_for(root: str, budget: Budget, arm: str, seed: int) -> str:
    """Budget-tagged, because passing `workdir=` bypasses DCL's structural
    hashing and it resumes from whatever it finds -- a --quick run followed by a
    full run at the same path would train on the quick run's samples while
    reporting the full budget (MODULE_MAP.md:243-246)."""
    return os.path.join(root, budget.tag, f"{arm}_seed{seed}")


def result_path(budget: Budget, arm: str, seed: int) -> str:
    return os.path.join(RESULTS_ROOT, budget.tag, f"{arm}_seed{seed}.json")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def _dcl(arm: Arm, budget: Budget, mdp, wd: str, seed: int, workers: int):
    return dp.DCL(mdp, dp.RandomPolicy(mdp), features=arm.featurizer, workdir=wd,
                  seed=DCL_SEED, network=arm.net(),
                  train={**budget.train, "seed": seed},
                  **budget.collect)


def collect_once(arm: Arm, budget: Budget, mdp, wd0: str,
                 workers: int) -> tuple[str, float | None]:
    """Collect this arm's ONE sample set into seed 0's workdir, timed on its own.

    Collection is a separate phase rather than the first half of seed 0's
    `DCL.run()` so that `train_seconds` means TRAINING for every seed. Otherwise
    seed 0 silently carries the collection cost and the per-arm training figure --
    the thing this benchmark is asked to report -- is not comparable across seeds
    or across arms.

    Returns (samples_dir, seconds) with seconds None when a stored set was reused.
    """
    dst = os.path.join(wd0, "samples_gen0")
    if os.path.isdir(dst):
        print(f"    [collect] reusing stored samples at {dst}")
        return dst, None
    d = _dcl(arm, budget, mdp, wd0, 0, workers)
    t0 = time.time()
    ss = d.collect(generation=0)
    ss.save(dst)
    elapsed = time.time() - t0
    periods = budget.collect["n"] * budget.collect["m"] * budget.collect["h"]
    print(f"    [collect] {len(ss)} samples in {elapsed:.1f}s "
          f"(~{periods / elapsed:,.0f} rollout periods/s)")
    return dst, elapsed


def train_arm_seed(arm: Arm, seed: int, budget: Budget, mdp, wd: str,
                   samples_src: str | None, workers: int) -> tuple[object, dict]:
    """Train one (arm, seed) and return (agent, metadata). Idempotent -- DCL
    reloads an existing generation instead of recomputing it.

    `samples_src` always exists by the time this is called (`collect_once` ran
    first), so `DCL.run` prints "loading stored samples_gen0" and the timing below
    is pure training. Copying the arm's one sample set into each seed's workdir
    also means every seed fits BYTE-IDENTICAL labels, so the seed sweep measures
    weight-init and shuffle variance on a fixed dataset -- which is the population
    an architecture comparison wants.
    """
    if samples_src is not None and os.path.isdir(samples_src):
        dst = os.path.join(wd, "samples_gen0")
        if not os.path.exists(dst):
            os.makedirs(wd, exist_ok=True)
            shutil.copytree(samples_src, dst)

    # Seed torch's GLOBAL rng HERE, not just via train(seed=...). `train_network`
    # builds the module BEFORE calling torch.manual_seed (dynaplex/nn.py:404 vs
    # :408), so weight INITIALISATION is not covered by that seed -- it draws from
    # whatever global state the process happens to be in. Without this line two
    # identical invocations give different results for the same arm, which would
    # quietly turn an architecture comparison into a measurement of init luck
    # (MODULE_MAP.md:236-242).
    torch.manual_seed(seed)

    d = _dcl(arm, budget, mdp, wd, seed, workers)
    t0 = time.time()
    agents = d.run(generations=1)
    elapsed = time.time() - t0
    agent = agents[-1]
    info = dict(agent.info)
    epochs_run = info.get("epochs_run") or 0
    meta = {
        "train_seconds": elapsed,
        # the size-independent training cost: epochs differ between arms because
        # early stopping fires at different points, so total seconds alone would
        # confound "slower per epoch" with "trained for longer"
        "seconds_per_epoch": (elapsed / epochs_run) if epochs_run else None,
        "epochs_run": info.get("epochs_run"),
        "epochs_max": info.get("epochs"),
        "stopped_early": info.get("stopped_early"),
        "best_val_loss": info.get("best_val_loss"),
        "val_regret": info.get("val_regret"),
        "final_train_loss": info.get("final_loss"),
        "num_parameters": int(sum(p.numel() for p in agent.module.parameters())),
    }
    return agent, meta


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", choices=sorted(BUDGETS), default="default")
    ap.add_argument("--seeds", type=int, default=8,
                    help="training seeds per arm; the statistical unit. Raising "
                         "it later re-uses everything already computed.")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma-separated; known: " + ",".join(ALL_ARMS))
    ap.add_argument("--workers", type=int, default=0, help="0 = all logical cores")
    ap.add_argument("--force", action="store_true",
                    help="recompute and overwrite existing result JSONs")
    ap.add_argument("--root", default=WORKDIR_ROOT)
    args = ap.parse_args()

    budget = BUDGETS[args.budget]
    workers = args.workers or multiprocessing.cpu_count()
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in ALL_ARMS]
    if unknown:
        ap.error("unknown arm(s) " + str(unknown) + "; known: " + str(sorted(ALL_ARMS)))
    arms = [ALL_ARMS[a] for a in arm_names]
    seeds = list(range(args.seeds))

    os.makedirs(os.path.join(RESULTS_ROOT, budget.tag), exist_ok=True)

    periods = budget.collect["n"] * budget.collect["m"] * budget.collect["h"]
    print(f"=== benchmark | budget={budget.name} ({budget.tag}) | workers={workers}")
    print(f"    arms  : {', '.join(a.name for a in arms)}")
    print(f"    seeds : {seeds}  ({len(arms) * len(seeds)} training runs)")
    print(f"    ~{periods:,} rollout periods per arm (collected ONCE, shared "
          f"across that arm's seeds)")
    print(f"    results -> {os.path.join(RESULTS_ROOT, budget.tag)}")
    for a in arms:
        print(f"      {a.name:11s} {a.encoding.name:12s} {a.blurb}")

    # ---- phase A: train ---------------------------------------------------
    # `pending` holds the trained agents that still need evaluating; arm-seeds
    # with a persisted result are skipped entirely, which is what makes --seeds
    # extendable without recomputing anything.
    pending: dict[tuple[str, int], object] = {}
    meta: dict[tuple[str, int], dict] = {}
    digests: dict[str, dict | None] = {}
    mdps: dict[str, object] = {}

    collect_seconds: dict[str, float | None] = {}

    for arm in arms:
        mdp = mdps.setdefault(arm.encoding.name,
                              VehicleRoutingMDP(action_encoding=arm.encoding))
        todo = [s for s in seeds
                if args.force or not os.path.exists(result_path(budget, arm.name, s))]
        wd0 = workdir_for(args.root, budget, arm.name, seeds[0])
        samples_src: str | None = os.path.join(wd0, "samples_gen0")
        if todo:
            print(f"\n=== {arm.name}: collecting once for {len(todo)} seed(s)")
            samples_src, collect_seconds[arm.name] = collect_once(
                arm, budget, mdp, wd0, workers)
        for seed in seeds:
            wd = workdir_for(args.root, budget, arm.name, seed)
            rp = result_path(budget, arm.name, seed)
            if os.path.exists(rp) and not args.force:
                print(f"\n--- {arm.name} seed={seed}: result exists, skipping ({rp})")
                continue
            print(f"\n--- {arm.name} seed={seed} -> {wd}")
            agent, m = train_arm_seed(arm, seed, budget, mdp, wd, samples_src,
                                      workers)
            m["collect_seconds"] = collect_seconds.get(arm.name)
            share, hist = degeneracy(agent, mdp, arm.featurizer)
            m["degeneracy_share"] = share
            m["degeneracy_hist"] = hist
            flag = "  <-- DEGENERATE" if share > DEGENERACY_LIMIT else ""
            stop = " (early stop)" if m["stopped_early"] else " (HIT CAP)"
            spe = m["seconds_per_epoch"]
            print(f"    train {m['train_seconds']:.1f}s"
                  f"  epochs {m['epochs_run']}/{m['epochs_max']}{stop}"
                  f"  {spe:.1f}s/epoch"
                  f"  val_regret={m['val_regret']}"
                  f"  greedy dominance {share:.1%}{flag}")
            pending[(arm.name, seed)] = agent
            meta[(arm.name, seed)] = m
        digests[arm.name] = sample_label_digest(
            workdir_for(args.root, budget, arm.name, seeds[0]))

    print("\n=== sample-set digests (validity gate 3: are the arms data-matched?)")
    for name, dg in digests.items():
        print(f"    {name:11s} {dg}")

    if not pending:
        print("\nnothing to evaluate; every requested arm-seed already has a "
              "result. Run analyze_benchmark.py.")
        return

    # ---- phase B: evaluate ------------------------------------------------
    # One comparer call per ENCODING (the CELL-encoded pointer arm needs its own
    # MDP). Both calls use the same seed and config, and myopic/random are in
    # every call -- analyze_benchmark.py checks their per-trajectory returns are
    # element-wise identical across calls, which is what licenses splicing the
    # two tables into one paired frame (validity gate 1).
    by_encoding: dict[str, list[tuple[str, int]]] = {}
    for key in pending:
        by_encoding.setdefault(ALL_ARMS[key[0]].encoding.name, []).append(key)

    env = {
        "schema": SCHEMA,
        "budget": budget.name,
        "budget_tag": budget.tag,
        "collect": budget.collect,
        "train": budget.train,
        "comparer": budget.comparer,
        "dcl_seed": DCL_SEED,
        "comparer_seed": COMPARER_SEED,
        "workers": workers,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }

    for enc_name, keys in by_encoding.items():
        mdp = mdps[enc_name]
        print(f"\n=== evaluating {enc_name} arms ({len(keys)} agents + 2 anchors), "
              f"paired on common random numbers")
        comparer = dp.PolicyComparer(mdp, seed=COMPARER_SEED, workers=workers,
                                     **budget.comparer)
        policies = {"myopic": MyopicPolicy(mdp), "random": dp.RandomPolicy(mdp)}
        for k in keys:
            policies[f"{k[0]}_s{k[1]}"] = pending[k]
        t0 = time.time()
        results = comparer.compare(policies, benchmark=BENCHMARK)
        print(f"    ({time.time() - t0:.1f}s)\n")
        print(results)

        anchors = {
            name: {
                "mean": float(results[name].mean),
                "error": float(results[name].error),
                "returns": [float(x) for x in results[name].returns],
            }
            for name in ("myopic", "random")
        }
        for (arm_name, seed) in keys:
            a = results[f"{arm_name}_s{seed}"]
            payload = dict(env)
            payload.update({
                "arm": arm_name,
                "seed": seed,
                "encoding": enc_name,
                "featurizer": (ALL_ARMS[arm_name].featurizer.__module__ + "."
                               + ALL_ARMS[arm_name].featurizer.__qualname__),
                "network": ALL_ARMS[arm_name].factory,
                "net_kwargs": ALL_ARMS[arm_name].net_kwargs,
                "sample_digest": digests.get(arm_name),
                "mean": float(a.mean),
                "error": float(a.error),
                "delta_mean": float(a.delta_mean),
                "delta_error": float(a.delta_error),
                "returns": [float(x) for x in a.returns],
                "anchors": anchors,
            })
            payload.update(meta[(arm_name, seed)])
            rp = result_path(budget, arm_name, seed)
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            with open(rp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
            print(f"    wrote {rp}")

        for name in ("myopic", "random"):
            got = float(results[name].mean)
            bounds = (budget.anchors or {}).get(name)
            if bounds is None:
                note = "(no reference range at this budget)"
            else:
                lo, hi = bounds
                note = "ok" if lo <= got <= hi else f"OUT OF RANGE [{lo}, {hi}]"
            print(f"    anchor {name:8s} {got:8.3f}  {note}")

    print(f"\ndone. Now: python analyze_benchmark.py --budget {budget.name}")
    print("This script deliberately prints no verdict -- the gates and the "
          "pre-registered tests live in analyze_benchmark.py.")


if __name__ == "__main__":
    main()
