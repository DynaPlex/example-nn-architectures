"""Ablation A: does the inter-action attention earn its keep, or is it the features?

THE QUESTION. `run_sat.py` reached 2.826 against myopic's 3.096, beating the
order-token attention baseline's 3.324. But that comparison changed two things at
once:

  1. FEATURES — SAT gives every candidate action its own feature vector,
     including `nearest_order_dist` (the myopic benchmark's own decision
     statistic) and `has_order` for the target cell. The older `attention_net`
     had no per-action features at all: it read the focal vehicle's output vector
     and emitted 5 logits from an MLP.
  2. ARCHITECTURE — one token per action plus AGILE's complete-graph attention
     over [state ; actions], so an action's representation depends on which
     others are available.

So what is established is "SAT's featurization AND AGILE beats the baseline", not
"AGILE beats the baseline". This script isolates (2).

THE DESIGN. Both arms use a byte-identical `SATFeaturizer`, so they can share ONE
sample set, and the only difference is what sits between the action nodes and the
scoring head:

  * `inter_action="attention"` — AGILE. Action nodes attend to each other and to
    the state node.
  * `inter_action="mlp"` — the CAPACITY-MATCHED control. Same depth, same FFN
    width, same residuals and norms; the attention sublayer removed, so each
    action is scored from its own features and the pooled state only.
    `test_networks_sat.test_mlp_ablation_removes_cross_action_conditioning`
    asserts action 0's logit is *bitwise* unchanged when action 4 is perturbed.

Simply deleting the stack (`inter_action="none"`) would remove conditioning AND
capacity at once, so a gap could be either; that is why the control is "mlp".

Three things are held identical between the arms, which is most of the value here:

  * the SAMPLES — collected once, then copied into the second arm's workdir, so
    both classifiers fit byte-identical states and labels and sampling noise
    cancels exactly (DCL prints "loading stored samples_gen0" for arm 2);
  * the TRAINING SEED — `train_network(seed=...)` governs the shuffle and the
    validation split, so both see the same data order and the same held-out rows;
  * the EVALUATION — a single `PolicyComparer.compare` call scores both agents
    plus the benchmarks, paired on common random numbers, so episode *i* is the
    same episode for every policy.

WHAT IT CAN AND CANNOT SETTLE. A large gap is decisive: the attention earns its
keep. A small gap is NOT decisive at one seed per arm, because the two nets have
different parameter counts and therefore consume weight-init RNG differently —
CRN pairing removes evaluation noise, not training noise. Use `--seeds 3` if the
gap comes out small; the playground's own notes report architecture differences
landing within seed noise on its benchmark suite.

Usage:
    python run_sat_ablation.py --quick    # ~2 min, checks the wiring
    python run_sat_ablation.py           # ~1h45m on 16 cores
    python run_sat_ablation.py --seeds 3 # if the gap is small
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
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

from featurizers_sat import SATFeaturizer
from mdp import ActionEncoding, MyopicPolicy, VehicleRoutingMDP

ARMS = ("attention", "mlp")


def degeneracy(agent, mdp, n_states: int = 60, seed: int = 5) -> tuple[float, list]:
    """(dominant-action share, histogram) of an agent's greedy choices.

    A guard against reading a meaningless comparison. An undertrained net has
    near-flat logits, so its argmax collapses onto one arbitrary action and it
    behaves as a CONSTANT policy — cost then climbs to holding_cost*max_orders
    regardless of architecture. Two such arms can even collapse onto the SAME
    action and produce byte-identical evaluations, which looks like a bug in the
    harness but is really "the budget was too small to compare anything".
    Anything above ~0.9 here means the arm is degenerate and its score says
    nothing about the architecture.
    """
    ctx = make_context(mdp, seed, eval=True)
    state = mdp.get_initial_state(ctx)
    states = []
    while len(states) < n_states:
        states.append(copy.deepcopy(state))
        mdp.modify_state_with_action(state, ctx, 0)
        while state.category.name == "AWAIT_EVENT":
            mdp.modify_state_with_event(state, ctx)

    spec = feature_spec(SATFeaturizer, mdp)
    arrays = allocate_arrays(spec, len(states))
    fz = featurizer_template(SATFeaturizer, mdp)
    fz = fz.bind(holder_from_featurizer(SATFeaturizer)(**arrays))
    mask = np.zeros((len(states), mdp.num_actions), dtype=np.bool_)
    from dynaplex.modelling.features import ValidityWriter
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget; checks the wiring, not the architecture")
    ap.add_argument("--seeds", type=int, default=1,
                    help="training seeds per arm; >1 when the gap is small "
                         "enough that init noise could explain it")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--root", default="dynaplex_runs_sat_ablation")
    args = ap.parse_args()

    workers = args.workers or multiprocessing.cpu_count()
    mdp = VehicleRoutingMDP(action_encoding=ActionEncoding.DIRECTIONAL)

    if args.quick:
        budget = dict(n=1000, m=25, h=15)
        train = dict(epochs=3, batch_size=64, lr=1e-4)
        cmp_kw = dict(number_of_trajectories=64, warmup_time=50, horizon=200)
    else:
        # matches run_sat.py's default budget so the "attention" arm is directly
        # comparable to the 2.826 already on record
        budget = dict(n=10000, m=500, h=100)
        train = dict(epochs=40, batch_size=64, lr=1e-4, patience=8)
        cmp_kw = dict(number_of_trajectories=128, warmup_time=100, horizon=500)

    print(f"=== Ablation A: inter-action attention vs capacity-matched MLP")
    print(f"    workers={workers}  budget={budget}  seeds={args.seeds}")
    print(f"    ~{budget['n'] * budget['m'] * budget['h']:,} rollout periods "
          f"(collected ONCE, shared by both arms)")

    # The workdir name carries the BUDGET. This matters: passing `workdir=`
    # bypasses DCL's structural hashing ("you then own uniqueness"), and DCL
    # resumes from whatever artifacts it finds there — so a --quick run followed
    # by a full run at the same path would silently train on the quick run's
    # 1,000 stored samples while reporting the full budget. Encoding n/m/h in the
    # path makes that collision impossible.
    tag = f"n{budget['n']}m{budget['m']}h{budget['h']}"

    def workdir(arm: str, seed: int) -> str:
        return os.path.join(args.root, tag, f"{arm}_seed{seed}")

    agents: dict[str, object] = {}
    samples_src: str | None = None

    for seed in range(args.seeds):
        for arm in ARMS:
            wd = workdir(arm, seed)
            # Reuse the one collected sample set. DCL resumes from stored
            # artifacts, so dropping samples_gen0 into a fresh workdir makes it
            # skip collection and only train — which is what keeps the arms
            # paired on identical labels.
            if samples_src is not None:
                dst = os.path.join(wd, "samples_gen0")
                if not os.path.exists(dst):
                    os.makedirs(wd, exist_ok=True)
                    shutil.copytree(samples_src, dst)

            print(f"\n--- arm={arm} seed={seed} -> {wd}")
            # Seed torch's GLOBAL rng here, not just via train(seed=...).
            # `train_network` builds the module BEFORE calling
            # torch.manual_seed(seed) (dynaplex/nn.py: `module = network.build(...)`
            # on line ~52, `torch.manual_seed(seed)` on ~56), so weight
            # INITIALISATION is not covered by that seed — it draws from whatever
            # global state the process happens to be in. Without this line two
            # identical invocations of this script gave different results for the
            # same arm, which would have quietly turned an architecture
            # comparison into a measurement of init luck.
            torch.manual_seed(seed)
            d = dp.DCL(
                mdp, dp.RandomPolicy(mdp), features=SATFeaturizer, workdir=wd,
                network=dp.Net("networks_sat.sat_agile_net", d_model=64,
                               n_layers=2, n_heads=4, inter_action=arm),
                train={**train, "seed": seed},
                **budget)
            t0 = time.time()
            trained = d.run(generations=1)
            print(f"    {time.time() - t0:.1f}s")
            agents[f"{arm}_s{seed}"] = trained[-1]

            if samples_src is None:
                samples_src = os.path.join(wd, "samples_gen0")

    # Degeneracy check BEFORE the comparer, so a meaningless table is labelled
    # as such rather than read as an architecture result.
    print("\n=== degeneracy check (greedy action histogram over 60 states)")
    suspect = []
    for name, ag in agents.items():
        share, hist = degeneracy(ag, mdp)
        flag = "  <-- DEGENERATE, score is uninformative" if share > 0.9 else ""
        print(f"    {name:16s} dominant {share:5.1%}  hist={hist}{flag}")
        if share > 0.9:
            suspect.append(name)
    if suspect:
        print(f"\n  WARNING: {suspect} collapsed onto ~one action. Their scores "
              f"reflect\n  budget, not architecture — increase epochs/n before "
              f"drawing any conclusion.")

    # ONE comparer call: every policy on the same episodes, paired on CRN
    print("\n=== evaluating all arms together (paired on common random numbers)")
    comparer = dp.PolicyComparer(mdp, seed=42, workers=workers, **cmp_kw)
    policies = {"myopic": MyopicPolicy(mdp), "random": dp.RandomPolicy(mdp)}
    policies.update(agents)
    print()
    print(comparer.compare(policies, benchmark="myopic"))

    print("\nHow to read this. 'attention_*' is AGILE; 'mlp_*' is the same net "
          "with\ncross-action conditioning removed and FFN capacity held equal. "
          "A large\ngap credits the architecture; a small gap says the win is "
          "the per-action\nFEATURES (nearest_order_dist / has_order), not the "
          "relational layer —\nand at one seed per arm a small gap is not "
          "decisive, since the two nets\nconsume weight-init RNG differently. "
          "Reference: run_sat.py's attention arm\nscored 2.826 vs myopic 3.096; "
          "the order-token baseline was 3.324.")


if __name__ == "__main__":
    main()
