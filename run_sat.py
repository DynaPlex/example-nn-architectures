"""Phase 2a: DCL on the structured-action transformer, through the engine.

The twin of `train.py --net attention` / `--net pointer`, using the same budget
so the architectures compare directly:

  * train.py --net attention  order tokens (variable) + focal-vehicle moves5 head,
                           DIRECTIONAL                     -> 3.324 vs myopic 3.096
  * train.py --net pointer    all 100 locations as tokens, shared per-cell scoring
                           head, CELL
  * this run               SAT/AGILE: typed context tokens + one token per
                           ACTION, complete-graph attention over
                           [state ; actions], DIRECTIONAL

Why this exists at all — the Phase 1 spike (`run_sat_spike.py`) ran the same
architecture in CPython at ~19 timesteps/s and never trained a usable policy.
Here the simulator is compiled: gen-0 collection sustains ~2.2M rollout
periods/s on 8 laptop workers, ~25M on a Snellius Genoa node. That is the whole
point of the engine bridge.

DCL is also a different learning problem from the spike's PPO, which matters for
interpreting the result: it labels each collected state with the best action
found by sequential-halving rollouts and trains a classifier on those labels, so
there is no exploration problem to collapse into. The spike's failure said
nothing about the architecture; this run is the one that can.

Usage:
    python run_sat.py --quick     # ~3 min laptop pipes check
    python run_sat.py             # laptop-scale learning run (~20 min)
    python run_sat.py --paper     # the run_attention budget (needs a cluster)
"""
from __future__ import annotations

import argparse
import multiprocessing
import time

import dynaplex as dp

from featurizers_sat import SATFeaturizer
from mdp import ActionEncoding, MyopicPolicy, VehicleRoutingMDP

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true",
                help="tiny budget; checks the pipes, not the policy")
ap.add_argument("--paper", action="store_true",
                help="the paper budget of train.py (n=10000, m=2000, h=100) — "
                     "sized for a 192-core node, not a laptop")
ap.add_argument("--encoding", choices=["directional", "cell"],
                default="directional",
                help="DIRECTIONAL writes 5 action tokens; CELL writes G*G, "
                     "which is 20x the action-side cost for actions that are "
                     "inadmissible 95%% of the time")
ap.add_argument("--generations", type=int, default=1,
                help="gen 1 rolls out the heuristic; later gens roll out the "
                     "previously trained net (the batched-inference path)")
ap.add_argument("--workers", type=int, default=0,
                help="engine worker worlds; 0 = all logical cores")
args = ap.parse_args()

enc = (ActionEncoding.DIRECTIONAL if args.encoding == "directional"
       else ActionEncoding.CELL)
mdp = VehicleRoutingMDP(action_encoding=enc)
workers = args.workers or multiprocessing.cpu_count()

if args.quick:
    budget = dict(n=2000, m=50, h=25)
    train = dict(epochs=5, batch_size=64, lr=1e-4)
    cmp_kw = dict(number_of_trajectories=64, warmup_time=50, horizon=200)
elif args.paper:
    # identical to `train.py --net pointer` at paper scale so the numbers are comparable
    budget = dict(n=10000, m=2000, h=100)
    train = dict(epochs=200, batch_size=64, lr=1e-4, patience=15)
    cmp_kw = dict(number_of_trajectories=256, warmup_time=100, horizon=500)
else:
    # Laptop-scale: the paper's n and h with a quarter of its rollouts per
    # sample. The split is deliberate — at ~2.7M rollout periods/s on 16 cores,
    # collection is nearly free (5e8 periods is ~3 min) and the CPU transformer
    # training is now the bottleneck (~48 s/epoch at n=10000), which is the
    # opposite of the Phase 1 spike's problem. So spend on label quality (m) and
    # cap the epochs.
    budget = dict(n=10000, m=500, h=100)
    train = dict(epochs=40, batch_size=64, lr=1e-4, patience=8)
    cmp_kw = dict(number_of_trajectories=128, warmup_time=100, horizon=500)

periods = budget["n"] * budget["m"] * budget["h"]
print(f"=== SAT/AGILE DCL | {enc.name} | num_actions={mdp.num_actions} "
      f"| workers={workers}")
print(f"    budget {budget} x {args.generations} generation(s)")
print(f"    ~{periods:,} rollout periods per generation")

d = dp.DCL(mdp, dp.RandomPolicy(mdp), features=SATFeaturizer,
           seed=0,
           network=dp.Net("networks_sat.sat_agile_net", d_model=64,
                          n_layers=2, n_heads=4),
           train=train, root="dynaplex_runs_sat",
           **budget)

t0 = time.time()
agents = d.run(generations=args.generations)
elapsed = time.time() - t0
print(f"\ntrained {len(agents)} agent(s) in {elapsed:.1f}s "
      f"(~{periods * args.generations / elapsed:,.0f} rollout periods/s "
      f"including training)")

comparer = dp.PolicyComparer(mdp, seed=42, workers=workers, **cmp_kw)
policies = {"myopic": MyopicPolicy(mdp), "random": dp.RandomPolicy(mdp)}
policies.update({f"agent_gen{i}": a for i, a in enumerate(agents, start=1)})
print()
print(comparer.compare(policies, benchmark="myopic"))
print("\nreference (engine comparer, 256 trajectories): myopic 3.096 | "
      "flat MLP 4.07 | order-token attention 3.324")
