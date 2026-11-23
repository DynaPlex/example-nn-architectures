"""Train every architecture with Deep Controlled Learning, then score each
against the myopic policy.

Deep Controlled Learning: collect states under the myopic policy, label each
with the best move found by rollouts, and train a network to imitate those
labels.

The model names a vehicle's move in one of two ways (README, "The action
representation"): as the grid cell it goes to, or as a direction. The GIN
scores cells, the flat MLP and the move-token network score directions. A
collection belongs to one model, so there are two: one per representation.
Collecting does not involve a featurizer, so the flat MLP and the move-token
network are trained on exactly the same labelled states.

Runs are idempotent: the collection and each agent are stored under
dynaplex_runs/, so running the script again loads what is there, and adding
an architecture costs that architecture's training only.

    .venv/bin/python train_dcl_architectures.py
"""
import dynaplex as dp

from featurizer_flat import VehicleFeaturizer
from featurizer_graph import GridGraphFeaturizer
from featurizer_moves import MoveTokenFeaturizer
from mdp import ActionEncoding, MyopicPolicy, VehicleRoutingMDP

# ---- actions as grid cells: the GIN --------------------------------------
cell_mdp = VehicleRoutingMDP(action_encoding=ActionEncoding.CELL)

# Collect: 10 000 labelled states, 500 rollouts of 100 periods per move.
# torch_threads=1: these networks are small, and on a laptop CPU one thread trains
# them faster than ten (the threads cost more to coordinate than they save).
cell_dcl = dp.DCL(cell_mdp, MyopicPolicy(cell_mdp),
                  n=10000, m=500, h=100, seed=0, torch_threads=1)
cell_dcl.collect_gen0()

gin = cell_dcl.train_policy(
    features=GridGraphFeaturizer,
    network=dp.Net("networks.gin_factory", d_model=64, n_layers=6),
    train=dict(epochs=200, patience=60, lr=1e-3))

# ---- actions as directions: the flat MLP and the move tokens -------------
move_mdp = VehicleRoutingMDP(action_encoding=ActionEncoding.DIRECTIONAL)

move_dcl = dp.DCL(move_mdp, MyopicPolicy(move_mdp),
                  n=10000, m=500, h=100, seed=0, torch_threads=1)
move_dcl.collect_gen0()

# No network = DynaPlex's default MLP.
flat_mlp = move_dcl.train_policy(
    features=VehicleFeaturizer,
    train=dict(epochs=200, patience=60))

moves = move_dcl.train_policy(
    features=MoveTokenFeaturizer,
    network=dp.Net("networks.move_token_factory", d_model=64, n_layers=2, n_heads=4),
    train=dict(epochs=200, patience=60))

# A second generation collects again, now with the trained agent in the place of
# myopic, and trains on that. `run` takes the arguments of `train_policy` and
# returns the agents of all generations; what is stored already is loaded:
#     gin, gin_gen2 = cell_dcl.run(2, features=..., network=..., train=...)
# It costs far more than the first (every rollout step evaluates the network, and
# each architecture needs its own collection): work for a GPU or a cluster.

# ---- compare --------------------------------------------------------------
# A comparer works on one model, so there is one table per representation. Both
# use the same seed and the dynamics are the same, so both tables are about the
# same episodes: the myopic row is identical. Lower cost is better.
cell_comparer = dp.PolicyComparer(
    cell_mdp, number_of_trajectories=256, warmup_time=100, horizon=500, seed=42)
print()
print(cell_comparer.compare(
    {"myopic": MyopicPolicy(cell_mdp),
     "random": dp.RandomPolicy(cell_mdp),
     "gin": gin},
    benchmark="myopic"))

move_comparer = dp.PolicyComparer(
    move_mdp, number_of_trajectories=256, warmup_time=100, horizon=500, seed=42)
print()
print(move_comparer.compare(
    {"myopic": MyopicPolicy(move_mdp),
     "random": dp.RandomPolicy(move_mdp),
     "flat_mlp": flat_mlp,
     "moves": moves},
    benchmark="myopic"))
