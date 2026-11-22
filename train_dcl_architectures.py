"""Train every architecture with Deep Controlled Learning on the SAME collected
states, then score them all in one comparison table against the myopic policy.

Deep Controlled Learning: collect states under the myopic policy, label each
with the best move found by rollouts, and train a network to imitate those
labels. Collecting does not involve a featurizer, so the states are collected
and labelled once and every architecture is trained on exactly those states:
the only difference between the rows of the table is the featurizer/network
pair.

Runs are idempotent: the collection and each agent are stored under
dynaplex_runs/, so running the script again loads what is there, and adding
an architecture costs that architecture's training only.

    .venv/bin/python train_dcl_architectures.py
"""
import dynaplex as dp

from featurizer_flat import VehicleFeaturizer
from featurizer_graph import GridGraphFeaturizer
from featurizer_moves import MoveTokenFeaturizer
from mdp import MyopicPolicy, VehicleRoutingMDP

mdp = VehicleRoutingMDP()
myopic = MyopicPolicy(mdp)

# Collect once: 10 000 labelled states, 500 rollouts of 100 periods per move.
# torch_threads=1: these networks are small, and on a laptop CPU one thread trains
# them faster than ten (the threads cost more to coordinate than they save).
dcl = dp.DCL(mdp, myopic, n=10000, m=500, h=100, seed=0, torch_threads=1)
dcl.collect_gen0()

# Train each architecture on that collection. No network = DynaPlex's default MLP.
flat_mlp = dcl.train_policy(
    features=VehicleFeaturizer,
    train=dict(epochs=200, patience=60))

gin = dcl.train_policy(
    features=GridGraphFeaturizer,
    network=dp.Net("networks.gin_factory", d_model=64, n_layers=6),
    train=dict(epochs=200, patience=60, lr=1e-3))

moves = dcl.train_policy(
    features=MoveTokenFeaturizer,
    network=dp.Net("networks.move_token_factory", d_model=64, n_layers=2, n_heads=4),
    train=dict(epochs=200, patience=60))

# Score everything on the same episodes (common random numbers); lower is better.
comparer = dp.PolicyComparer(
    mdp, number_of_trajectories=256, warmup_time=100, horizon=500, seed=42)
print()
print(comparer.compare(
    {"myopic": myopic,
     "random": dp.RandomPolicy(mdp),
     "flat_mlp": flat_mlp,
     "gin": gin,
     "moves": moves},
    benchmark="myopic"))
