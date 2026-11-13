"""Train a policy with Deep Controlled Learning: collect states under the
myopic policy, label each with the best move found by rollouts, and train a
network to imitate those labels. One generation; ends with a comparer table
against the myopic and random benchmarks on common random numbers.

Pick the architecture by editing the two lines under "the architecture". Each
featurizer/network pair in this repository is documented in its own module:

    featurizer_flat.VehicleFeaturizer              flat count maps  -> default MLP
    featurizer_graph.GridGraphFeaturizer      grid graph       -> networks.gin_net
    featurizer_moves.MoveTokenFeaturizer candidate tokens -> networks.move_token_net
    featurizer_cells.GridTokenFeaturizer     cell tokens      -> networks.attention_pointer_net

Collecting and training are separate. The collection (the states and their
labels) depends on the model and the budget only, not on the architecture, so
it is made once and every featurizer/network pair you try is trained on the
same stored states. Runs are idempotent: DCL keeps the collection and the
agents under dynaplex_runs/, so re-running with the same settings loads the
stored result instead of recomputing it.

    .venv/bin/python train_dcl.py
"""
import dynaplex
from dynaplex.nn import Net

from featurizer_graph import GridGraphFeaturizer
from mdp import MyopicPolicy, VehicleRoutingMDP

# the architecture
FEATURES = GridGraphFeaturizer
NETWORK = Net("networks.gin_net", d_model=64, n_layers=6)

# the budget: labelled states, rollouts per candidate action, rollout horizon
N, M, H = 10000, 500, 100

mdp = VehicleRoutingMDP()
dcl = dynaplex.DCL(mdp, MyopicPolicy(mdp), n=N, m=M, h=H, seed=0)
dcl.collect_gen0()                      # once; skipped when the collection is already there
agent = dcl.train_policy(features=FEATURES, network=NETWORK,
                         train=dict(epochs=200, patience=60, lr=1e-3))

comparer = dynaplex.PolicyComparer(
    mdp, number_of_trajectories=256, warmup_time=100, horizon=500, seed=42)
print()
print(comparer.compare({"myopic": MyopicPolicy(mdp), "random": dynaplex.RandomPolicy(mdp),
                        "trained": agent}, benchmark="myopic"))
