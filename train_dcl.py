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

Runs are idempotent: DCL stores samples and agents under dynaplex_runs/ keyed
by the model, budget and network, so re-running with the same settings loads
the stored result instead of recomputing it.

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
dcl = dynaplex.DCL(mdp, MyopicPolicy(mdp), features=FEATURES, network=NETWORK,
                   n=N, m=M, h=H, seed=0, train=dict(epochs=200, patience=60, lr=1e-3))
agent = dcl.run(generations=1)[-1]

comparer = dynaplex.PolicyComparer(
    mdp, number_of_trajectories=256, warmup_time=100, horizon=500, seed=42)
print()
print(comparer.compare({"myopic": MyopicPolicy(mdp), "random": dynaplex.RandomPolicy(mdp),
                        "trained": agent}, benchmark="myopic"))
