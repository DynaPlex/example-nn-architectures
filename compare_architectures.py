"""Train every architecture on the SAME collected states, then score them all
in one PolicyComparer table against myopic.

The states are collected once, under the myopic policy, and stored. Collecting
does not involve a featurizer, so every arm is trained on exactly those states
and labels: the only difference between the rows is the featurizer/network
pair. Runs are idempotent: the collection and each arm's agent are stored, so
re-running after adding an arm costs that arm's training only.

    .venv/bin/python compare_architectures.py
"""
import dynaplex
from dynaplex.nn import Net

from featurizer_flat import VehicleFeaturizer
from featurizer_moves import MoveTokenFeaturizer
from featurizer_graph import GridGraphFeaturizer
from mdp import MyopicPolicy, VehicleRoutingMDP

# ---- the budget (one generation; gen-0 rollouts by the myopic policy) ------
N, M, H = 10000, 500, 100          # labelled states, rollouts per action, horizon
SEED = 0
TRAIN = dict(epochs=200, patience=60)

# ---- the arms: featurizer, network (None = DynaPlex's default MLP), and ----
# ---- training overrides on top of TRAIN (the GIN wants a higher lr) --------
ARMS = {
    "flat_mlp": (VehicleFeaturizer, None, {}),
    "gin": (GridGraphFeaturizer,
            Net("networks.gin_net", d_model=64, n_layers=6),
            dict(lr=1e-3)),
    "moves": (MoveTokenFeaturizer,
              Net("networks.move_token_net", d_model=64, n_layers=2, n_heads=4), {}),
    "moves_independent": (MoveTokenFeaturizer,
                          Net("networks.move_token_net", d_model=64, n_layers=2,
                              n_heads=4, interact=False), {}),
}


def main() -> None:
    mdp = VehicleRoutingMDP()
    dcl = dynaplex.DCL(mdp, MyopicPolicy(mdp), n=N, m=M, h=H, seed=SEED)
    dcl.collect_gen0()                  # the one collection every arm trains on
    agents = {}
    for name, (features, network, train) in ARMS.items():
        print(f"\n==== {name} ====")
        agents[name] = dcl.train_policy(features=features, network=network,
                                        train={**TRAIN, **train})
    comparer = dynaplex.PolicyComparer(
        mdp, number_of_trajectories=256, warmup_time=100, horizon=500, seed=42)
    policies = {"myopic": MyopicPolicy(mdp), "random": dynaplex.RandomPolicy(mdp)}
    policies.update(agents)
    print()
    print(comparer.compare(policies, benchmark="myopic"))


if __name__ == "__main__":
    main()
