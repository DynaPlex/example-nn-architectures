"""Train every architecture with one DCL generation at the SAME budget and
seed, then score them all in one PolicyComparer table against myopic.

Same seed means the same states are visited and labelled identically for every
arm (features never steer generation-0 collection), so the only difference
between the rows is the featurizer/network pair. Runs are idempotent per arm:
DCL resumes from its stored samples and agents, so re-running after adding an
arm is free for the existing ones.

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
    agents = {}
    for name, (features, network, train) in ARMS.items():
        print(f"\n==== {name} ====")
        dcl = dynaplex.DCL(mdp, MyopicPolicy(mdp), features=features, network=network,
                           n=N, m=M, h=H, seed=SEED, train={**TRAIN, **train})
        agents[name] = dcl.run(generations=1)[-1]
    comparer = dynaplex.PolicyComparer(
        mdp, number_of_trajectories=256, warmup_time=100, horizon=500, seed=42)
    policies = {"myopic": MyopicPolicy(mdp), "random": dynaplex.RandomPolicy(mdp)}
    policies.update(agents)
    print()
    print(comparer.compare(policies, benchmark="myopic"))


if __name__ == "__main__":
    main()
