"""Train a policy with PPO from scratch, then score it against the myopic and
random benchmarks.

This is the reinforcement-learning route to the same model: no rollout labels,
the policy improves from its own experience. On this model PPO is the harder
road: it has to discover that driving toward orders pays off before it can
refine anything. So treat this script as an illustration of the API, and
train_dcl_architectures.py as the route that produces strong policies.

The script trains the GIN. The same calls for the other two architectures are
below it, commented out: only the featurizer, the network and the model's
action representation change.

PPO also trains a critic (see critic_networks.py). For the flat feature vector
DynaPlex builds one itself; the GIN and the move tokens need their own.

    .venv/bin/python train_ppo.py
"""
import dynaplex as dp

from featurizer_flat import VehicleFeaturizer
from featurizer_graph import GridGraphFeaturizer
from featurizer_moves import MoveTokenFeaturizer
from mdp import ActionEncoding, MyopicPolicy, VehicleRoutingMDP

config = dp.PPOConfig(minibatch_size=4096, total_timesteps=500_000, seed=0)

# The GIN scores grid cells.
mdp = VehicleRoutingMDP(action_encoding=ActionEncoding.CELL)
ppo = dp.PPO(
    mdp, features=GridGraphFeaturizer, config=config,
    network=dp.Net("networks.gin_factory", d_model=64, n_layers=6),
    critic_network=dp.Net("critic_networks.gin_critic_factory", d_model=64, n_layers=6))

# The flat MLP scores directions. No network = DynaPlex's default MLP and critic.
# mdp = VehicleRoutingMDP(action_encoding=ActionEncoding.DIRECTIONAL)
# ppo = dp.PPO(mdp, features=VehicleFeaturizer, config=config)

# The move tokens score directions.
# mdp = VehicleRoutingMDP(action_encoding=ActionEncoding.DIRECTIONAL)
# ppo = dp.PPO(
#     mdp, features=MoveTokenFeaturizer, config=config,
#     network=dp.Net("networks.move_token_factory", d_model=64, n_layers=2, n_heads=4),
#     critic_network=dp.Net("critic_networks.move_token_critic_factory",
#                           d_model=64, n_layers=2, n_heads=4))

agent = ppo.train()

comparer = dp.PolicyComparer(
    mdp, number_of_trajectories=256, warmup_time=100, horizon=500, seed=42)
print()
print(comparer.compare(
    {"myopic": MyopicPolicy(mdp),
     "random": dp.RandomPolicy(mdp),
     "trained": agent},
    benchmark="myopic"))
