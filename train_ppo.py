"""Train a policy with PPO from scratch on the engine-backed vector
environment, then score it against the myopic and random benchmarks.

Uses the flat count-map features and DynaPlex's default MLP. This is the
reinforcement-learning route to the same model: no rollout labels, the policy
improves from its own experience. On this model PPO is the harder road — it
has to discover that walking toward orders pays off before it can refine
anything — so treat this script as the API illustration and train_dcl.py as
the route that produces strong policies.

    .venv/bin/python train_ppo.py
"""
import dynaplex as dp

from featurizer_flat import VehicleFeaturizer
from mdp import MyopicPolicy, VehicleRoutingMDP

mdp = VehicleRoutingMDP()
config = dp.PPOConfig(minibatch_size=4096, total_timesteps=500_000, seed=0)
ppo = dp.PPO(mdp, features=VehicleFeaturizer, config=config)
agent = ppo.train()

comparer = dp.PolicyComparer(
    mdp, number_of_trajectories=256, warmup_time=100, horizon=500, seed=42)
print()
print(comparer.compare({"myopic": MyopicPolicy(mdp), "random": dp.RandomPolicy(mdp),
                        "trained": agent}, benchmark="myopic"))
