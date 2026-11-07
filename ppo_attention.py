"""Local PPO trial of the dot-attention policy on the token bundle.

PPO's actor is `networks.attention_net` (DIRECTIONAL encoding: the moves5
head's 5 logits are the action logits); the critic must be supplied too,
because the built-in PPO value head is single-tensor and the token bundle is
not — `networks.attention_critic` is the same attention trunk reading the
focal vehicle's vector into a scalar value.

Sized for a LOCAL laptop trial (~10-20 min CPU): small batch budget, frequent
comparer evaluations against myopic on common random numbers. gamma/ent_coef
follow the earlier DVRP PPO tuning (gamma=0.998 per decision, entropy 0.03).

Run from the repo root with the repo venv:

    .venv/bin/python ppo_attention.py
"""
import dynaplex as dp
from dynaplex.nn import Net

from featurizers_tokens import TokenFeaturizer
from mdp import ActionEncoding, VehicleRoutingMDP

mdp = VehicleRoutingMDP(action_encoding=ActionEncoding.DIRECTIONAL)

config = dp.PPOConfig(
    minibatch_size=512,
    num_envs=128, num_steps=64,          # batch = 8192 env steps per update
    total_timesteps=200_000,             # ~24 updates — a trial, not a run
    gamma=0.998, ent_coef=0.03,          # earlier DVRP PPO tuning
    update_epochs=4, lr=2.5e-4,
    eval_interval=5, eval_trajectories=64,
    device="cpu",                        # try "mps" for larger configs
    seed=0,
)

ppo = dp.PPO(mdp, features=TokenFeaturizer,
             network=Net("networks.attention_net", d_model=64, n_layers=2,
                            n_heads=4, head="moves5"),
             critic_network=Net("networks.attention_critic", d_model=64,
                                   n_layers=2, n_heads=4),
             config=config, root="dynaplex_runs_tokens")

agent = ppo.train()
print("\nbest agent:", ppo.workdir)
