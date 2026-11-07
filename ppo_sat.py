"""Local PPO trial of the SAT/AGILE policy on the engine.

The twin of `ppo_attention.py`, and the second half of the Phase 2a story: DCL
(`run_sat.py`) is the primary path, but PPO exercises the same featurizer and
network through a completely different consumer, which is worth having for two
reasons.

First, it is the only thing that runs `networks_sat.sat_agile_critic` in anger.
PPO's built-in value head is single-tensor (`ppo.py:183-205`) and the SAT
observation is a bundle, so a critic must be supplied; here it is the same trunk
read off the state node after it has attended over the available actions.

Second, it is where an integer-dtype bug would actually bite. PPO's rollout
buffer allocates EVERY spec tensor with `torch.zeros(...)` and no dtype
(`dynaplex/algorithms/ppo.py:404`), so the int64 `orders_count` companion arrives
here as float32 while DCL delivers it as int64. `networks_sat` casts count
tensors with `.to(torch.long)` for exactly this reason, and
`test_networks_sat.test_orders_count_dtype_does_not_change_the_output` pins it —
but a live PPO run is the end-to-end proof.

Note the contrast with the Phase 1 spike, which was also PPO and failed to train:
that was CPython at ~19 timesteps/s. This is the compiled simulator behind a
vectorized env, so the batch budget below is ~7x the spike's ENTIRE run.

Sized for a laptop trial (~10-20 min CPU), not a learning run. gamma/ent_coef
follow the earlier DVRP PPO tuning that `ppo_attention.py` uses, so the two are
comparable.

    .venv/bin/python ppo_sat.py
"""
import dynaplex as dp

from featurizers_sat import SATFeaturizer
from mdp import ActionEncoding, VehicleRoutingMDP

# DIRECTIONAL: 5 action tokens rather than G*G, so the action side of the
# attention is 20x cheaper for actions that are inadmissible 95% of the time.
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

ppo = dp.PPO(mdp, features=SATFeaturizer,
             network=dp.Net("networks_sat.sat_agile_net", d_model=64,
                            n_layers=2, n_heads=4),
             critic_network=dp.Net("networks_sat.sat_agile_critic", d_model=64,
                                   n_layers=2, n_heads=4),
             config=config, root="dynaplex_runs_sat")

agent = ppo.train()
print("\nbest agent:", ppo.workdir)
