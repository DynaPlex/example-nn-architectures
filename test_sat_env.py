"""Tests for the CPython vector env and the trainer's action masking (Phase 1).

Two things here are worth more than the rest of the Phase 1 tests combined,
because both were live bugs found while porting:

  * ACTION VALIDITY. The action tokens are written complete, so an unmasked
    softmax spans inadmissible actions. Upstream never masked, because its MDPs
    enumerated only admissible actions. Sampling an off-grid move here would
    walk a vehicle out of the grid.
  * TRUNCATION. VRP is infinite-horizon and never reaches
    `StateCategory.FINAL`, so upstream's `done` flag was False forever: no
    auto-reset, one endless episode, and GAE with nothing to bootstrap.
"""
import numpy as np
import pytest
import torch

from sat_compat import make_context
from dynaplex.modelling import StateCategory

from mdp import ActionEncoding, VehicleRoutingMDP
from sat_env import SATVectorEnv
from sat_schema import build_sat_observation
from sat_tokenizer import SequenceBuilder
from sat_trainer import _flat_to_padded, _padded_valid


def _mdp(encoding=ActionEncoding.DIRECTIONAL):
    return VehicleRoutingMDP(grid_size=6, n_vehicles=3, capacity=3,
                            max_orders=12, action_encoding=encoding)


def _env(mdp, num_envs=4, max_steps=10, seed=0):
    sb = SequenceBuilder(skip_root=True)
    sb.build(build_sat_observation(
        mdp.get_initial_state(make_context(mdp, seed)), mdp))
    return SATVectorEnv(
        mdp=mdp, make_state=mdp.get_initial_state,
        make_action_set=lambda s: build_sat_observation(s, mdp),
        seq_builder=sb, num_envs=num_envs, seed=seed, max_steps=max_steps)


def _greedy_valid(env, seqs):
    """Pick the first admissible action per env, using the attached mask."""
    return [int(np.flatnonzero(s["_action_valid"].numpy())[0]) for s in seqs]


# ---------------------------------------------------------------------------
# Action validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding",
                         [ActionEncoding.DIRECTIONAL, ActionEncoding.CELL])
def test_observations_carry_a_validity_mask_over_action_indices(encoding):
    mdp = _mdp(encoding)
    env = _env(mdp)
    for seq in env.reset():
        m = seq["_action_valid"]
        assert m.dtype == torch.bool
        assert m.shape[0] == mdp.num_actions == seq["num_action_groups"]
        assert m.any(), "every VRP state admits at least 'stay'"


def test_mask_matches_write_action_validity_exactly():
    """The mask must be the MDP's own, not a reconstruction."""
    mdp = _mdp()
    env = _env(mdp)
    seqs = env.reset()
    ctx = make_context(mdp, 123)
    for i, seq in enumerate(seqs):
        mdp.write_action_validity(env.states[i], ctx.valid)
        want = np.asarray(ctx.valid.arr[0], dtype=np.bool_)
        assert np.array_equal(seq["_action_valid"].numpy(), want)


def test_mask_is_copied_not_aliased_to_the_context_scratch():
    """`ctx.valid` is a REUSED [1, num_actions] buffer, fully overwritten on
    every call. Aliasing it would make every stored observation in the rollout
    silently share the newest state's mask — and the PPO update would then
    optimise against a mask that never applied."""
    mdp = _mdp()
    env = _env(mdp)
    seqs = env.reset()
    before = [s["_action_valid"].clone() for s in seqs]

    # force the scratch to be rewritten for a different state
    for i in range(env.num_envs):
        mdp.write_action_validity(env.states[(i + 1) % env.num_envs],
                                  env.contexts[i].valid)
    for s, b in zip(seqs, before):
        assert torch.equal(s["_action_valid"], b)


@pytest.mark.parametrize("encoding",
                         [ActionEncoding.DIRECTIONAL, ActionEncoding.CELL])
def test_masked_sampling_never_selects_an_inadmissible_action(encoding):
    """The end-to-end guard: sample the way the trainer does, 200 times, and
    assert every drawn action is admissible.

    Uses deliberately extreme logits so an unmasked sampler would almost surely
    pick an invalid action — without the mask this test fails immediately.
    """
    mdp = _mdp(encoding)
    env = _env(mdp, num_envs=4, max_steps=50)
    seqs = env.reset()
    gen = torch.Generator().manual_seed(0)

    for _ in range(50):
        B, A = len(seqs), mdp.num_actions
        flat = torch.randn(B * A, generator=gen) * 20.0
        offsets = torch.arange(B) * A
        lengths = torch.full((B,), A, dtype=torch.long)
        logits = _flat_to_padded(flat, offsets, lengths)
        valid = _padded_valid(seqs, logits.shape[1])
        probs = torch.softmax(logits.masked_fill(~valid, -1e9), dim=1)
        acts = torch.multinomial(probs, 1, generator=gen).squeeze(1).tolist()

        for i, a in enumerate(acts):
            assert seqs[i]["_action_valid"][a], (
                f"env {i}: sampled inadmissible action {a}")
        seqs, _r, _d, _t, _f = env.step(acts, seqs)


def test_padded_valid_falls_back_to_all_groups_without_a_mask():
    """A seq dict with no `_action_valid` must not silently mask everything
    out — that would make its softmax uniform over -1e9 and its gradients junk."""
    fake = [{"num_action_groups": 3}, {"num_action_groups": 5}]
    v = _padded_valid(fake, 5)
    assert v[0].tolist() == [True, True, True, False, False]
    assert v[1].tolist() == [True] * 5


def test_vehicles_stay_on_the_grid_under_masked_play():
    """The consequence the mask exists to prevent, asserted on the state."""
    mdp = _mdp()
    env = _env(mdp, num_envs=3, max_steps=40)
    seqs = env.reset()
    for _ in range(60):
        seqs, _r, _d, _t, _f = env.step(_greedy_valid(env, seqs), seqs)
        for state in env.states:
            for veh in state.vehicles:
                assert 0 <= veh.row < mdp.grid_size
                assert 0 <= veh.col < mdp.grid_size
            assert state.vehicle_count.sum() == mdp.n_vehicles


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation_fires_at_max_steps_and_terminal_never_does():
    mdp = _mdp()
    max_steps, n = 10, 4
    env = _env(mdp, num_envs=n, max_steps=max_steps)
    seqs = env.reset()

    n_trunc = 0
    steps = 30
    for t in range(steps):
        seqs, _r, dones, truncs, finals = env.step(_greedy_valid(env, seqs), seqs)
        assert not dones.any(), "VRP is infinite-horizon; FINAL is unreachable"
        if truncs.any():
            assert (t + 1) % max_steps == 0
            n_trunc += int(truncs.sum())
            # a truncated env must hand back the observation it was cut at
            for i in np.flatnonzero(truncs):
                assert finals[i] is not None
            for i in np.flatnonzero(~truncs):
                assert finals[i] is None
    assert n_trunc == n * (steps // max_steps)


def test_truncation_resets_the_episode_and_the_step_counter():
    mdp = _mdp()
    env = _env(mdp, num_envs=2, max_steps=5)
    seqs = env.reset()
    episodes_before = list(env._episode)
    for _ in range(5):
        seqs, _r, _d, truncs, _f = env.step(_greedy_valid(env, seqs), seqs)
    assert truncs.all()
    assert env._steps == [0, 0]
    assert env._episode == [e + 1 for e in episodes_before]


def test_final_observation_is_a_usable_seq_dict():
    """The bootstrap runs the value net on it, so it must be a complete
    observation — not a stub."""
    mdp = _mdp()
    env = _env(mdp, num_envs=2, max_steps=3)
    seqs = env.reset()
    for _ in range(3):
        seqs, _r, _d, truncs, finals = env.step(_greedy_valid(env, seqs), seqs)
    assert truncs.all()
    for f in finals:
        assert f is not None
        assert f["num_action_groups"] == mdp.num_actions
        assert set(f["type_features"]) <= {ts for ts in seqs[0]["type_features"]} | {
            "OrderTok"}
        assert torch.isfinite(f["type_features"]["VehicleTok"]).all()


def test_post_reset_observation_differs_from_the_final_one():
    """`step` must return the NEW episode's observation, not the cut one."""
    mdp = _mdp()
    env = _env(mdp, num_envs=2, max_steps=3)
    seqs = env.reset()
    for _ in range(3):
        seqs, _r, _d, truncs, finals = env.step(_greedy_valid(env, seqs), seqs)
    assert truncs.all()
    # a fresh initial state has exactly the configured initial order count
    for nxt in seqs:
        assert nxt["type_features"]["OrderTok"].shape[0] == min(30, mdp.max_orders)


# ---------------------------------------------------------------------------
# Seeding / determinism
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_the_same_rollout():
    mdp = _mdp()
    def rollout():
        env = _env(mdp, num_envs=3, max_steps=8, seed=7)
        seqs = env.reset()
        rs = []
        for _ in range(20):
            seqs, r, _d, _t, _f = env.step(_greedy_valid(env, seqs), seqs)
            rs.append(r.copy())
        return np.stack(rs)
    assert np.array_equal(rollout(), rollout())


def test_parallel_envs_get_independent_streams():
    """Env i uses the `sample=i` seed coordinate, so two envs must not be
    running the same trajectory (which would silently cut the effective batch)."""
    mdp = _mdp()
    env = _env(mdp, num_envs=4, max_steps=50, seed=3)
    env.reset()
    fingerprints = {
        tuple((v.row, v.col, v.capacity_left) for v in s.vehicles) +
        (int(s.has_order.sum()),)
        for s in env.states
    }
    assert len(fingerprints) > 1


def test_successive_episodes_of_one_env_differ():
    """The `trajectory` coordinate advances per episode, so a restarted env must
    not replay its first episode."""
    mdp = _mdp()
    env = _env(mdp, num_envs=1, max_steps=4)
    seqs = env.reset()
    first = env.states[0].has_order.copy()
    for _ in range(4):
        seqs, _r, _d, truncs, _f = env.step(_greedy_valid(env, seqs), seqs)
    assert truncs.all()
    assert not np.array_equal(env.states[0].has_order, first)


def test_env_uses_the_dynaplex_generator_not_numpy():
    """DynaPlex MDPs must draw through `ctx.rng` (xoshiro256++), whose draws are
    bit-identical in CPython and compiled code. The port dropped upstream's
    per-env `np.random.default_rng`, which was a different stream entirely and
    would have made these trajectories unreproducible on the engine."""
    mdp = _mdp()
    env = _env(mdp)
    env.reset()
    assert not hasattr(env, "rngs")
    for ctx in env.contexts:
        assert type(ctx.rng).__module__.startswith("dynaplex")
        assert ctx.valid.arr.shape == (1, mdp.num_actions)
