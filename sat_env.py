"""Vectorized CPython environment for the SAT pipeline (Phase 1 spike).

Ported from dynaplex2_playground `algorithms/sat_vector_env.py`. Manages N
parallel MDP instances in lock-step, stepping them in place and handing the
caller one `SequenceBuilder` seq dict per env.

This is the ONE file in the port surface that touches DynaPlex, and it is
CPython-only: the engine's own collection path (DCL) and NN evaluation harness
are compiled-only, so nothing here feeds them. It exists to answer the Phase 1
question — does the AGILE architecture do anything on this MDP — cheaply, before
the engine bridge is built. Numbers it produces are RATIOS against a benchmark
run through this same loop, never absolutes comparable to the engine's.

Divergences from upstream, all forced by dynaplex 1.12.2 / this MDP:

  * `TrajectoryContext` is built by `make_context(mdp, seed, sample, trajectory)`,
    not by calling the constructor. Upstream passed three fields; 1.12.2's
    dataclass has five (`rng, policy_rng, cumulative_cost, time_elapsed, valid`),
    so the old call is a TypeError. The factory also derives each generator its
    own `combine_seeds` stream and sizes `ctx.valid` to `[1, num_actions]`, which
    is what makes `mdp.write_action_validity(state, ctx.valid)` usable here.
  * The per-env `np.random.default_rng([seed, i])` generators are GONE. DynaPlex
    MDPs draw through `context.rng`, a `dynaplex.Generator` (xoshiro256++) whose
    draws are bit-identical in CPython and compiled code. A numpy Generator is a
    different stream entirely, so seeding one here would have produced
    trajectories that no engine run could reproduce.
  * TRUNCATION is now first-class. VRP is `HorizonType.INFINITE` and
    `_advance_cursor` only ever sets AWAIT_ACTION/AWAIT_EVENT, so
    `category == FINAL` is unreachable: upstream's `done` would be False forever,
    auto-reset would never fire, and GAE would see one infinite episode. `step`
    therefore returns `truncs` alongside `dones` and hands back the pre-reset
    observation in `final_seqs` so the trainer can bootstrap `gamma * V(s_final)`
    instead of treating the cut as terminal.
  * Callback mode, `_get_action_root_features`, and the `_start_idx`/`_date_idx`
    metadata (a foreign MDP's out-of-sample hooks) are removed, along with the
    `try/except` around `resolve_action` that would have masked a broken
    group -> action mapping as a fallback path.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from sat_compat import make_context
from dynaplex.modelling import StateCategory
from sat_actions import resolve_action
from sat_tokenizer import SequenceBuilder


def _clone_state_shallow(state: Any) -> Any:
    """Fast state clone: shallow copy + explicit array/list duplication.

    Only used when `reward_shaping` needs a pre-action snapshot. Lists of
    dataclasses are copied element-by-element so later in-place mutation of the
    originals does not reach the snapshot; much cheaper than `copy.deepcopy`.
    """
    new = copy.copy(state)  # shallow — gives us a new outer object
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        for f in dataclasses.fields(state):
            v = getattr(state, f.name)
            if isinstance(v, np.ndarray):
                object.__setattr__(new, f.name, v.copy())
            elif isinstance(v, list):
                if v and dataclasses.is_dataclass(v[0]):
                    object.__setattr__(new, f.name, [copy.copy(item) for item in v])
                else:
                    object.__setattr__(new, f.name, list(v))
    return new


class SATVectorEnv:
    """N parallel MDPs stepped in lock-step, emitting SAT seq dicts.

    Parameters
    ----------
    mdp :
        The MDP instance. Stepped in place via `modify_state_with_action` /
        `modify_state_with_event`.
    make_state :
        ``(ctx) -> state``. Normally `mdp.get_initial_state`. Takes the
        TrajectoryContext (not a bare rng) because that is the MDP contract and
        because the context owns the draw stream.
    make_action_set :
        ``(state) -> SATObservation``. Normally
        ``lambda s: build_sat_observation(s, mdp)``.
    seq_builder :
        The `SequenceBuilder`. Its type registry closes on the first `build`, so
        it must see a representative observation first (any AWAIT_ACTION state
        works — `build_sat_observation` populates every token type on every
        state).
    num_envs :
        Number of parallel environments.
    seed :
        Base experiment seed. Env *i* gets the `combine_seeds` coordinate
        ``(seed, sample=i, trajectory=<episode counter>)``, so streams never
        overlap across envs or across an env's successive episodes.
    max_steps :
        Decision steps before an episode is TRUNCATED (not terminated). Required
        for an infinite-horizon MDP.
    reward_shaping :
        Optional ``(prev_state, state, action_key, base_reward) -> reward``.
    """

    def __init__(
        self,
        mdp: Any,
        make_state: Callable[[Any], Any],
        make_action_set: Callable[[Any], Any],
        seq_builder: SequenceBuilder,
        num_envs: int = 1,
        seed: int = 0,
        max_steps: int = 200,
        reward_shaping: Optional[Callable[[Any, Any, Any, float], float]] = None,
    ):
        if seq_builder is None:
            raise ValueError("SATVectorEnv requires `seq_builder`.")
        if mdp is None:
            raise ValueError("SATVectorEnv requires `mdp`.")

        self.mdp = mdp
        self.make_state = make_state
        self.make_action_set = make_action_set
        self.seq_builder = seq_builder
        self.num_envs = num_envs
        self.seed = seed
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping

        self._has_event = hasattr(mdp, "modify_state_with_event")

        self.states: List[Any] = [None] * num_envs
        self.contexts: List[Any] = [None] * num_envs
        # per-env episode counter -> the `trajectory` seed coordinate, so a
        # restarted env draws a fresh non-overlapping stream rather than
        # replaying its first episode
        self._episode: List[int] = [0] * num_envs
        self._steps: List[int] = [0] * num_envs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _start_episode(self, i: int) -> None:
        """(Re)start env *i* at a fresh seed coordinate and advance to a decision."""
        self.contexts[i] = make_context(self.mdp, self.seed, sample=i,
                                        trajectory=self._episode[i])
        self.states[i] = self.make_state(self.contexts[i])
        self._steps[i] = 0
        self._episode[i] += 1
        self._advance_to_action(i)

    def _advance_to_action(self, i: int) -> None:
        """Run events until env *i* owes a decision (or has terminated)."""
        if not self._has_event:
            return
        state = self.states[i]
        while state.category == StateCategory.AWAIT_EVENT:
            self.mdp.modify_state_with_event(state, self.contexts[i])

    def _observe(self, i: int) -> Dict[str, Any]:
        """Build env *i*'s seq dict, plus the action-validity mask.

        The mask is attached as `_action_valid` (bool tensor over action groups,
        i.e. over action indices) rather than folded into the tokens. It has to
        stay a separate object: the action tokens are written complete so that
        logit position == action index, so SOMETHING downstream must crop them,
        and on the engine side that something is this same mask
        (`write_action_validity` -> masked CE / masked argmax). Keeping the split
        here means the Phase 1 spike trains against the engine's action space.

        `ctx.valid` is the context's reusable [1, num_actions] scratch, fully
        overwritten by each call, so this copies out of it.
        """
        state = self.states[i]
        seq = self.seq_builder.build(self.make_action_set(state))
        if state.category == StateCategory.AWAIT_ACTION:
            self.mdp.write_action_validity(state, self.contexts[i].valid)
            seq["_action_valid"] = torch.from_numpy(
                np.asarray(self.contexts[i].valid.arr[0], dtype=np.bool_).copy())
        return seq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> List[Dict[str, Any]]:
        """Reset **all** environments and return initial sequence dicts."""
        seq_dicts: List[Dict[str, Any]] = []
        for i in range(self.num_envs):
            self._start_episode(i)
            seq_dicts.append(self._observe(i))
        return seq_dicts

    def step(
        self,
        action_indices: List[int],
        current_seq_dicts: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray,
               List[Optional[Dict[str, Any]]]]:
        """Step all environments with the chosen action-group indices.

        Parameters
        ----------
        action_indices :
            Per-env group index chosen by the policy. Because
            `build_sat_observation` writes the action list complete and in
            action-index order, group g IS action g; `resolve_action` reads the
            `ActionKey` field to confirm it rather than assuming it.
        current_seq_dicts :
            The seq dicts the actions were chosen from.

        Returns
        -------
        next_seq_dicts :
            One per env; post-reset when the episode ended or was truncated.
        rewards : ndarray[num_envs] float32
            ``-(per-step cost)``.
        dones : ndarray[num_envs] bool
            TERMINAL (`category == FINAL`). Always False for infinite-horizon
            VRP — see `truncs`.
        truncs : ndarray[num_envs] bool
            Episode cut at `max_steps`. The successor state is still valuable,
            so the trainer must bootstrap rather than zero the return.
        final_seqs :
            The observation reached BEFORE the auto-reset, for exactly the envs
            where `dones | truncs`; None elsewhere. This is what the bootstrap
            value is computed from.
        """
        n = self.num_envs
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=np.bool_)
        truncs = np.zeros(n, dtype=np.bool_)
        next_seq_dicts: List[Dict[str, Any]] = [{}] * n
        final_seqs: List[Optional[Dict[str, Any]]] = [None] * n

        for i in range(n):
            seq = current_seq_dicts[i]
            if seq["num_action_groups"] == 0:
                # nothing to decide — leave the env untouched and let the
                # trainer's validity handling skip the slot
                next_seq_dicts[i] = seq
                continue

            action_key = resolve_action(seq["_actions"], action_indices[i])

            state = self.states[i]
            ctx = self.contexts[i]
            prev_cost = ctx.cumulative_cost
            prev_state = (_clone_state_shallow(state)
                          if self.reward_shaping is not None else None)

            self.mdp.modify_state_with_action(state, ctx, action_key)
            self._advance_to_action(i)

            reward = -(ctx.cumulative_cost - prev_cost)
            if self.reward_shaping is not None:
                reward = self.reward_shaping(prev_state, state, action_key, reward)
            rewards[i] = reward

            self._steps[i] += 1
            done = state.category == StateCategory.FINAL
            trunc = (not done) and self._steps[i] >= self.max_steps
            dones[i] = done
            truncs[i] = trunc

            if done or trunc:
                # the observation that ends the segment, captured BEFORE reset:
                # on truncation its value bootstraps the return
                final_seqs[i] = self._observe(i)
                self._start_episode(i)

            next_seq_dicts[i] = self._observe(i)

        return next_seq_dicts, rewards, dones, truncs, final_seqs
