"""PPO trainer for the reference structured-action transformer (Phase 1 spike).

Ported from dynaplex2_playground `algorithms/sat_trainer.py`. Trains
`sat_net_ref`'s AGILE policy + value nets over `sat_env`'s CPython vector env,
with the observation tokenised by `sat_tokenizer` from the `sat_schema` layout.

Architecture it trains (all in `sat_net_ref`):

- per-type input projections + learned type embeddings
- a context self-attention encoder, mean-pooled to a state vector
- a block-diagonal within-group layer, then pooling to one node per action
  (this is what supports multi-token actions; VRP's are single-token, so the
  layer degenerates to a per-token MLP here)
- complete-graph attention over [state node ; action nodes] — the AGILE step
- intra-group positional embeddings for order-sensitive nested actions

PPO with GAE(λ), batched forward passes (padding + attention masks rather than
per-sample calls), and a vectorized rollout/update.

Scope. This is the SPIKE trainer, not the production path: DCL and the
comparer's NN harness are engine-only, and nothing here crosses into the engine.
Throughput is ~19 timesteps/s, bounded by per-observation dataclass walking and
per-sample embedding in Python — which is itself the argument for the Phase 2
engine bridge, where DCL fills 97,280 slots per round on 190 workers.

Returns ``(policy_net, value_net, seq_builder, episode_rewards)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from sat_tokenizer import SequenceBuilder
from sat_net_ref import (
    TransformerPolicyNet,
    TransformerValueNet,
    build_transformer_policy_net,
    build_transformer_value_net,
)
from sat_env import SATVectorEnv
from sat_actions import resolve_action
from sat_compat import make_context

__all__ = [
    "SATTrainerConfig",
    "train_sat_pointer_ppo",
    "select_action_sat",
    "_seq_to_action_probs",
    "_seq_to_value",
    "_get_action_root_features",
    "_get_all_action_root_features",
]


# ============================================================================
# Network forward helpers
# ============================================================================

def _seq_to_action_probs(
    policy_net: TransformerPolicyNet,
    seq: Dict[str, Any],
) -> torch.Tensor:
    """Run the policy net on a sequence dict and return action probabilities.

    WARNING — UNMASKED. This softmaxes over every action token, and the tokens
    span every action index including inadmissible ones (see
    `sat_schema.MoveAction`). Sampling or argmaxing this distribution WILL pick
    illegal actions: measured on VRP, 6% of draws under DIRECTIONAL and 92%
    under CELL. It is kept as the reference single-sample forward, matching
    `_seq_to_value`; anything that actually selects an action must apply the
    validity mask first — see `select_action_sat` and `_padded_valid`.
    """
    logits = policy_net.forward_logits(
        type_features=seq["type_features"],
        type_ids=seq["type_ids"],
        action_mask=seq["action_mask"],
        indices_by_type=seq["indices_by_type"],
        action_group_ids=seq.get("action_group_ids"),
        num_action_groups=seq.get("num_action_groups"),
        intra_group_pos=seq.get("intra_group_pos"),
    )
    return torch.softmax(logits, dim=0)


def _seq_to_value(
    value_net: TransformerValueNet,
    seq: Dict[str, Any],
) -> torch.Tensor:
    """Run the value net on a sequence dict and return scalar value.

    Mirrors the batched path (``TransformerValueNet.forward_batched``) used in
    the training loop: it must pass ``intra_group_pos`` so the action tokens get
    the same positional embeddings.  Omitting it (as an earlier version did)
    makes single-sample values diverge from the trained value function — see
    ``_seq_to_action_probs`` for the matching policy-side contract.
    """
    return value_net(
        type_features=seq["type_features"],
        type_ids=seq["type_ids"],
        indices_by_type=seq["indices_by_type"],
        action_mask=seq["action_mask"],
        intra_group_pos=seq.get("intra_group_pos"),
    )


# ============================================================================
# Action feature extraction helpers
# ============================================================================

def _get_action_root_features(seq: Dict[str, Any], action_idx: int) -> torch.Tensor:
    """Return the raw feature vector of the *root* token of action group ``action_idx``.

    When actions are flat (group size 1) this is the single action token.
    When actions contain nested children the root token is identified via
    ``is_group_root``.
    """
    # Fast path: use precomputed map (O(1) list lookup, no tensor ops)
    arm = seq.get("_action_root_map")
    if arm is not None:
        tname, local_i = arm[action_idx]
        return seq["type_features"][tname][local_i]
    # Fallback: tensor-based lookup (backward compat)
    group_root_indices = seq["is_group_root"].nonzero(as_tuple=True)[0]
    global_idx = group_root_indices[action_idx].item()
    tname = seq["token_type_names"][global_idx]
    local_idxs = seq["indices_by_type"][tname]
    # indices_by_type may be a Tensor or a list
    if isinstance(local_idxs, torch.Tensor):
        local_i = (local_idxs == global_idx).nonzero(as_tuple=True)[0][0].item()
    else:
        local_i = local_idxs.index(global_idx)
    return seq["type_features"][tname][local_i]


def _get_all_action_root_features(seq: Dict[str, Any]) -> List[torch.Tensor]:
    """Return list of root-token feature vectors for every action group."""
    n = seq["num_action_groups"]
    return [_get_action_root_features(seq, i) for i in range(n)]


# ============================================================================
# Vectorized helpers
# ============================================================================

def _flat_to_padded(
    flat_values: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    fill_value: float = -1e9,
) -> torch.Tensor:
    """Scatter concatenated per-sample values into a padded ``[B, max_L]`` matrix.

    Fully differentiable w.r.t. *flat_values* — autograd tracks the indexed
    assignment so gradients flow back through the scatter.

    Parameters
    ----------
    flat_values : Tensor[total]
        Concatenated values across all samples in the batch.
    offsets : LongTensor[B]
        Start index into *flat_values* for each sample.
    lengths : LongTensor[B]
        Number of values per sample.
    fill_value : float
        Value for padding positions (default ``-1e9`` for logits so that
        ``softmax`` maps padding to ≈ 0).

    Returns
    -------
    padded : Tensor[B, max_L]
    """
    B = offsets.shape[0]
    total = flat_values.shape[0]
    device = flat_values.device

    max_L = int(lengths.max().item()) if total > 0 else 1
    max_L = max(max_L, 1)

    padded = torch.full(
        (B, max_L), fill_value, device=device, dtype=flat_values.dtype,
    )

    if total > 0:
        batch_idx = torch.repeat_interleave(
            torch.arange(B, device=device), lengths)
        local_pos = (
            torch.arange(total, device=device)
            - torch.repeat_interleave(offsets, lengths)
        )
        padded[batch_idx, local_pos.long()] = flat_values

    return padded


def _padded_valid(seq_dicts: List[Dict[str, Any]], max_L: int) -> torch.Tensor:
    """Stack per-sample action-validity masks into ``[B, max_L]`` bool, False-padded.

    WHY THIS EXISTS. The action tokens are written COMPLETE — every action index
    gets a token, admissible or not (see sat_schema.MoveAction). That is
    deliberate: logit position must equal action index, and the network must emit
    finite logits everywhere so a NaN can never reach an argmax. The consequence
    is that the raw softmax puts probability on INADMISSIBLE actions, and
    sampling one would hand `modify_state_with_action` an off-grid move.

    Upstream never had to deal with this: its MDPs enumerated only admissible
    actions, so num_action_groups WAS the valid set. Here the mask is a separate,
    orthogonal object — exactly as on the engine side, where DCL applies it as
    `masked_fill(~mask, -1e9)` (dynaplex/nn.py:464) and act-time selection uses a
    masked argmax (nn.py:77-86). This helper is the Phase 1 equivalent, so the
    spike trains against the same action space the engine would.

    `_action_valid` is attached per observation by `SATVectorEnv._observe`.
    """
    B = len(seq_dicts)
    valid = torch.zeros(B, max_L, dtype=torch.bool)
    for i, seq in enumerate(seq_dicts):
        m = seq.get("_action_valid")
        if m is None:
            # no mask supplied: treat every present action group as admissible
            n = min(int(seq.get("num_action_groups", 0)), max_L)
            valid[i, :n] = True
        else:
            n = min(m.shape[0], max_L)
            valid[i, :n] = m[:n]
    return valid


def _init_orthogonal_(net: torch.nn.Module) -> None:
    """Orthogonally initialise every ``nn.Linear`` in *net* (PPO best practice).

    Hidden layers use gain ``sqrt(2)`` (ReLU); scalar output heads
    (``out_features == 1``: the policy logit head and the value head) use gain
    ``0.01`` so the initial policy is near-uniform and the initial value near
    zero.  Biases are zeroed.  Applied generically so it covers any policy-net
    factory (SAT, pointer, AGILE, Utility) plus the value net.

    ``nn.MultiheadAttention`` keeps its own (xavier) in-projection init — its
    weights are raw Parameters, not ``nn.Linear`` submodules, so they are left
    untouched here.
    """
    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            gain = 0.01 if module.out_features == 1 else math.sqrt(2.0)
            torch.nn.init.orthogonal_(module.weight, gain)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def _lr_scale(iteration: int, num_iterations: int, config: "SATTrainerConfig") -> float:
    """Multiplier in [0, 1] for the base LR at a given iteration.

    Linear warmup over the first ``lr_warmup_frac`` of iterations, then cosine
    decay from 1.0 down to ``lr_min_frac``.  Returns 1.0 when scheduling is off.
    """
    if not getattr(config, "lr_schedule", False):
        return 1.0
    warmup = max(1, int(config.lr_warmup_frac * num_iterations))
    if iteration < warmup:
        return float(iteration + 1) / float(warmup)
    progress = (iteration - warmup) / max(1, num_iterations - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.lr_min_frac + (1.0 - config.lr_min_frac) * cosine


def _entropy_scale(iteration: int, num_iterations: int, config: "SATTrainerConfig") -> float:
    """Multiplier in [0, 1] for the base entropy_coef at a given iteration.

    No warmup (full entropy pressure from iteration 0 — that's when premature
    collapse onto a degenerate action is most likely); cosine decay from 1.0
    down to ``entropy_min_frac`` over the run. Returns 1.0 when scheduling is off.
    """
    if not getattr(config, "entropy_coef_schedule", False):
        return 1.0
    progress = iteration / max(1, num_iterations - 1)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_frac = float(getattr(config, "entropy_min_frac", 0.1))
    return min_frac + (1.0 - min_frac) * cosine


# ============================================================================
# Config
# ============================================================================

class RunningMeanStd:
    """Welford online tracker of reward mean and variance for normalization."""
    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(eps)

    def update(self, x: "np.ndarray") -> None:
        batch_count = float(x.size)
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / tot) / tot
        self.count = tot

    def normalize(self, x: "np.ndarray") -> "np.ndarray":
        return x / (self.var ** 0.5 + 1e-8)


@dataclass
class SATTrainerConfig:
    num_episodes: int = 1000
    num_envs: int = 8
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    # 0.998, not 0.99. VRP is HorizonType.INFINITE with an average-cost
    # objective, and that is what the 3.096/3.324 baselines measure. gamma=0.99
    # gives an effective horizon of ~100 periods, which optimises a visibly
    # different objective — holding cost accrues every period forever, so
    # discounting it away rewards deferring pickups past the horizon.
    # ppo_attention.py:31 uses 0.998 on this MDP for the same reason.
    gamma: float = 0.998
    log_every: int = 50
    seed: int = 123
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    ppo_epochs: int = 4
    minibatch_size: int = 64
    gae_lambda: float = 0.95
    # Rollout length per PPO iteration. DECOUPLED from max_steps_per_episode:
    # upstream used one number for both, which forced every env to truncate
    # exactly on the last rollout step and left the truncation bootstrap
    # untested. Keeping them different means truncation lands mid-rollout,
    # which is the case the bootstrap exists for.
    rollout_steps: int = 128
    # Decision steps before an episode is TRUNCATED (not terminated). Required
    # for an infinite-horizon MDP, where category == FINAL is unreachable.
    max_steps_per_episode: int = 512
    weight_decay: float = 0.0
    # Reward normalisation — important when reward_scale != 1.
    # Divides per-step rewards by a running std estimate before GAE so that
    # value targets stay in a numerically stable range regardless of the
    # MDP's reward_scale setting.
    normalize_rewards: bool = True
    # Value-loss coefficient in the combined PPO loss.
    value_loss_coef: float = 0.5
    # Gradient-clipping max norm applied separately to policy and value nets.
    grad_clip_norm: float = 0.5

    # --- PPO stability hygiene (Section 4.6 of docs/beyond_agile_ideas.md) ---
    # Applied uniformly to *every* policy net (SAT, pointer, AGILE, Utility),
    # since they all train through this loop.  All default ON.
    # Per-minibatch advantage normalisation (CleanRL/SB3 standard); replaces
    # the previous whole-rollout normalisation.
    normalize_advantages: bool = True
    # PPO value-loss clipping: clip the value prediction to within
    # +/- value_clip_ratio of the rollout value before the squared error, and
    # take the elementwise max of clipped/unclipped loss.  0 disables clipping.
    value_clip_ratio: float = 0.2
    # Orthogonal initialisation of all nn.Linear weights (gain sqrt(2) for
    # hidden layers, 0.01 for scalar output heads), biases zeroed.
    orthogonal_init: bool = True
    # Learning-rate schedule: linear warmup over the first ``lr_warmup_frac`` of
    # iterations, then cosine decay to ``lr_min_frac`` * learning_rate.
    lr_schedule: bool = True
    lr_warmup_frac: float = 0.05
    lr_min_frac: float = 0.0
    # Entropy-coefficient annealing: cosine decay of entropy_coef from its
    # full value (iteration 0) down to entropy_min_frac * entropy_coef by the
    # final iteration. Off by default (existing behaviour unchanged); opt in
    # for problems prone to premature collapse onto a degenerate action.
    entropy_coef_schedule: bool = False
    entropy_min_frac: float = 0.1
    # Value network learns at learning_rate * value_lr_multiplier (own Adam
    # param group, independent of the policy's LR). 1.0 = unchanged behaviour.
    value_lr_multiplier: float = 1.0


# ============================================================================
# Training function (vectorized)
# ============================================================================

def train_sat_pointer_ppo(
    mdp: Any,
    make_state: Callable[[Any], Any],
    make_action_set: Callable[[Any], Any],
    config: SATTrainerConfig | None = None,
    reward_shaping: Callable[[Any, Any, Any, float], float] | None = None,
    sample_weight_fn: Callable[[Dict[str, Any]], float] | None = None,
    # Optional policy-net factory.  When None the default (AGILE)
    # TransformerPolicyNet is built.  Any factory with the signature
    # ``(raw_dims, type_names, d_model, nhead, num_layers) -> nn.Module`` whose
    # module exposes ``forward_logits`` / ``forward_logits_batched`` /
    # ``extend_for_new_type`` can be substituted while keeping the *entire* PPO
    # loop, value net, budget and seed identical.
    policy_net_factory: Callable[..., Any] | None = None,
    # Optional validation callback for in-trainer checkpointing / early stopping.
    # If provided, this callable is invoked as val_callback(policy_net, seq_builder)
    # and should return a scalar metric (larger is better). When the metric
    # improves the trainer will save the current checkpoint to val_save_dir.
    val_callback: Callable[[TransformerPolicyNet, SequenceBuilder], float] | None = None,
    val_save_dir: str | None = None,
    early_stop_patience: int = 0,
) -> Tuple[TransformerPolicyNet, TransformerValueNet, SequenceBuilder, List[float]]:
    """Train a Transformer policy with PPO over a vectorized CPython environment.

    Phase 1 of the SAT integration: this trains the AGILE net against a DynaPlex
    MDP stepped in plain CPython. It is a SPIKE, not the production path — DCL
    and the comparer's NN harness are engine-only, so nothing here can produce a
    number directly comparable to the 3.096 / 3.324 baselines. Evaluate by
    running a benchmark policy through the same loop and reporting the ratio.

    Divergences from the upstream trainer:

    - **MDP-aware only.** Callback mode (`step_env`/`action_to_key`) is gone;
      `mdp` is now required and positional.
    - **Truncation is bootstrapped, not treated as terminal.** VRP never reaches
      `StateCategory.FINAL`, so an episode ends only by hitting
      `max_steps_per_episode`. Zeroing the return there would teach the policy
      that cost stops accruing at the cut. The value of the pre-reset
      observation is added into the GAE delta instead (see 4b).
    - **Action validity is applied.** The action tokens are complete, so the raw
      softmax spans inadmissible actions; `_padded_valid` masks them in both the
      rollout and the update, mirroring the engine's masked CE / masked argmax.
    - **No dynamic-type extension.** The tokenizer's registry is closed
      (`sat_tokenizer._assert_registry`), so there is no `_on_new_type` callback
      and the optimizer is never rebuilt mid-run.
    - **No `seq_builder.pkl`.** Checkpoints are weights + JSON metadata.

    Args:
        mdp: The MDP instance, stepped in place.
        make_state: ``(ctx) -> state``; normally `mdp.get_initial_state`. Takes
            the TrajectoryContext, not a bare rng — the context owns the draw
            stream and DynaPlex MDPs draw through `ctx.rng`.
        make_action_set: ``(state) -> SATObservation``; normally
            ``lambda s: build_sat_observation(s, mdp)``.
        config: Training configuration.
        reward_shaping: Optional ``(prev_state, state, action_key,
            base_reward) -> reward``.

    Returns:
        (policy_net, value_net, seq_builder, episode_rewards)
    """
    if config is None:
        config = SATTrainerConfig()

    # Reset per-run state stored on the function object (bug: persisted across calls)
    train_sat_pointer_ppo._best_val_metric = -float("inf")
    train_sat_pointer_ppo._no_improve_count = 0

    # ------------------------------------------------------------------
    # 1. Build SequenceBuilder and discover types from a sample
    # ------------------------------------------------------------------
    seq_builder = SequenceBuilder(skip_root=True)
    # config.seed previously only seeded the numpy RNG (env/rollout sampling);
    # torch's global RNG — which governs policy/value net weight
    # initialization — was left uncontrolled. Seed it too so config.seed
    # fully determines the run (and so runs are actually reproducible/
    # comparable across seeds, as callers reasonably expect).
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)   # the PPO minibatch shuffle uses global numpy

    # The probe build is what CLOSES the tokenizer's type registry, so it has to
    # see every token type. `build_sat_observation` emits all of them on any
    # AWAIT_ACTION state (empty lists, never absent types), so an initial state
    # suffices. Uses a throwaway context on the eval seed space so it cannot
    # collide with the envs' training streams.
    probe_ctx = make_context(mdp, config.seed, eval=True)
    sample_state = make_state(probe_ctx)
    sample_seq = seq_builder.build(make_action_set(sample_state))

    raw_dims = seq_builder.raw_dims
    type_names = seq_builder.type_names

    print(f"[SAT Trainer] Sequence builder types: {type_names}")
    print(f"[SAT Trainer] Raw dims: {raw_dims}")
    print(f"[SAT Trainer] Sample sequence length: {sample_seq['seq_len']}")
    print(f"[SAT Trainer] Sample action groups: {sample_seq['num_action_groups']}")

    # ------------------------------------------------------------------
    # 2. Build Transformer networks
    # ------------------------------------------------------------------
    _policy_factory = policy_net_factory or build_transformer_policy_net
    policy = _policy_factory(
        raw_dims=raw_dims,
        type_names=type_names,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
    )
    value_net = build_transformer_value_net(
        raw_dims=raw_dims,
        type_names=type_names,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
    )

    # Orthogonal init (applies to whichever policy factory was used + value net)
    if getattr(config, "orthogonal_init", False):
        _init_orthogonal_(policy)
        _init_orthogonal_(value_net)

    # Value net gets its own param group so it can learn faster than the
    # policy (config.value_lr_multiplier) without changing the policy LR —
    # isolates "does the critic need a stronger learning signal" as a variable.
    _value_lr = config.learning_rate * float(getattr(config, "value_lr_multiplier", 1.0))
    optim = torch.optim.Adam(
        [
            {"params": list(policy.parameters()), "lr": config.learning_rate},
            {"params": list(value_net.parameters()), "lr": _value_lr},
        ],
        weight_decay=float(getattr(config, "weight_decay", 0.0)),
    )
    base_lrs = [g["lr"] for g in optim.param_groups]

    # NOTE: there is deliberately no new-type callback here. Upstream registered
    # one that called `extend_for_new_type` on both nets and REBUILT the
    # optimizer whenever the tokenizer met an unseen token type. That is
    # unrepresentable on the engine (the observation spec is frozen before the
    # build and 190 worker worlds hold featurizer clones), so the tokenizer's
    # registry is now closed and a late type raises instead. Keeping the
    # callback would have been dead code that quietly re-enabled a growing spec.

    # ------------------------------------------------------------------
    # 3. Create vectorized environment
    # ------------------------------------------------------------------
    vec_env = SATVectorEnv(
        mdp=mdp,
        make_state=make_state,
        make_action_set=make_action_set,
        seq_builder=seq_builder,
        num_envs=config.num_envs,
        seed=config.seed,
        max_steps=config.max_steps_per_episode,
        reward_shaping=reward_shaping,
    )
    seq_dicts = vec_env.reset()

    num_envs = config.num_envs
    num_steps = config.rollout_steps
    num_iterations = config.num_episodes

    # One entry per ITERATION: mean reward per step over that rollout. See the
    # comment at the append site for why this is not per-episode totals.
    episode_rewards: List[float] = []
    # Whole-episode totals, appended as episodes finish. Kept separate so the
    # two scales never mix in one series.
    completed_episode_totals: List[float] = []
    env_running_rewards = np.zeros(num_envs, dtype=np.float64)

    print(f"[SAT Trainer] Vectorized: {num_envs} envs × {num_steps} steps "
          f"× {num_iterations} iterations "
          f"= {num_envs * num_steps * num_iterations:,} total timesteps")

    # ------------------------------------------------------------------
    # Pre-allocate rollout storage ONCE (reused across iterations)
    # ------------------------------------------------------------------
    stored_seqs: List[Dict[str, Any] | None] = [None] * (num_steps * num_envs)
    actions_buf = torch.zeros(num_steps, num_envs, dtype=torch.long)
    logprobs_buf = torch.zeros(num_steps, num_envs)
    rewards_buf = torch.zeros(num_steps, num_envs)
    # dones_buf is the EPISODE-BOUNDARY mask (terminal OR truncated): it is what
    # stops GAE propagating across a reset. The two cases differ only in the
    # bootstrap, which boot_buf carries.
    dones_buf = torch.zeros(num_steps, num_envs, dtype=torch.bool)
    values_buf = torch.zeros(num_steps, num_envs)
    # gamma * V(final observation) for TRUNCATED steps, 0 elsewhere. Held apart
    # from rewards_buf on purpose: rewards get run through the running
    # normaliser below, and a value estimate is already in normalised units —
    # folding it into a raw reward and then normalising would corrupt it.
    boot_buf = torch.zeros(num_steps, num_envs)

    # GAE buffers — pre-allocated once
    advantages = torch.zeros(num_steps, num_envs)
    returns_buf = torch.zeros(num_steps, num_envs)
    lastgaelam = torch.zeros(num_envs)

    # Running reward normaliser (keeps values in a numerically stable range
    # regardless of MDP reward_scale; updated once per rollout before GAE).
    reward_rms = RunningMeanStd() if getattr(config, "normalize_rewards", True) else None

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    for iteration in range(num_iterations):

        # LR schedule: warmup + cosine decay.  Set directly on the param groups
        # each iteration so it survives optimizer rebuilds (new-type discovery).
        if getattr(config, "lr_schedule", False):
            lr_scale_now = _lr_scale(iteration, num_iterations, config)
            for pg, base_lr in zip(optim.param_groups, base_lrs):
                pg["lr"] = base_lr * lr_scale_now

        # Entropy-coefficient schedule (see _entropy_scale docstring).
        entropy_coef_now = config.entropy_coef * _entropy_scale(iteration, num_iterations, config)

        # ==============================================================
        # 4a. Rollout phase — collect transitions from all envs
        # ==============================================================
        # boot_buf is sparse (only truncated steps write it), so it must be
        # cleared between rollouts or last iteration's bootstraps leak in
        boot_buf.zero_()

        for step in range(num_steps):
            with torch.no_grad():
                # Store current seq_dicts (flat 1D — no nested list)
                base = step * num_envs
                stored_seqs[base:base + num_envs] = seq_dicts

                # --- Batched forward pass ---
                flat_logits, act_offsets, act_lengths = (
                    policy.forward_logits_batched(seq_dicts))
                values_buf[step] = value_net.forward_batched(seq_dicts)

                # --- Vectorized action sampling ---
                # Scatter flat logits → [num_envs, max_actions] padded matrix
                # then batch-multinomial all envs at once.
                padded_logits = _flat_to_padded(
                    flat_logits, act_offsets, act_lengths)
                # crop to ADMISSIBLE actions before the softmax: the tokens span
                # every action index, so an unmasked sample would hand the MDP an
                # off-grid move. -1e9 (not -inf) keeps the row finite so log/
                # entropy stay well-defined on fully-masked-out columns.
                a_valid = _padded_valid(seq_dicts, padded_logits.shape[1])
                padded_logits = padded_logits.masked_fill(~a_valid, -1e9)
                padded_probs = torch.softmax(padded_logits, dim=1)
                sampled = torch.multinomial(padded_probs, 1).squeeze(1)
                actions_buf[step] = sampled
                logprobs_buf[step] = torch.log(
                    padded_probs.gather(
                        1, sampled.unsqueeze(1)).squeeze(1) + 1e-8)

            # --- Step all envs ---
            action_list = actions_buf[step].tolist()
            cur_seqs = stored_seqs[base:base + num_envs]
            seq_dicts, step_rewards, step_dones, step_truncs, final_seqs = (
                vec_env.step(action_list, cur_seqs)
            )
            rewards_buf[step] = torch.from_numpy(step_rewards)
            # boundary = terminal OR truncated: either way the next stored
            # observation belongs to a different episode
            boundary = step_dones | step_truncs
            dones_buf[step] = torch.from_numpy(boundary)

            # Truncation bootstrap: value the observation we were cut at, so the
            # policy is not told that cost stops accruing at the horizon. Only
            # truncated envs — a genuinely terminal state is worth 0 by
            # definition. (VRP is infinite-horizon, so in practice every
            # boundary here is a truncation.)
            trunc_idx = np.flatnonzero(step_truncs)
            if trunc_idx.size:
                fin = [final_seqs[i] for i in trunc_idx]
                with torch.no_grad():
                    v_fin = value_net.forward_batched(fin)
                boot_buf[step, torch.from_numpy(trunc_idx)] = (
                    config.gamma * v_fin)

            # Track per-env running totals so completed episodes can be reported
            env_running_rewards += step_rewards
            done_mask = boundary.astype(bool)
            if done_mask.any():
                completed_episode_totals.extend(
                    env_running_rewards[done_mask].tolist())
                env_running_rewards[done_mask] = 0.0

        # ONE metric, on ONE scale: mean reward per step over this rollout.
        #
        # Upstream appended two different quantities to a single list —
        # per-rollout sums when no episode boundary fell inside the rollout, and
        # whole-episode totals when one did. With rollout_steps=64 and
        # max_steps_per_episode=512 that mixes values ~8x apart, so the logged
        # average oscillated by a factor of ~2.5 between consecutive logs and
        # showed no trend whether or not the policy was improving. Per-step
        # reward is invariant to rollout length and to where boundaries fall, so
        # it is actually comparable across iterations. Whole-episode totals are
        # still collected (above) for callers that want them.
        episode_rewards.append(float(rewards_buf.mean()))

        # ==============================================================
        # 4b. GAE(λ) computation — vectorized over all envs
        # ==============================================================
        # Optionally normalise rewards so value targets stay near 0 in
        # scale regardless of MDP reward_scale.
        if reward_rms is not None:
            raw_rewards_np = rewards_buf.numpy()
            reward_rms.update(raw_rewards_np)
            rewards_buf = torch.from_numpy(reward_rms.normalize(raw_rewards_np)).float()

        with torch.no_grad():
            # Bootstrap values for the state after the last rollout step
            next_values = value_net.forward_batched(seq_dicts)  # [num_envs]

            # Re-use pre-allocated buffers (zero in-place)
            lastgaelam.zero_()

            for t in reversed(range(num_steps)):
                nextnonterminal = 1.0 - dones_buf[t].float()
                if t == num_steps - 1:
                    nextvalues = next_values
                else:
                    nextvalues = values_buf[t + 1]
                # boot_buf[t] is gamma*V(s_final) at a truncation and 0
                # otherwise; nextnonterminal has already zeroed the
                # across-the-reset term, so the two never double-count.
                delta = (rewards_buf[t]
                         + boot_buf[t]
                         + config.gamma * nextvalues * nextnonterminal
                         - values_buf[t])
                lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam

            torch.add(advantages, values_buf, out=returns_buf)

        # ==============================================================
        # 4c. PPO update — multiple epochs over shuffled mini-batches
        # ==============================================================
        batch_size = num_steps * num_envs

        # stored_seqs is already a flat 1D list [num_steps * num_envs]
        b_seqs = stored_seqs
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns_buf.reshape(-1)
        b_values = values_buf.reshape(-1)  # rollout values (for value clipping)

        # Build per-sample weights (optional). Defaults to 1.0 for every stored sample.
        if sample_weight_fn is not None:
            b_weights_list: List[float] = []
            for seq in b_seqs:
                if seq is None:
                    b_weights_list.append(0.0)
                else:
                    try:
                        w = float(sample_weight_fn(seq))
                    except Exception:
                        w = 1.0
                    b_weights_list.append(w)
            b_weights = torch.tensor(b_weights_list, dtype=torch.float32)
        else:
            b_weights = torch.ones(batch_size, dtype=torch.float32)

        # Advantage normalisation is done per-minibatch below (config-gated),
        # which is the standard PPO recipe (CleanRL/SB3).

        mb_size = min(config.minibatch_size, batch_size)

        for _ppo_epoch in range(config.ppo_epochs):
            b_inds = np.random.permutation(batch_size)

            for start in range(0, batch_size, mb_size):
                mb_idx = b_inds[start : start + mb_size]

                # Collect valid samples in this minibatch
                valid_orig: list[int] = [
                    int(i) for i in mb_idx
                    if b_seqs[i] is not None
                    and b_seqs[i]["num_action_groups"] > 0
                ]
                if not valid_orig:
                    continue

                valid_seqs = [b_seqs[idx] for idx in valid_orig]
                vi = torch.tensor(valid_orig, dtype=torch.long)

                # --- Batched forward passes ---
                flat_logits, offsets, lengths = (
                    policy.forward_logits_batched(valid_seqs))
                v_preds = value_net.forward_batched(valid_seqs)

                # --- Vectorized PPO loss ---
                # Scatter flat logits → padded [mb, max_L] matrix
                padded_logits = _flat_to_padded(
                    flat_logits, offsets, lengths)
                # the SAME validity crop as the rollout — if the update
                # optimised an unmasked distribution the ratios would be taken
                # against a different policy than the one that acted
                padded_logits = padded_logits.masked_fill(
                    ~_padded_valid(valid_seqs, padded_logits.shape[1]), -1e9)
                padded_probs = torch.softmax(padded_logits, dim=1)

                # Gather log-probs for the actions taken during rollout
                mb_act = b_actions[vi]
                max_L = padded_probs.shape[1]
                safe_act = mb_act.clamp(max=max_L - 1)
                # True where the stored action index is within range
                valid_mask = mb_act < lengths

                n_valid = int(valid_mask.sum().item())
                if n_valid == 0:
                    continue

                new_lp = torch.log(
                    padded_probs.gather(
                        1, safe_act.unsqueeze(1)).squeeze(1) + 1e-8)

                # PPO clipped surrogate — fully vectorized
                ratios = torch.exp(new_lp - b_logprobs[vi])
                advs = b_advantages[vi]
                # Per-minibatch advantage normalisation (over the valid entries
                # that actually contribute to the loss).
                if getattr(config, "normalize_advantages", False):
                    a_valid = advs[valid_mask]
                    if a_valid.numel() > 1:
                        advs = (advs - a_valid.mean()) / (a_valid.std() + 1e-8)
                surr1 = ratios * advs
                surr2 = torch.clamp(
                    ratios,
                    1 - config.clip_ratio,
                    1 + config.clip_ratio,
                ) * advs
                surr = torch.min(surr1, surr2)

                # Apply per-sample weights if provided
                mb_w = b_weights[vi].to(surr.device)
                if mb_w.sum().item() > 0:
                    weighted = surr * mb_w
                    policy_loss = -weighted[valid_mask].sum() / (mb_w[valid_mask].sum() + 1e-12)
                else:
                    policy_loss = -surr[valid_mask].mean()

                # Entropy: sum over action dim, mean over valid samples
                # (padding positions have softmax ≈ 0, contributing ≈ 0)
                entropy = -(
                    padded_probs * torch.log(padded_probs + 1e-8)
                ).sum(dim=1)[valid_mask]
                # weight entropy similarly
                if mb_w.sum().item() > 0:
                    entropy = (entropy * mb_w[valid_mask]).sum() / (mb_w[valid_mask].sum() + 1e-12)
                else:
                    entropy = entropy.mean()

                # Value loss — fully vectorized, with optional PPO value-loss
                # clipping: penalise the larger of the unclipped error and the
                # error of the value prediction clipped to within
                # +/- value_clip_ratio of the rollout value.
                mb_returns = b_returns[vi]
                value_err = (v_preds - mb_returns) ** 2
                _vc = float(getattr(config, "value_clip_ratio", 0.0))
                if _vc > 0:
                    mb_old_v = b_values[vi]
                    v_clipped = mb_old_v + torch.clamp(
                        v_preds - mb_old_v, -_vc, _vc)
                    value_err = torch.max(value_err, (v_clipped - mb_returns) ** 2)
                if mb_w.sum().item() > 0:
                    value_loss = (value_err[valid_mask] * mb_w[valid_mask]).sum() / (mb_w[valid_mask].sum() + 1e-12)
                else:
                    value_loss = value_err[valid_mask].mean()

                loss = (policy_loss
                        + getattr(config, "value_loss_coef", 0.5) * value_loss
                        - entropy_coef_now * entropy)

                optim.zero_grad()
                loss.backward()
                _gc = float(getattr(config, "grad_clip_norm", 0.5))
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=_gc)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=_gc)
                optim.step()

        # ==============================================================
        # 4d. Logging
        # ==============================================================
        if (iteration + 1) % config.log_every == 0:
            # one entry per iteration now, so the window is `log_every` long
            recent = episode_rewards[-config.log_every:]
            avg = float(np.mean(recent)) if recent else 0.0
            msg = (f"iter={iteration + 1:4d}  reward/step={avg:8.4f}  "
                   f"(cost/step={-avg:7.4f})")
            if getattr(config, "entropy_coef_schedule", False):
                msg += f"  entropy_coef={entropy_coef_now:.4f}"
            print(msg, flush=True)

        # Optional: run validation callback / checkpointing
        if val_callback is not None and (iteration + 1) % getattr(config, "val_eval_every", config.log_every) == 0:
            try:
                val_metric = float(val_callback(policy, seq_builder))
            except Exception:
                val_metric = -float("inf")

            # Keep best metric and save checkpoint to provided directory
            if val_metric > train_sat_pointer_ppo._best_val_metric:
                train_sat_pointer_ppo._best_val_metric = val_metric
                train_sat_pointer_ppo._no_improve_count = 0
                if val_save_dir is not None:
                    import json
                    import os
                    os.makedirs(val_save_dir, exist_ok=True)
                    # Save state_dicts (more robust than pickling full objects)
                    torch.save(policy.state_dict(), os.path.join(val_save_dir, "policy_state.pt"))
                    torch.save(value_net.state_dict(), os.path.join(val_save_dir, "value_state.pt"))
                    # Save minimal model metadata required to reconstruct architecture
                    meta = {
                        "raw_dims": raw_dims,
                        "type_names": type_names,
                        "d_model": config.d_model,
                        "nhead": config.nhead,
                        "num_layers": config.num_layers,
                    }
                    with open(os.path.join(val_save_dir, "model_meta.json"), "w") as mf:
                        json.dump(meta, mf, indent=2)
                    # No seq_builder.pkl. The tokenizer is reconstructible from
                    # the schema (sat_schema is the source of truth) and
                    # `raw_dims`/`type_names` above pin the architecture, so
                    # pickling it would only add a code-bearing artifact that
                    # breaks on any refactor. DynaPlex agents are weights + a
                    # JSON sidecar for exactly this reason.
                    print(f"  [val] metric={val_metric:.6f}  -> saved checkpoint to {val_save_dir}")
            else:
                train_sat_pointer_ppo._no_improve_count += 1
                print(f"  [val] metric={val_metric:.6f}  (no improvement {train_sat_pointer_ppo._no_improve_count})")

            # Early stopping
            if early_stop_patience > 0 and train_sat_pointer_ppo._no_improve_count >= early_stop_patience:
                print(f"  Early stopping: no improvement for {early_stop_patience} validations. Stopping training.")
                break

    return policy, value_net, seq_builder, episode_rewards


# ============================================================================
# Inference helper
# ============================================================================

def select_action_sat(
    policy_net: TransformerPolicyNet,
    seq_builder: SequenceBuilder,
    state: Any,
    make_action_set: Callable[[Any], Any],
    valid: Any | None = None,
    deterministic: bool = True,
) -> Any:
    """Select an action using the trained Transformer policy (SAT pipeline).

    Action keys are resolved directly from the Action dataclass objects stored
    in ``seq["_actions"]`` via :func:`sat_actions.resolve_action`.

    Args:
        valid: the action-validity mask for `state`, as a bool array over action
            indices — normally ``ctx.valid.arr[0]`` after
            ``mdp.write_action_validity(state, ctx.valid)``. REQUIRED for a
            correct choice: the action tokens span every action index, so an
            unmasked argmax will happily return an inadmissible action. It is
            optional only so callers on MDPs whose action set is inherently
            all-valid need not pass it. (Upstream's `action_to_key` parameter
            occupied this slot and was ignored; it is gone.)

    Returns:
        The action key suitable for ``mdp.modify_state_with_action``.
    """
    policy_net.eval()
    sat_obs = make_action_set(state)
    seq = seq_builder.build(sat_obs)

    with torch.no_grad():
        logits = policy_net.forward_logits(
            type_features=seq["type_features"],
            type_ids=seq["type_ids"],
            action_mask=seq["action_mask"],
            indices_by_type=seq["indices_by_type"],
            action_group_ids=seq.get("action_group_ids"),
            num_action_groups=seq.get("num_action_groups"),
            intra_group_pos=seq.get("intra_group_pos"),
        )

        # Protect against empty action sets: sequence builders may occasionally
        # produce num_action_groups == 0 (no valid actions). In that case
        # return the idle action key (-1) as a safe default.
        if seq.get("num_action_groups", 0) == 0 or logits.numel() == 0:
            return -1

        if valid is not None:
            m = torch.as_tensor(np.asarray(valid, dtype=np.bool_))
            logits = logits.masked_fill(~m[:logits.shape[0]], -1e9)
        probs = torch.softmax(logits, dim=0)

        if deterministic:
            chosen = int(torch.argmax(probs).item())
        else:
            chosen = int(torch.multinomial(probs, 1).item())

        key = resolve_action(seq["_actions"], chosen)

    policy_net.train()
    return key
