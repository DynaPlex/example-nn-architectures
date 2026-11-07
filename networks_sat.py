"""AGILE-style structured-action network over the dense SAT token bundle.

Consumed by DCL/PPO via `dp.Net("networks_sat.sat_agile_net", ...)` — a dotted
path plus JSON kwargs, because agent artifacts must rebuild themselves on cold
cluster nodes; a live module can never be the specification.

Token bundle in (see `featurizers_sat.SATFeaturizer`): `summary` [B,1,4],
`depot` [B,1,2], `vehicles` [B,V,4], `orders` [B,max_n,2] + `orders_count` [B],
`moves` [B,num_actions,6]. Out: `[B, num_actions]` logits, in action order.

Architecture, following Jain et al.'s AGILE as ported in `sat_net_ref`:

  1. each token tensor gets its own linear projection to d_model;
  2. the CONTEXT tokens (summary | depot | vehicles | orders) go through a
     transformer encoder with padded orders masked out, then a masked mean-pool
     to one state vector per sample;
  3. the ACTION tokens become one node each, tagged with a role embedding, and
     attend over the complete graph [state_node ; action_nodes] — this is the
     AGILE step: an action's representation is conditioned on which other
     actions are available;
  4. a shared head scores each relational action node against the state vector.

Three simplifications versus `sat_net_ref`, each because the dense layout makes
the machinery it needed unnecessary rather than because it was wrong:

  * NO type embeddings. `sat_net_ref` adds `type_embeddings[type_id]` to every
    token so one shared projection can tell types apart. Here each type already
    has its own projection, whose bias absorbs exactly that constant — the
    embedding would be a redundant parameter. This also deletes both
    `clamp(max=num_embeddings-1)` safe-id hacks, which silently aliased an
    unknown type onto the last embedding row.
  * NO within-group action layer. It exists to let the tokens of a MULTI-token
    action attend to each other. VRP actions are single-token, so at group size
    1 that layer's attention is a softmax over one element — it degenerates to a
    per-token linear map plus FFN, i.e. dead compute with respect to its
    purpose. Re-add it (as `[B,G,C,d] -> [B*G,C,d]` with an ordinary
    key-padding mask from an `is_real` column) if an MDP with nested actions
    arrives; note that within a group all tokens share the block BY
    CONSTRUCTION, so it needs neither the `block & ~eye` self-edge hack nor the
    `nan_to_num` that `sat_net_ref` requires.
  * NO padding anywhere on the action side. `featurizers_sat` writes the action
    set COMPLETE, so G == num_actions on every sample. That removes the -inf key
    bias, the `gids_pad`/`max_G + 1` dummy pooling bin, and the NaN guards that
    went with them — the most error-prone code in the reference net.

The hand-rolled `nn.MultiheadAttention` post-norm block is kept even though the
mask it protected against is gone, because the trap is real and returns the
moment padding does: `nn.TransformerEncoder`'s eval fast path silently DROPS a
float `attn_mask`, so a model that trains correctly can evaluate incorrectly.

Masking contract: this network emits finite logits for ALL num_actions,
including inadmissible ones, and never reads the validity mask. Cropping happens
downstream — masked CE in DCL training (`dynaplex/nn.py:464`), masked argmax when
acting (`nn.py:77-86`), `masked_fill(~mask, -inf)` in PPO. Two consequences: the
net must never emit -inf or NaN (a NaN poisons the `.max()` that selects the
action), and its output must be unaffected by the mask's contents.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

# The context tensors, in the order they are concatenated into the encoder.
# Fixed here rather than derived from the spec because token order is part of
# the network's identity: a trained agent reloaded with a different order would
# silently mismatch its weights.
_CONTEXT = ("summary", "depot", "vehicles", "orders")
# Variable-count context tensors -> the companion holding their true length.
_COUNTS = {"orders": "orders_count"}
_ACTIONS = "moves"
# `moves` columns 0..3 are pure geometry (action_index, row, col, on_grid);
# 4..5 are has_order and nearest_order_dist -- the myopic policy's own decision
# statistic (featurizers_sat.py:205-224). `action_features="geometry"` zeroes the
# tail so the SAME sample set can answer "was it the features or the architecture".
_ACTION_GEOMETRY_COLS = 4


class _FullSelfAttnLayer(nn.Module):
    """Post-norm self-attention block over the complete [state ; actions] graph.

    Built directly on `nn.MultiheadAttention` rather than using
    `nn.TransformerEncoderLayer` so that any additive float attention bias is
    applied reliably in BOTH train and eval mode — `nn.TransformerEncoder`'s eval
    fast path silently drops a float mask. No bias is passed today (the action
    set is complete, so there is nothing to mask), but the block is written to
    accept one so re-introducing padding cannot reintroduce that bug.
    """

    def __init__(self, d_model: int, n_heads: int, ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff), nn.ReLU(),
                                 nn.Linear(ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        """x: [B, N, d]; attn_bias: [B*n_heads, N, N] additive bias or None."""
        sa_out, _ = self.self_attn(x, x, x, attn_mask=attn_bias)
        x = self.norm1(x + sa_out)
        return self.norm2(x + self.ffn(x))


class _PerNodeFFNLayer(nn.Module):
    """The ABLATION control: `_FullSelfAttnLayer` with the attention removed.

    Exists so "does the inter-action attention earn its keep" can be asked
    without confounding the answer with capacity. Simply deleting the
    inter-action stack would remove cross-action conditioning AND `n_layers`
    residual FFN blocks at once, so a gap could be either. This keeps the depth,
    the FFN width, the residuals and the LayerNorms identical, and removes only
    the attention sublayer — i.e. only the ability of one action node to see
    another.

    Parameter count differs from `_FullSelfAttnLayer` by exactly the MultiheadAttention
    projections (4 * d_model^2 + biases); the FFN, which dominates at
    d_model=64 / ff=128, is unchanged.
    """

    def __init__(self, d_model: int, ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff), nn.ReLU(),
                                 nn.Linear(ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        """Same signature as _FullSelfAttnLayer; `attn_bias` is ignored."""
        x = self.norm1(x)
        return self.norm2(x + self.ffn(x))


class _SATAgileNet(nn.Module):
    """Shared trunk; `head_kind` selects the actor or the critic readout."""

    def __init__(self, spec: dict, num_actions: int, *, d_model: int,
                 n_layers: int, n_heads: int, ff: int, hidden_head: int,
                 head_kind: str, inter_action: str = "attention",
                 action_features: str = "full"):
        super().__init__()
        assert inter_action in ("attention", "mlp", "none"), inter_action
        assert action_features in ("full", "geometry"), action_features
        missing = [k for k in (*_CONTEXT, _ACTIONS) if k not in spec]
        assert not missing, (
            f"sat_agile_net consumes featurizers_sat.SATFeaturizer's bundle; "
            f"spec is missing {missing} (got {sorted(spec)})")
        for tensor, count in _COUNTS.items():
            assert count in spec, (
                f"{tensor!r} is variable-count, so its {count!r} companion must "
                f"be in the spec — see SATFeaturizer.spec()")

        self.head_kind = head_kind
        self.action_features = action_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_actions = spec[_ACTIONS].sample_shape[0]
        self.max_orders = spec["orders"].sample_shape[0]

        if head_kind == "actions":
            assert num_actions == self.n_actions, (
                f"sat_agile_net pairs action tokens 1:1 with actions: "
                f"num_actions={num_actions} but the featurizer declares "
                f"{self.n_actions} move tokens (is the mdp's ActionEncoding the "
                f"one the sample set was collected under?)")

        # one projection per token tensor; its bias is what makes the dropped
        # type embedding redundant
        self.proj = nn.ModuleDict({
            name: nn.Linear(spec[name].sample_shape[-1], d_model)
            for name in (*_CONTEXT, _ACTIONS)
        })

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff, batch_first=True,
            norm_first=True)
        self.context_encoder = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False)

        # "attention" = AGILE (each action node sees the others and the state
        # node); "mlp" = the capacity-matched control that sees neither; "none" =
        # no blocks at all. See _PerNodeFFNLayer.
        self.inter_action = inter_action
        if inter_action == "attention":
            blocks = [_FullSelfAttnLayer(d_model, n_heads, ff)
                      for _ in range(n_layers)]
        elif inter_action == "mlp":
            blocks = [_PerNodeFFNLayer(d_model, ff) for _ in range(n_layers)]
        else:
            blocks = []
        self.inter_action_layers = nn.ModuleList(blocks)
        # 0 = an action node, 1 = the state node
        self.node_role_emb = nn.Embedding(2, d_model)

        if head_kind == "actions":
            self.head = nn.Sequential(
                nn.Linear(2 * d_model, hidden_head), nn.ReLU(),
                nn.Linear(hidden_head, 1))
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden_head), nn.ReLU(),
                nn.Linear(hidden_head, num_actions))

    def _context(self, batch: dict) -> tuple[Tensor, Tensor]:
        """Project and concatenate the context tokens; return (tokens, pad_mask).

        EVERY object in the state is a context token here — the summary, the
        depot, all vehicles, and one per waiting order — exactly as in
        `sat_net_ref`, which likewise compresses them through an encoder and a
        mean-pool into ONE state node before the relational layer. The action
        tokens are additional, not a replacement.

        pad_mask is True where a token is padding, reconstructed from the
        `_count` companions (`arange(max) >= count`) the same way
        `networks.py:74-77` does, so no mask tensor has to cross the engine
        boundary.

        The mask is built in the SAME loop that concatenates the tokens, so the
        two cannot desynchronise. An earlier version precomputed the variable
        block's start offset as "the total width of the fixed tensors", which is
        only right while every variable tensor sits after every fixed one in
        `_CONTEXT` — reordering that tuple would have silently masked the wrong
        slice, and a wrong padding mask is invisible to every shape check.
        """
        B = batch[_CONTEXT[0]].shape[0]
        parts: list[Tensor] = []
        masks: list[Tensor] = []
        for name in _CONTEXT:
            x = batch[name]
            parts.append(self.proj[name](x))
            width = x.shape[1]
            if name in _COUNTS:
                # Cast defensively: PPO's rollout buffer allocates EVERY spec
                # tensor as float32 (dynaplex/algorithms/ppo.py:404 passes no
                # dtype), so an int64 count arrives as a float there and as
                # int64 under DCL. The comparison would promote either way, but
                # casting keeps this correct if a count is ever used as an index.
                cnt = batch[_COUNTS[name]].to(torch.long).reshape(B, 1)
                idx = torch.arange(width, device=x.device).unsqueeze(0)
                masks.append(idx >= cnt)
            else:
                masks.append(torch.zeros(B, width, dtype=torch.bool,
                                         device=x.device))
        return torch.cat(parts, dim=1), torch.cat(masks, dim=1)

    def _action_input(self, batch: dict) -> Tensor:
        """The `moves` slab, with the hand-crafted columns zeroed under
        `action_features="geometry"`.

        Zeroing rather than slicing keeps every parameter shape identical, and it
        is EXACTLY equivalent to a featurizer that never wrote those columns: a
        zero input contributes nothing to the output and its weights receive no
        gradient. So this ablation runs on the SAME collected sample set as the
        full net, which is what makes it cheap -- no re-collection, byte-identical
        labels, sampling noise cancels.
        """
        raw = batch[_ACTIONS]
        if self.action_features == "full":
            return raw
        keep = raw[..., :_ACTION_GEOMETRY_COLS]
        drop = torch.zeros_like(raw[..., _ACTION_GEOMETRY_COLS:])
        return torch.cat([keep, drop], dim=-1)

    def forward(self, batch: dict) -> Tensor:
        tokens, pad = self._context(batch)
        enc = self.context_encoder(tokens, src_key_padding_mask=pad)
        # A fully-padded row would make softmax produce NaN; VRP always has at
        # least the summary and depot tokens real, so this is belt-and-braces.
        enc = enc.nan_to_num(0.0)

        # masked mean-pool over real context tokens only -> [B, d]
        keep = (~pad).unsqueeze(-1).to(enc.dtype)
        state_vec = (enc * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)

        # AGILE: complete-graph attention over [state_node ; action_nodes].
        # No padding mask — the action set is complete, so every node is real.
        B = tokens.shape[0]
        action_nodes = self.proj[_ACTIONS](self._action_input(batch))  # [B, A, d]
        zero = torch.zeros(1, dtype=torch.long, device=tokens.device)
        action_nodes = action_nodes + self.node_role_emb(zero)
        state_node = (state_vec + self.node_role_emb(zero + 1)).unsqueeze(1)
        nodes = torch.cat([state_node, action_nodes], dim=1)   # [B, 1 + A, d]
        for layer in self.inter_action_layers:
            nodes = layer(nodes)

        if self.head_kind != "actions":
            # critic: one scalar per sample, read off the state node after it has
            # attended over the available actions
            return self.head(nodes[:, 0])

        rel_actions = nodes[:, 1:]                                   # [B, A, d]
        state_rep = state_vec.unsqueeze(1).expand(B, rel_actions.shape[1], -1)
        feat = torch.cat([rel_actions, state_rep], dim=-1)
        # token i IS action i (featurizers_sat writes moves in action order), so
        # these logits are already in action order; the validity mask crops them
        # downstream and this net never masks itself
        return self.head(feat).squeeze(-1)                           # [B, A]


def sat_agile_net(spec: dict, num_actions: int, *, d_model: int = 64,
                  n_layers: int = 2, n_heads: int = 4, ff: int = 128,
                  hidden_head: int = 64,
                  inter_action: str = "attention",
                  action_features: str = "full") -> nn.Module:
    """Factory entry point: `dp.Net("networks_sat.sat_agile_net", d_model=...)`.

    Args:
        spec: the observation spec — must be `SATFeaturizer`'s bundle
            (summary/depot/vehicles/orders/orders_count/moves). Extra entries
            are ignored.
        num_actions: the mdp's action count; must equal the number of move
            tokens the featurizer declares (asserted).
        d_model / n_layers / n_heads / ff: transformer size knobs. `n_layers`
            applies to the context encoder AND the inter-action stack.
        hidden_head: hidden width of the scoring head.
        inter_action: what sits between the action nodes and the scoring head.
            `"attention"` (default) is AGILE — complete-graph attention over
            `[state ; actions]`, so an action's representation depends on which
            others are available. `"mlp"` is the CAPACITY-MATCHED ABLATION: the
            same depth and FFN width with the attention sublayer removed, so each
            action is scored from its own features and the pooled state only.
            `"none"` drops the stack entirely (removes conditioning *and*
            capacity — prefer `"mlp"` for a fair comparison). A plain string so it
            round-trips through `dp.Net`'s JSON kwargs to cold cluster nodes.
        action_features: `"full"` (default) uses all six `moves` columns.
            `"geometry"` zeroes `has_order` and `nearest_order_dist`, which is
            exactly equivalent to a featurizer that never wrote them -- so the
            feature ablation runs on the SAME sample set as the full net.
    """
    return _SATAgileNet(spec, num_actions, d_model=d_model, n_layers=n_layers,
                        n_heads=n_heads, ff=ff, hidden_head=hidden_head,
                        head_kind="actions", inter_action=inter_action,
                        action_features=action_features)


def sat_agile_critic(spec: dict, num_actions: int, *, d_model: int = 64,
                     n_layers: int = 2, n_heads: int = 4, ff: int = 128,
                     hidden_head: int = 64) -> nn.Module:
    """PPO value head over the same trunk, returning `[B, 1]`.

    PPO builds critics as `factory(spec, 1)`, so `num_actions` is 1 here and the
    readout is the state node after it has attended over the action nodes —
    which is the natural critic for this architecture, since the availability of
    actions is itself state information. Needed because PPO's built-in
    `_ValueHead` is single-tensor only (`ppo.py:183-205`) and this observation is
    a bundle. Training machinery only: it lives in PPO's checkpoint, never in
    the agent artifact.
    """
    return _SATAgileNet(spec, num_actions, d_model=d_model, n_layers=n_layers,
                        n_heads=n_heads, ff=ff, hidden_head=hidden_head,
                        head_kind="value")
