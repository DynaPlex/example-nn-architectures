"""Transformer-based policy and value networks.

The policy network is **AGILE** (Jain et al., ICLR 2022, "Know Your Action
Set"), ported into the SAT tokenisation pipeline.  It is a relational
"pool-then-relate" net:

    1. Per-type linear projections: raw features → d_model
    2. Learned type embeddings added to projected features
    3. Action tokens receive **intra-group positional embeddings** so the model
       can distinguish token order within each action (e.g. stop sequence)
    4. Context tokens (non-actions) run through a self-attention encoder, then
       mean-pool to one state vector
    5. Within-action self-attention (block-diagonal over action_group_id) →
       mean-pool each group to one **atomic action node**
    6. Complete-graph self-attention over the atomic action nodes plus a
       role-tagged state node → score each action through the head → logits

    For flat actions (group size 1) the within-action attention collapses to a
    per-token MLP and the positional embeddings are a no-op.

Value net architecture (self-attention over all tokens):
    1. Same per-type projections + type embeddings
    2. Self-attention over ALL tokens
    3. Mean-pool → MLP → scalar value
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


# ============================================================================
# Shared helper: build embeddings from sequence dict
# ============================================================================

def _build_embeddings(
    input_projs: nn.ModuleDict,
    type_embeddings: nn.Embedding,
    type_features: Dict[str, Tensor],
    type_ids: Tensor,
    indices_by_type: Dict[str, List[int]],
    d_model: int,
) -> Tensor:
    """Project raw features and add type embeddings.  Returns [seq_len, d_model]."""
    seq_len = type_ids.shape[0]
    device = type_ids.device
    embeddings = torch.zeros(seq_len, d_model, device=device)
    for tname, proj in input_projs.items():
        if tname not in indices_by_type:
            continue
        idxs = indices_by_type[tname]
        feats = type_features[tname]
        projected = proj(feats)
        # Vectorized scatter — replaces per-element Python loop
        if isinstance(idxs, torch.Tensor):
            idx_t = idxs
        else:
            idx_t = torch.tensor(idxs, dtype=torch.long, device=device)
        embeddings.index_copy_(0, idx_t, projected)
    # Clamp type_ids in case a late-discovered type has an id beyond
    # the embedding table size (it will share the last type embedding).
    safe_type_ids = type_ids.clamp(max=type_embeddings.num_embeddings - 1)
    embeddings = embeddings + type_embeddings(safe_type_ids)
    return embeddings


def _build_embeddings_batched(
    input_projs: nn.ModuleDict,
    type_embeddings: nn.Embedding,
    seq_dicts: List[Dict[str, Any]],
    d_model: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Batch-build embeddings for B same-structure sequences.

    All sequences must have the same seq_len and type layout (checked by
    caller).  Performs ONE ``Linear`` projection per type across all B
    samples instead of B individual ``_build_embeddings`` calls.

    Returns
    -------
    emb : Tensor[B, S, d_model]
    pad_mask : BoolTensor[B, S]  (all False for uniform sequences)
    """
    B = len(seq_dicts)
    ref = seq_dicts[0]
    S = ref["type_ids"].shape[0]

    emb = torch.zeros(B, S, d_model, device=device)

    for tname, proj in input_projs.items():
        ibt = ref["indices_by_type"]
        if tname not in ibt:
            continue
        idxs = ibt[tname]
        if isinstance(idxs, Tensor):
            idx_t = idxs.to(device)
        else:
            idx_t = torch.tensor(idxs, dtype=torch.long, device=device)

        # Stack features from all B samples: [B, n_tok, raw_dim]
        stacked = torch.stack(
            [seq["type_features"][tname] for seq in seq_dicts],
        ).to(device)
        n_tok = stacked.shape[1]
        # Single batched projection for ALL B*n_tok tokens at once
        projected = proj(
            stacked.reshape(B * n_tok, -1),
        ).reshape(B, n_tok, d_model)
        emb[:, idx_t] = projected

    # Type embeddings — identical for every sample, broadcast [1, S, d]
    type_ids = ref["type_ids"].to(device)
    safe_ids = type_ids.clamp(max=type_embeddings.num_embeddings - 1)
    emb = emb + type_embeddings(safe_ids).unsqueeze(0)

    pad_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    return emb, pad_mask


# ============================================================================
# Action-group pooling helper
# ============================================================================

def _pool_action_groups(
    action_emb: Tensor,
    action_group_ids_for_actions: Tensor,
    num_groups: int,
) -> Tensor:
    """Mean-pool action-token embeddings per action group.

    Args:
        action_emb: [num_action_tokens, d_model]
        action_group_ids_for_actions: [num_action_tokens] with values in [0, num_groups)
        num_groups: number of distinct action groups

    Returns:
        pooled: [num_groups, d_model]
    """
    d_model = action_emb.shape[-1]
    device = action_emb.device
    pooled = torch.zeros(num_groups, d_model, device=device)
    counts = torch.zeros(num_groups, 1, device=device)

    # Vectorized scatter-add mean pooling (replaces per-group Python loop)
    gids_expanded = action_group_ids_for_actions.unsqueeze(1).expand_as(action_emb)
    pooled.scatter_add_(0, gids_expanded, action_emb)
    counts.scatter_add_(
        0,
        action_group_ids_for_actions.unsqueeze(1),
        torch.ones(action_emb.shape[0], 1, device=device),
    )
    pooled = pooled / counts.clamp(min=1)

    return pooled


# ============================================================================
# AGILE building blocks (within-action encoder + complete-graph relational layer)
# ============================================================================

class _WithinGroupSelfAttnLayer(nn.Module):
    """Self-attention restricted to within an action group, then FFN.

    Tokens are masked so that token ``i`` can only attend to token ``j`` when
    they share the same ``action_group_id``.  For flat actions (group size 1)
    this collapses to "each token attends to itself" — i.e. a per-token MLP,
    exactly matching the independent-action assumption.  Builds the *atomic*
    action node consumed by the relational layer below.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, block_mask: Tensor) -> Tensor:
        """x: [1, A, d];  block_mask: [A, A] bool, True = NOT allowed to attend."""
        sa_out, _ = self.self_attn(x, x, x, attn_mask=block_mask)
        # A row that is fully masked (a singleton group) yields NaN from
        # softmax over an all -inf row; zero those out.
        sa_out = sa_out.nan_to_num(0.0)
        x = self.norm1(x + sa_out)
        x = self.norm2(x + self.ffn(x))
        return x


class _FullSelfAttnLayer(nn.Module):
    """Complete-graph self-attention + FFN with an optional additive bias.

    Structurally identical to ``nn.TransformerEncoderLayer`` (post-norm, same
    FFN), but built on ``nn.MultiheadAttention`` directly so a custom additive
    attention bias (the padding mask of the batched path) is applied *reliably*
    in both train and eval mode.  (``nn.TransformerEncoder``'s eval fast path
    silently drops a float ``mask``.)
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        """x: [1, N, d];  attn_bias: [nhead, N, N] additive bias or None."""
        sa_out, _ = self.self_attn(x, x, x, attn_mask=attn_bias)
        x = self.norm1(x + sa_out)
        x = self.norm2(x + self.ffn(x))
        return x


# ============================================================================
# AGILE policy net — the project's policy network
#
# Relational net: complete-graph attention over *atomic* action nodes (pool
# each action to one vector, then relate them, plus a role-tagged state node).
# Faithful in-framework port of AGILE (Jain et al., ICLR 2022, "Know Your
# Action Set").
# ============================================================================

class TransformerPolicyNet(nn.Module):
    """Relational (AGILE) policy net: complete-graph attention over atomic actions.

    Pipeline (per observation): per-type projection + type/positional
    embeddings → context self-attention → mean-pool to a state vector →
    within-action self-attention → mean-pool each group to one atomic action
    node → complete-graph self-attention over the action nodes + a role-tagged
    state node → score each action through a shared head.
    """

    def __init__(
        self,
        raw_dims: Dict[str, int],
        type_names: List[str],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        max_group_size: int = 32,
    ):
        super().__init__()
        self.type_names = list(type_names)
        self.d_model = d_model
        self.nhead = nhead

        self.input_projs = nn.ModuleDict()
        for tname in type_names:
            self.input_projs[tname] = nn.Linear(raw_dims[tname], d_model)

        self.type_embeddings = nn.Embedding(len(type_names), d_model)
        self.intra_group_pos_emb = nn.Embedding(max_group_size, d_model)

        # State encoder — self-attention over context tokens.
        ctx_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(ctx_layer, num_layers=num_layers)

        # Within-action encoder — atomic action representation; no inter-action
        # flow yet (block-diagonal over action_group_id).
        self.action_layers = nn.ModuleList([
            _WithinGroupSelfAttnLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # The AGILE bit: complete-graph attention over the pooled atomic action
        # nodes plus a state node.
        self.inter_action_layers = nn.ModuleList([
            _FullSelfAttnLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        # Distinguishes the state ("environment") node from action nodes.
        self.node_role_emb = nn.Embedding(2, d_model)  # 0 = action, 1 = state

        # Scoring head: [relational action repr ; global state vector].
        self.action_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    # ------------------------------------------------------------------
    # Dynamic-type discovery — add an input projection + type-embedding row.
    # ------------------------------------------------------------------
    def extend_for_new_type(self, tname: str, raw_dim: int) -> bool:
        if tname in self.input_projs:
            return False

        device = next(self.parameters()).device

        new_proj = nn.Linear(raw_dim, self.d_model).to(device)
        nn.init.xavier_uniform_(new_proj.weight)
        nn.init.zeros_(new_proj.bias)
        self.input_projs[tname] = new_proj
        self.type_names.append(tname)

        old_emb = self.type_embeddings
        new_num = old_emb.num_embeddings + 1
        new_emb = nn.Embedding(new_num, self.d_model).to(device)
        with torch.no_grad():
            new_emb.weight[:old_emb.num_embeddings] = old_emb.weight.data
            nn.init.normal_(new_emb.weight[old_emb.num_embeddings:], std=0.02)
        self.type_embeddings = new_emb
        return True

    # ------------------------------------------------------------------
    # Core single-sequence logit computation.
    # ------------------------------------------------------------------
    def _core(
        self,
        type_features: Dict[str, Tensor],
        type_ids: Tensor,
        action_mask: Tensor,
        indices_by_type: Dict[str, List[int]],
        action_group_ids: Optional[Tensor],
        num_action_groups: Optional[int],
        intra_group_pos: Optional[Tensor],
    ) -> Tensor:
        embeddings = _build_embeddings(
            self.input_projs, self.type_embeddings,
            type_features, type_ids, indices_by_type, self.d_model,
        )

        action_idx = action_mask.nonzero(as_tuple=True)[0]
        context_idx = (~action_mask).nonzero(as_tuple=True)[0]

        if action_idx.numel() == 0:
            return torch.empty(0, device=embeddings.device)

        # Intra-group positional embeddings on action tokens.
        if intra_group_pos is not None:
            pos_ids = intra_group_pos[action_idx].clamp(
                max=self.intra_group_pos_emb.num_embeddings - 1)
            embeddings[action_idx] = (
                embeddings[action_idx] + self.intra_group_pos_emb(pos_ids))

        # ---- Global state vector from context tokens (no action info) ----
        if context_idx.numel() > 0:
            ctx = self.context_encoder(embeddings[context_idx].unsqueeze(0))
            state_vec = ctx.squeeze(0).mean(dim=0)  # [d]
        else:
            state_vec = torch.zeros(self.d_model, device=embeddings.device)

        # ---- Within-action encoding (atomic action node), then pool ----
        act = embeddings[action_idx].unsqueeze(0)  # [1, A, d]

        if action_group_ids is not None:
            gids = action_group_ids[action_idx]  # [A]
        else:
            gids = torch.arange(action_idx.numel(), device=embeddings.device)

        # block_mask[i, j] == True → token i may NOT attend to token j.
        block_mask = gids.unsqueeze(1) != gids.unsqueeze(0)  # [A, A] bool
        for layer in self.action_layers:
            act = layer(act, block_mask)
        act = act.squeeze(0)  # [A, d]

        if action_group_ids is not None and num_action_groups is not None:
            pooled = _pool_action_groups(act, gids, num_action_groups)  # [G, d]
        else:
            pooled = act  # [A, d], A == G
        G = pooled.shape[0]
        device = pooled.device

        # ---- AGILE: complete-graph attention over [state_node ; actions] ----
        action_nodes = pooled + self.node_role_emb(
            torch.zeros(G, dtype=torch.long, device=device))
        state_node = (state_vec + self.node_role_emb(
            torch.ones(1, dtype=torch.long, device=device))).view(1, self.d_model)
        nodes = torch.cat([state_node, action_nodes], dim=0)  # [G+1, d]

        nodes = nodes.unsqueeze(0)  # [1, G+1, d]
        for layer in self.inter_action_layers:
            nodes = layer(nodes)
        nodes = nodes.squeeze(0)
        rel_actions = nodes[1:]  # [G, d] — drop the state node

        # ---- Scoring: head([relational action repr ; state]) ----
        state_rep = state_vec.unsqueeze(0).expand(G, -1)
        feat = torch.cat([rel_actions, state_rep], dim=-1)
        logits = self.action_head(feat).squeeze(-1)  # [G]
        return logits

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def forward_logits(
        self,
        type_features: Dict[str, Tensor],
        type_ids: Tensor,
        action_mask: Tensor,
        indices_by_type: Dict[str, List[int]],
        action_group_ids: Optional[Tensor] = None,
        num_action_groups: Optional[int] = None,
        intra_group_pos: Optional[Tensor] = None,
    ) -> Tensor:
        return self._core(
            type_features, type_ids, action_mask, indices_by_type,
            action_group_ids, num_action_groups, intra_group_pos,
        )

    def forward_logits_batched(
        self,
        seq_dicts: List[Dict[str, Any]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Batched forward over B sequences → SAT flat/offsets/lengths contract.

        Every heavy op (context encoder, within-action encoder, inter-action
        complete-graph attention, head) runs **once** over the whole batch with
        padding + masks.  Numerically matches the single-sample
        ``forward_logits`` (padded keys are excluded via the block/padding
        masks), as the parity test guards.
        """
        B = len(seq_dicts)
        device = next(self.parameters()).device
        if B == 0:
            empty = torch.empty(0, device=device)
            z = torch.empty(0, dtype=torch.long, device=device)
            return empty, z, z
        return self._forward_logits_batched_padded(seq_dicts, device)

    # ------------------------------------------------------------------
    # Padded batched core (variable action-token AND action-group counts).
    # ------------------------------------------------------------------
    def _forward_logits_batched_padded(
        self,
        seq_dicts: List[Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B = len(seq_dicts)
        d = self.d_model
        nh = self.nhead

        # --- 1. Per-sample embeddings + context/action split (cheap) ---
        valid: List[Dict[str, Any]] = []   # one entry per sample that has actions
        for i, seq in enumerate(seq_dicts):
            if seq is None or seq.get("num_action_groups", 0) == 0:
                continue
            amask = seq["action_mask"]
            act_idx = amask.nonzero(as_tuple=True)[0]
            if act_idx.numel() == 0:
                continue
            emb = _build_embeddings(
                self.input_projs, self.type_embeddings,
                seq["type_features"], seq["type_ids"],
                seq["indices_by_type"], d,
            )
            igp = seq.get("intra_group_pos")
            if igp is not None:
                pos = igp[act_idx].clamp(
                    max=self.intra_group_pos_emb.num_embeddings - 1)
                emb[act_idx] = emb[act_idx] + self.intra_group_pos_emb(pos)
            ctx_idx = (~amask).nonzero(as_tuple=True)[0]
            agids = seq.get("action_group_ids")
            if agids is not None:
                gids = agids[act_idx].to(device)
                G = int(seq["num_action_groups"])
            else:
                gids = torch.arange(act_idx.numel(), device=device)
                G = int(act_idx.numel())
            valid.append(dict(orig=i, ctx=emb[ctx_idx], act=emb[act_idx],
                              gids=gids, G=G))

        V = len(valid)
        if V == 0:
            flat = torch.empty(0, device=device)
            zeros = torch.zeros(B, dtype=torch.long, device=device)
            return flat, zeros.clone(), zeros

        C_list = [v["ctx"].shape[0] for v in valid]
        A_list = [v["act"].shape[0] for v in valid]
        G_list = [v["G"] for v in valid]
        max_C = max(1, max(C_list))
        max_A = max(A_list)
        max_G = max(G_list)
        G_t = torch.tensor(G_list, device=device).unsqueeze(1)          # [V,1]
        valid_g = torch.arange(max_G, device=device).unsqueeze(0) < G_t  # [V,max_G]

        # --- 2. Context encoder (batched) → per-sample state vector ---
        ctx_padded = torch.zeros(V, max_C, d, device=device)
        ctx_mask = torch.ones(V, max_C, dtype=torch.bool, device=device)  # True=pad
        for k, v in enumerate(valid):
            c = C_list[k]
            if c > 0:
                ctx_padded[k, :c] = v["ctx"]
                ctx_mask[k, :c] = False
        ctx_enc = self.context_encoder(ctx_padded, src_key_padding_mask=ctx_mask)
        ctx_enc = ctx_enc.nan_to_num(0.0)
        vmask = (~ctx_mask).unsqueeze(-1).float()
        state_vec = (ctx_enc * vmask).sum(1) / vmask.sum(1).clamp(min=1)   # [V,d]

        # --- 3. Within-action encoder (block-diagonal over groups, batched) ---
        act_padded = torch.zeros(V, max_A, d, device=device)
        block = torch.ones(V, max_A, max_A, dtype=torch.bool, device=device)  # True=blocked
        gids_pad = torch.full((V, max_A), max_G, dtype=torch.long, device=device)
        for k, v in enumerate(valid):
            A = A_list[k]
            act_padded[k, :A] = v["act"]
            g = v["gids"]
            block[k, :A, :A] = g.unsqueeze(1) != g.unsqueeze(0)  # same-group → attend
            gids_pad[k, :A] = g
        # Every token attends to itself so no query row is fully -inf.  For real
        # tokens this is a no-op (self shares its group); for padded tokens it
        # avoids an all-masked softmax row whose internal NaN would back-prop as
        # 0*NaN=NaN into the shared QKV weights.  Padded outputs are discarded.
        eye = torch.eye(max_A, dtype=torch.bool, device=device)
        block = block & ~eye
        block_exp = block.unsqueeze(1).expand(V, nh, max_A, max_A).reshape(
            V * nh, max_A, max_A)
        act_enc = act_padded
        for layer in self.action_layers:
            act_enc = layer(act_enc, block_exp)

        # --- 4. Pool tokens → atomic action nodes (padded tokens → dummy bin) ---
        pooled_full = torch.zeros(V, max_G + 1, d, device=device)
        counts_full = torch.zeros(V, max_G + 1, 1, device=device)
        pooled_full.scatter_add_(
            1, gids_pad.unsqueeze(-1).expand(V, max_A, d), act_enc)
        counts_full.scatter_add_(
            1, gids_pad.unsqueeze(-1), torch.ones(V, max_A, 1, device=device))
        pooled = pooled_full[:, :max_G] / counts_full[:, :max_G].clamp(min=1)  # [V,max_G,d]

        # --- 5. Role-tagged nodes ---
        state_bG = state_vec.unsqueeze(1).expand(V, max_G, d)
        action_nodes = pooled + self.node_role_emb(
            torch.zeros(1, dtype=torch.long, device=device))
        state_node = (state_vec + self.node_role_emb(
            torch.ones(1, dtype=torch.long, device=device))).unsqueeze(1)   # [V,1,d]
        nodes = torch.cat([state_node, action_nodes], dim=1)               # [V,max_G+1,d]

        # --- 6. Inter-action attention with padding mask over key nodes ---
        N = max_G + 1
        attn_bias = torch.zeros(V, nh, N, N, device=device)
        # Mask padded key nodes (state node always valid; action node g valid iff g<G)
        node_valid = torch.zeros(V, N, dtype=torch.bool, device=device)
        node_valid[:, 0] = True
        node_valid[:, 1:] = valid_g
        attn_bias = attn_bias.masked_fill(
            (~node_valid).view(V, 1, 1, N), float("-inf")).reshape(V * nh, N, N)

        nodes_enc = nodes
        for layer in self.inter_action_layers:
            nodes_enc = layer(nodes_enc, attn_bias)
        rel_actions = nodes_enc[:, 1:1 + max_G, :].nan_to_num(0.0)          # [V,max_G,d]

        # --- 7. Scoring head ---
        logits_full = self.action_head(
            torch.cat([rel_actions, state_bG], dim=-1)).squeeze(-1)        # [V,max_G]

        # --- 8. Scatter back into the flat/offsets/lengths contract ---
        per: List[Optional[Tensor]] = [None] * B
        for k, v in enumerate(valid):
            per[v["orig"]] = logits_full[k, :G_list[k]]
        chunks: List[Tensor] = []
        offsets: List[int] = []
        lengths: List[int] = []
        cum = 0
        for i in range(B):
            offsets.append(cum)
            if per[i] is None:
                lengths.append(0)
                continue
            chunks.append(per[i])
            lengths.append(int(per[i].shape[0]))
            cum += int(per[i].shape[0])
        flat = torch.cat(chunks, dim=0) if chunks else torch.empty(0, device=device)
        return (flat,
                torch.tensor(offsets, dtype=torch.long, device=device),
                torch.tensor(lengths, dtype=torch.long, device=device))


# ============================================================================
# Value net (self-attention over all tokens)
# ============================================================================

class TransformerValueNet(nn.Module):
    """Value network using self-attention over ALL tokens.

    Unlike the policy net (where actions relate through pooled nodes), the value
    net needs to see everything — including action tokens which may carry
    essential state information (e.g. bin weights in bin packing).
    """

    def __init__(
        self,
        raw_dims: Dict[str, int],
        type_names: List[str],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        max_group_size: int = 32,
    ):
        super().__init__()
        self.type_names = list(type_names)
        self.d_model = d_model

        self.input_projs = nn.ModuleDict()
        for tname in type_names:
            self.input_projs[tname] = nn.Linear(raw_dims[tname], d_model)

        self.type_embeddings = nn.Embedding(len(type_names), d_model)
        # Value net also accepts intra-group positional embeddings so it
        # observes the same ordering signal for action tokens as the policy.
        # Kept optional and backward-compatible; max_group_size matches the
        # policy default.
        self.intra_group_pos_emb = nn.Embedding(max_group_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def extend_for_new_type(self, tname: str, raw_dim: int) -> bool:
        """Dynamically add an input projection + type embedding row for *tname*.

        Mirror of ``TransformerPolicyNet.extend_for_new_type``; see that
        docstring for details.
        """
        if tname in self.input_projs:
            return False

        device = next(self.parameters()).device

        new_proj = nn.Linear(raw_dim, self.d_model).to(device)
        nn.init.xavier_uniform_(new_proj.weight)
        nn.init.zeros_(new_proj.bias)
        self.input_projs[tname] = new_proj
        self.type_names.append(tname)

        old_emb = self.type_embeddings
        new_num = old_emb.num_embeddings + 1
        new_emb = nn.Embedding(new_num, self.d_model).to(device)
        with torch.no_grad():
            new_emb.weight[:old_emb.num_embeddings] = old_emb.weight.data
            nn.init.normal_(new_emb.weight[old_emb.num_embeddings:], std=0.02)
        self.type_embeddings = new_emb

        return True

    def forward(
        self,
        type_features: Dict[str, Tensor],
        type_ids: Tensor,
        indices_by_type: Dict[str, List[int]],
        action_mask: Tensor | None = None,
        intra_group_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute state value from ALL tokens (full self-attention)."""
        embeddings = _build_embeddings(
            self.input_projs, self.type_embeddings,
            type_features, type_ids, indices_by_type, self.d_model,
        )
        # Add intra-group positional embeddings to action tokens when provided
        # (keeps behaviour identical when not used).
        if intra_group_pos is not None and action_mask is not None:
            action_positions = action_mask.nonzero(as_tuple=True)[0]
            if action_positions.numel() > 0:
                pos_ids = intra_group_pos[action_positions]
                pos_ids = pos_ids.clamp(max=self.intra_group_pos_emb.num_embeddings - 1)
                embeddings[action_positions] = (
                    embeddings[action_positions] + self.intra_group_pos_emb(pos_ids)
                )
        x = self.encoder(embeddings.unsqueeze(0)).squeeze(0)
        pooled = x.mean(dim=0, keepdim=True)
        value = self.value_head(pooled).squeeze(-1)
        return value

    def forward_batched(
        self,
        seq_dicts: List[Dict[str, Any]],
    ) -> Tensor:
        """Compute state values for a batch of sequences in one pass.

        Uses ``_build_embeddings_batched`` when all samples share the same
        sequence structure (common case), falling back to per-sample
        ``_build_embeddings`` otherwise.

        Returns
        -------
        values : Tensor[B]
            Scalar state value per sample.
        """
        B = len(seq_dicts)
        device = next(self.parameters()).device
        if B == 0:
            return torch.empty(0, device=device)

        # Detect uniform structure (same seq_len AND same per-type token counts)
        seq_lens = [seq["type_ids"].shape[0] for seq in seq_dicts]
        uniform = seq_lens.count(seq_lens[0]) == B

        # Also verify per-type token counts match across the batch; variable-
        # length context lists can produce the same total seq_len but different
        # per-type counts, which breaks the batched stacking in the fast path.
        if uniform and B > 1:
            ref_tf = seq_dicts[0]["type_features"]
            for seq in seq_dicts[1:]:
                for tname, ftens in seq["type_features"].items():
                    if tname not in ref_tf or ftens.shape[0] != ref_tf[tname].shape[0]:
                        uniform = False
                        break
                if not uniform:
                    break

        if uniform:
            # Fast path: one projection per type for the whole batch
            padded, mask = _build_embeddings_batched(
                self.input_projs, self.type_embeddings,
                seq_dicts, self.d_model, device,
            )
            # Batched intra-group positional embeddings (same layout for all)
            ref = seq_dicts[0]
            igp = ref.get("intra_group_pos")
            act_mask = ref.get("action_mask")
            if igp is not None and act_mask is not None:
                act_pos = act_mask.nonzero(as_tuple=True)[0]
                if act_pos.numel() > 0:
                    pos_ids = igp[act_pos].clamp(max=self.intra_group_pos_emb.num_embeddings - 1)
                    pos_emb = self.intra_group_pos_emb(pos_ids)  # [n_act, d]
                    padded[:, act_pos] = padded[:, act_pos] + pos_emb.unsqueeze(0)
        else:
            # General path: per-sample embedding + pad
            embs: List[Tensor] = []
            for seq in seq_dicts:
                embs.append(_build_embeddings(
                    self.input_projs, self.type_embeddings,
                    seq["type_features"], seq["type_ids"],
                    seq["indices_by_type"], self.d_model,
                ))
                # Per-sample intra-group positional embeddings
                igp = seq.get("intra_group_pos")
                action_mask = seq.get("action_mask")
                if igp is not None and action_mask is not None:
                    act_pos = action_mask.nonzero(as_tuple=True)[0]
                    if act_pos.numel() > 0:
                        pos_ids = igp[act_pos].clamp(max=self.intra_group_pos_emb.num_embeddings - 1)
                        embs[-1][act_pos] = embs[-1][act_pos] + self.intra_group_pos_emb(pos_ids)
            seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device)
            if all(s > 0 for s in seq_lens):
                padded = torch.nn.utils.rnn.pad_sequence(
                    embs, batch_first=True, padding_value=0.0)
            else:
                safe = [e if e.shape[0] > 0 else torch.zeros(1, self.d_model, device=device) for e in embs]
                padded = torch.nn.utils.rnn.pad_sequence(
                    safe, batch_first=True, padding_value=0.0)
            max_len = padded.shape[1]
            mask = torch.arange(max_len, device=device).unsqueeze(0) >= seq_lens_t.unsqueeze(1)

        # Batched self-attention encoder
        x = self.encoder(padded, src_key_padding_mask=mask)
        x = x.nan_to_num(0.0)

        # Masked mean-pool
        valid = (~mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1)
        pooled = pooled / valid.sum(dim=1).clamp(min=1)

        values = self.value_head(pooled).squeeze(-1)
        return values


# ============================================================================
# Factory functions
# ============================================================================

def build_transformer_policy_net(
    raw_dims: Dict[str, int],
    type_names: List[str],
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
) -> TransformerPolicyNet:
    """Create the AGILE relational policy network."""
    return TransformerPolicyNet(
        raw_dims=raw_dims,
        type_names=type_names,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )


def build_transformer_value_net(
    raw_dims: Dict[str, int],
    type_names: List[str],
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
) -> TransformerValueNet:
    """Create a Transformer value network."""
    return TransformerValueNet(
        raw_dims=raw_dims,
        type_names=type_names,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )