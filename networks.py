"""Dot-attention network factory (WvJ design 2026-08-14), consumed by DCL via
`Net("networks.attention_net", ...)` — dotted path from the repo root.

Token bundle in (see featurizers_tokens.TokenFeaturizer): `orders`
[B, max_n, 2] + `orders_count` [B], `vehicles` [B, V, 4] (feature 3 = is-focal
flag), `depot` [B, 1, 2]. Each token type gets its OWN small embedding MLP to d_model, the
stack attends over [depot | vehicles | orders] with padded orders masked via
`orders_count` (mask = arange(max_n) >= count), and the head reads the FOCAL
vehicle's output vector.

Heads (`head=` kwarg) x action encodings (mdp.ActionEncoding, read off
`num_actions`):

  * head="moves5" — an MLP from the focal vehicle's vector to 5 move logits
    (stay/up/down/left/right; the DIRECTIONAL action order).
      - DIRECTIONAL mdp (num_actions == 5): the 5 logits ARE the action
        logits — returned directly. The natural pairing.
      - CELL mdp (num_actions == G^2): the 5 logits are SCATTERED into the
        cell-action space at the focal vehicle's stay/neighbor cells, read
        off the focal token's (row, col) features. Off-grid moves contribute
        nothing and every other cell sits at -1e9 (the validity mask would
        kill them anyway; the -1e9 keeps the net's output well-defined
        without it). This adapter exists so the head can train on
        cell-encoded labels (e.g. pre-existing sample sets) — prefer the
        DIRECTIONAL mdp when collecting fresh.
  * head="map" — a plain MLP from the focal vector to all num_actions logits
    (with a CELL mdp: a G^2 desirability map, the closest apples-to-apples
    against the flat-MLP baseline). Works with either encoding.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _embed(d_in: int, d_model: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, d_model), nn.ReLU(), nn.Linear(d_model, d_model))


class _AttentionNet(nn.Module):
    def __init__(self, spec, num_actions: int, d_model: int, n_layers: int,
                 n_heads: int, ff: int, head: str, hidden_head: int):
        super().__init__()
        # num_actions identifies the mdp's action encoding: 5 = DIRECTIONAL
        # (moves5 logits are the action logits), a square = CELL (moves5
        # scatters into the grid; map emits the G^2 desirability map).
        self.directional = num_actions == 5
        if not self.directional:
            self.g = math.isqrt(num_actions)
            assert self.g * self.g == num_actions, \
                "CELL action space must be a square grid (or 5 for DIRECTIONAL)"
        self.max_n = spec["orders"].sample_shape[0]
        self.head_kind = head
        self.emb_orders = _embed(spec["orders"].sample_shape[1], d_model)
        self.emb_vehicles = _embed(spec["vehicles"].sample_shape[1], d_model)
        self.emb_depot = _embed(spec["depot"].sample_shape[1], d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        out = 5 if head == "moves5" else num_actions
        self.head = nn.Sequential(nn.Linear(d_model, hidden_head), nn.ReLU(),
                                  nn.Linear(hidden_head, out))

    def forward(self, batch):
        orders, counts = batch["orders"], batch["orders_count"]
        vehicles, depot = batch["vehicles"], batch["depot"]
        B = orders.shape[0]
        tokens = torch.cat([self.emb_depot(depot), self.emb_vehicles(vehicles),
                            self.emb_orders(orders)], dim=1)
        # key padding: depot + vehicles always real, orders masked past the count
        n_fixed = 1 + vehicles.shape[1]
        pad = torch.zeros(B, tokens.shape[1], dtype=torch.bool, device=tokens.device)
        pad[:, n_fixed:] = (torch.arange(self.max_n, device=tokens.device)[None, :]
                            >= counts[:, None])
        h = self.encoder(tokens, src_key_padding_mask=pad)
        # focal vehicle readout (feature 3 = is-focal flag, exactly one per row)
        focal_idx = vehicles[:, :, 3].argmax(dim=1)
        focal = h[torch.arange(B, device=h.device), 1 + focal_idx]
        logits = self.head(focal)
        if self.head_kind != "moves5" or self.directional:
            return logits   # map head, or moves5 on a DIRECTIONAL mdp (direct)
        # CELL mdp: scatter the 5 move logits into the G^2 cell-action space by
        # INDEXED ASSIGNMENT — never by adding large sentinels: an earlier
        # `-1e9 + (logit + 1e9)` construction quantized the logits to float32's
        # ~64-unit resolution at 1e9 and made training diverge in visible jumps.
        # Valid moves land on distinct cells, so the writes never collide;
        # off-grid moves are simply not written and their cells stay at -1e9
        # (the validity mask kills them anyway).
        gm1 = float(self.g - 1)
        row = (vehicles[torch.arange(B), focal_idx, 0] * gm1).round().long()
        col = (vehicles[torch.arange(B), focal_idx, 1] * gm1).round().long()
        out = torch.full((B, self.g * self.g), -1e9,
                         dtype=logits.dtype, device=logits.device)
        moves = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))   # stay/up/down/left/right
        rows = torch.arange(B, device=logits.device)
        for k, (dr, dc) in enumerate(moves):
            r, c = row + dr, col + dc
            ok = (r >= 0) & (r < self.g) & (c >= 0) & (c < self.g)
            cell = r * self.g + c
            out[rows[ok], cell[ok]] = logits[ok, k]
        return out


def attention_critic(spec: dict, num_actions: int, *, d_model: int = 64,
                     n_layers: int = 2, n_heads: int = 4, ff: int = 128,
                     hidden_head: int = 64) -> nn.Module:
    """PPO value head over the token bundle: the same attention trunk as
    `attention_net`, reading the focal vehicle's vector into a scalar value
    per sample ([B, 1]). Entry point for PPO's `critic_network` —
    `Net("networks.attention_critic", ...)` — which builds critics as
    `factory(spec, 1)`; `num_actions` is therefore always 1 here and the
    "map" head with one output IS the value head. Training machinery only:
    it lives in PPO's checkpoint, never in the agent artifact."""
    return _AttentionNet(spec, 1, d_model, n_layers, n_heads, ff,
                         "map", hidden_head)


def attention_net(spec: dict, num_actions: int, *, d_model: int = 64,
                  n_layers: int = 2, n_heads: int = 4, ff: int = 128,
                  head: str = "moves5", hidden_head: int = 64) -> nn.Module:
    """Factory entry point: `Net("networks.attention_net", d_model=..., ...)`.

    Args:
        spec: the observation spec (dict of TensorSpec) — must be
            TokenFeaturizer's bundle (orders/orders_count/vehicles/depot;
            extra entries are tolerated and ignored).
        num_actions: the mdp's action count; also selects the encoding
            (5 = DIRECTIONAL, a square = CELL — see the module docstring).
        d_model / n_layers / n_heads / ff: transformer size knobs.
        head: "moves5" (focal vehicle -> 5 move logits; the default) or
            "map" (focal vehicle -> all num_actions logits).
        hidden_head: hidden width of the head MLP.
    """
    return _AttentionNet(spec, num_actions, d_model, n_layers, n_heads, ff,
                         head, hidden_head)


class _AttentionPointerNet(nn.Module):
    """Grid-token pointer net (pairs with GridTokenFeaturizer + CELL encoding):
    every location is a token; the head scores each cell token with a shared
    MLP, and because the featurizer writes cell tokens in row-major cell order,
    the [B, G^2] scores ARE the action logits — the ordinary validity mask
    crops them downstream (masked CE / masked argmax); the network never
    handles masking. No padding mask either: all tokens are always real."""

    def __init__(self, spec, num_actions: int, d_model: int, n_layers: int,
                 n_heads: int, ff: int, hidden_head: int):
        super().__init__()
        n_cells = spec["cells"].sample_shape[0]
        assert num_actions == n_cells, (
            f"attention_pointer_net pairs cell tokens 1:1 with CELL actions: "
            f"num_actions={num_actions} but the featurizer declares {n_cells} "
            f"cell tokens (is the mdp using ActionEncoding.CELL?)")
        self.emb_cells = _embed(spec["cells"].sample_shape[1], d_model)
        self.emb_vehicles = _embed(spec["vehicles"].sample_shape[1], d_model)
        self.emb_depot = _embed(spec["depot"].sample_shape[1], d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, hidden_head), nn.ReLU(),
                                  nn.Linear(hidden_head, 1))

    def forward(self, batch):
        vehicles, depot, cells = batch["vehicles"], batch["depot"], batch["cells"]
        n_fixed = depot.shape[1] + vehicles.shape[1]
        tokens = torch.cat([self.emb_depot(depot), self.emb_vehicles(vehicles),
                            self.emb_cells(cells)], dim=1)
        h = self.encoder(tokens)
        # score each CELL token; token order == row-major cell order == action order
        return self.head(h[:, n_fixed:]).squeeze(-1)


def attention_pointer_net(spec: dict, num_actions: int, *, d_model: int = 64,
                          n_layers: int = 2, n_heads: int = 4, ff: int = 128,
                          hidden_head: int = 64) -> nn.Module:
    """Factory entry point: `Net("networks.attention_pointer_net", ...)`.
    Consumes GridTokenFeaturizer's bundle (cells/vehicles/depot) on a
    CELL-encoded mdp; see _AttentionPointerNet for the token-order contract."""
    return _AttentionPointerNet(spec, num_actions, d_model, n_layers, n_heads,
                                ff, hidden_head)
