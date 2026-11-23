"""The networks, one factory per architecture, consumed by DCL/PPO via
`Net("networks.<factory>", **kwargs)` — a dotted path plus JSON kwargs, so an
agent artifact can rebuild its network on any machine.

Every factory is `factory(spec, num_actions, **kwargs) -> nn.Module`, where
`spec` is the featurizer's observation spec (dict of TensorSpec, per-sample
shapes) and the module's forward takes the batch dict keyed like the spec and
returns [B, num_actions] logits. A network never reads the validity mask:
DCL training and acting crop the logits to the valid actions downstream, so a
network must emit finite values for every action.

The two networks answer in different action representations (README, "The
action representation"):

  gin_factory            one node per cell on the lattice, GIN message
                         passing, one logit per node = per CELL action
                                                           (featurizer_graph)
  move_token_factory     one token per move plus context tokens, one logit
                         per move token = per DIRECTIONAL action
                                                           (featurizer_moves)
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _embed(d_in: int, d_model: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, d_model), nn.ReLU(), nn.Linear(d_model, d_model))


# ============================================================================
# GIN on the grid graph (pairs with featurizer_graph.GridGraphFeaturizer)
# ============================================================================


def _lattice_adjacency(g: int) -> torch.Tensor:
    """[G*G, G*G] 0/1 matrix of the 4-neighbour grid graph, row-major cells."""
    adj = torch.zeros(g * g, g * g)
    for r in range(g):
        for c in range(g):
            i = r * g + c
            if r > 0:
                adj[i, i - g] = 1.0
            if r < g - 1:
                adj[i, i + g] = 1.0
            if c > 0:
                adj[i, i - 1] = 1.0
            if c < g - 1:
                adj[i, i + 1] = 1.0
    return adj


class _GINNet(nn.Module):
    """Graph Isomorphism Network (Xu et al. 2019) on the grid graph, adapted
    from the "Node Actions" architecture of Begnardi et al. (2023) to one-step
    move actions. An input projection, then `n_layers` residual rounds of

        x_i <- x_i + MLP(x_i + sum_{j adjacent to i} M(x_j)),

    where M is a per-layer MLP applied to every neighbour before the sum (the
    paper sums raw neighbours and weights the node itself by 1 + eps; here the
    self weight is 1 and M does the shaping). Because M is shared over nodes,
    it runs once on the whole node matrix and the sum over neighbours is one
    matmul with the fixed lattice adjacency (a buffer, not a parameter).

    Readout: every node is scored by a shared MLP from three vectors — its own
    embedding, the mean over all node embeddings (a global summary of the
    order field), and the embedding of the deciding vehicle's node (node
    feature 0 marks it) — so a candidate cell is scored as an origin–
    destination pair in the context of the whole grid. Nodes are written in
    row-major cell order by the featurizer, so the [B, G^2] logits are the
    CELL action logits; the validity mask crops them downstream.

    Why not the paper's layer as is: in the paper a destination can be far
    away and is itself a node that matters, so three hops of context around
    the scored node suffice. Here the admissible destinations are the
    vehicle's own cell and its neighbours: they have nearly the same few-hop
    view, and an order six cells away is invisible to all of them — hence six
    layers, residual connections to train them, and the two readouts.
    """

    def __init__(self, spec, num_actions: int, d_model: int, n_layers: int,
                 hidden_head: int):
        super().__init__()
        n_nodes, d_in = spec["nodes"].sample_shape
        assert num_actions == n_nodes, (
            f"gin_factory pairs nodes 1:1 with CELL actions: num_actions={num_actions} "
            f"but the featurizer declares {n_nodes} nodes")
        g = math.isqrt(n_nodes)
        assert g * g == n_nodes, "the node set must be a square grid"
        self.register_buffer("adj", _lattice_adjacency(g))
        self.inp = nn.Linear(d_in, d_model)

        def mlp():
            return nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                 nn.Linear(d_model, d_model))

        self.messages = nn.ModuleList(mlp() for _ in range(n_layers))
        self.updates = nn.ModuleList(mlp() for _ in range(n_layers))
        self.head = nn.Sequential(nn.Linear(3 * d_model, hidden_head), nn.ReLU(),
                                  nn.Linear(hidden_head, 1))

    def embed(self, nodes):
        """Node embeddings after message passing: [B, N, d]. A node's embedding
        depends on nodes within `n_layers` lattice hops and on nothing else."""
        x = torch.relu(self.inp(nodes))                             # [B, N, d]
        for message, update in zip(self.messages, self.updates):
            neighbours = torch.matmul(self.adj, message(x))         # [N, N] @ [B, N, d]
            x = x + torch.relu(update(x + neighbours))
        return x

    def forward(self, batch):
        nodes = batch["nodes"]
        x = self.embed(nodes)
        n = x.shape[1]
        graph = x.mean(dim=1, keepdim=True).expand(-1, n, -1)
        focal = nodes[:, :, 0:1]                                    # 1.0 on the deciding vehicle's node
        origin = (x * focal).sum(dim=1, keepdim=True).expand(-1, n, -1)
        return self.head(torch.cat([x, graph, origin], dim=-1)).squeeze(-1)  # [B, N]


def gin_factory(spec: dict, num_actions: int, *, d_model: int = 64, n_layers: int = 6,
                hidden_head: int = 64) -> nn.Module:
    """Factory entry point: `Net("networks.gin_factory", ...)`. Consumes
    GridGraphFeaturizer's `nodes` on a CELL-encoded mdp. Six layers are needed
    on the 10x10 grid (three leave the policy at myopic level); train with
    lr=1e-3, ten times DynaPlex's default."""
    return _GINNet(spec, num_actions, d_model, n_layers, hidden_head)


# ============================================================================
# Move-token net (pairs with featurizer_moves.MoveTokenFeaturizer)
# ============================================================================


class _MoveTokenNet(nn.Module):
    """Score the deciding vehicle's five moves against the state.

    Tokens: the 5 move tokens first (stay, up, down, left, right), then depot,
    vehicles and the (padded, masked) orders. One transformer encoder over all
    of them; a shared head scores each move token. Those five scores are the
    output: on a DIRECTIONAL mdp action k is move k, so token k answers for
    action k. A move that would leave the grid gets a score too; the validity
    mask rejects it downstream.

    `interact=False` forbids move tokens from attending to each other (each
    still attends to itself and to the context), so every move is scored
    independently of which other moves are available; `interact=True`
    (default) lets the moves condition on one another, the relational step of
    AGILE (Jain et al.). The difference is what the switch measures.
    """

    def __init__(self, spec, num_actions: int, d_model: int, n_layers: int,
                 n_heads: int, ff: int, hidden_head: int, interact: bool):
        super().__init__()
        self.n_cand = spec["candidates"].sample_shape[0]
        assert num_actions == self.n_cand, (
            f"move_token_factory scores one token per move: num_actions={num_actions} "
            f"but there are {self.n_cand} move tokens (use the DIRECTIONAL action encoding)")
        self.max_n = spec["orders"].sample_shape[0]
        self.interact = interact
        self.emb_candidates = _embed(spec["candidates"].sample_shape[1], d_model)
        self.emb_orders = _embed(spec["orders"].sample_shape[1], d_model)
        self.emb_vehicles = _embed(spec["vehicles"].sample_shape[1], d_model)
        self.emb_depot = _embed(spec["depot"].sample_shape[1], d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, hidden_head), nn.ReLU(),
                                  nn.Linear(hidden_head, 1))

    def forward(self, batch):
        cand, orders, counts = batch["candidates"], batch["orders"], batch["orders_count"]
        vehicles, depot = batch["vehicles"], batch["depot"]
        B = cand.shape[0]
        dev = cand.device
        tokens = torch.cat([self.emb_candidates(cand), self.emb_depot(depot),
                            self.emb_vehicles(vehicles), self.emb_orders(orders)], dim=1)
        T = tokens.shape[1]
        # key padding: everything real except orders past the count
        pad = torch.zeros(B, T, dtype=torch.bool, device=dev)
        pad[:, T - self.max_n:] = (torch.arange(self.max_n, device=dev)[None, :]
                                   >= counts[:, None])
        attn_mask = None
        if not self.interact:
            # True = may NOT attend. Block move token -> other move token.
            attn_mask = torch.zeros(T, T, dtype=torch.bool, device=dev)
            attn_mask[:self.n_cand, :self.n_cand] = ~torch.eye(
                self.n_cand, dtype=torch.bool, device=dev)
        h = self.encoder(tokens, mask=attn_mask, src_key_padding_mask=pad)
        return self.head(h[:, :self.n_cand]).squeeze(-1)     # [B, 5]: one score per move


def move_token_factory(spec: dict, num_actions: int, *, d_model: int = 64, n_layers: int = 2,
                       n_heads: int = 4, ff: int = 128, hidden_head: int = 64,
                       interact: bool = True) -> nn.Module:
    """Factory entry point: `Net("networks.move_token_factory", ...)`. Consumes
    MoveTokenFeaturizer's bundle on a DIRECTIONAL-encoded mdp; `interact` is
    the move-to-move attention switch (see _MoveTokenNet)."""
    return _MoveTokenNet(spec, num_actions, d_model, n_layers, n_heads, ff,
                         hidden_head, interact)
