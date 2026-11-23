"""Critic networks. Only PPO needs them (`train_ppo.py`); Deep Controlled
Learning has no critic.

PPO trains two networks: the actor, which scores the actions and becomes the
policy, and the critic, which estimates the cost that is still to come from a
state. The critic only helps during training and is not part of the trained
agent. For a featurizer with one flat vector DynaPlex builds a critic itself;
for the graph and the token featurizers it needs one from here, handed over as
`critic_network=dp.Net("critic_networks.<factory>", **kwargs)`.

A critic factory is `factory(spec, num_actions, **kwargs) -> nn.Module` like
an actor factory, called with num_actions = 1: the module returns one value
per state, [B, 1].

Both critics here are the actor's network with one change at the end: the
actor gives one score per node (or per move token), and the critic takes the
mean of those as the value of the state.
"""
from __future__ import annotations

from torch import nn

import networks


class _MeanOfScores(nn.Module):
    """One value per state: the mean of the scores of `scorer` ([B, K] -> [B, 1])."""

    def __init__(self, scorer: nn.Module):
        super().__init__()
        self.scorer = scorer

    def forward(self, batch):
        return self.scorer(batch).mean(dim=1, keepdim=True)


def gin_critic_factory(spec: dict, num_actions: int, *, d_model: int = 64,
                       n_layers: int = 6) -> nn.Module:
    """Critic for GridGraphFeaturizer: the GIN of `networks.gin_factory`, one
    score per node, averaged over the nodes."""
    n_nodes = spec["nodes"].sample_shape[0]
    return _MeanOfScores(networks.gin_factory(spec, n_nodes, d_model=d_model, n_layers=n_layers))


def move_token_critic_factory(spec: dict, num_actions: int, *, d_model: int = 64,
                              n_layers: int = 2, n_heads: int = 4) -> nn.Module:
    """Critic for MoveTokenFeaturizer: the network of
    `networks.move_token_factory`, one score per move token, averaged."""
    n_moves = spec["candidates"].sample_shape[0]
    return _MeanOfScores(networks.move_token_factory(
        spec, n_moves, d_model=d_model, n_layers=n_layers, n_heads=n_heads))
