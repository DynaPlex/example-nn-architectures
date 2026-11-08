"""Grid-graph featurizer for the GIN network: one node per grid cell.

Follows the "Node Actions" input of Begnardi, Baier, van Jaarsveld & Zhang
(2023), "Deep Reinforcement Learning for Two-sided Online Bipartite Matching
in Collaborative Order Picking", ACML (`begnardi24a.pdf`), Table 1: the grid
cells are the nodes of a graph whose edges join adjacent cells, and everything
the policy knows about vehicles and orders is written as NODE FEATURES. The
edges are the same for every state (the lattice is a property of the grid, not
of the state), so they are not features at all: `networks.gin_net` builds the
adjacency matrix once from the node count.

Node features, per cell, in this order:

  0  the deciding vehicle stands here                        (0/1)
  1  number of OTHER vehicles here / (n_vehicles - 1)        [0, 1]
  2  an order is waiting here                                (0/1)
  3  the depot is here                                       (0/1)
  4  Manhattan distance to the deciding vehicle / (2G - 2)   [0, 1]

Nodes are written in ROW-MAJOR CELL ORDER, so node i is cell i, which is also
action i under the CELL action encoding: a network that emits one logit per
node emits the action logits in the right order, and the validity mask crops
them to the deciding vehicle's neighborhood downstream.

`write_features` compiles into the engine (DynaML subset): for-range loops,
if/else, index arithmetic, float() casts, single comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from dynaplex.modelling import FixedObjectWriter, featurizer

from mdp import State, VehicleRoutingMDP

NODE_FEATURES: Final[int] = 5


@featurizer
@dataclass(slots=True)
class GridGraphFeaturizer:
    """Cell nodes with the features of the module docstring. Every field is
    fixed-size, so @featurizer synthesizes spec(): `nodes` [G*G, 5]."""
    mdp: Final[VehicleRoutingMDP]
    nodes: Final[FixedObjectWriter]

    def write_features(self, state: State) -> None:
        g = self.mdp.grid_size
        v = state.vehicles[state.current_vehicle]
        others = float(self.mdp.n_vehicles - 1)
        if others < 1.0:
            others = 1.0
        far = float(2 * (g - 1))
        for row in range(g):
            for col in range(g):
                self.nodes.begin_object()
                count = state.vehicle_count[row, col]
                if row == v.row and col == v.col:
                    self.nodes.append(1.0)
                    count = count - 1
                else:
                    self.nodes.append(0.0)
                self.nodes.append(float(count) / others)
                if state.has_order[row, col]:
                    self.nodes.append(1.0)
                else:
                    self.nodes.append(0.0)
                if row == self.mdp.depot.row and col == self.mdp.depot.col:
                    self.nodes.append(1.0)
                else:
                    self.nodes.append(0.0)
                dr = row - v.row
                if dr < 0:
                    dr = -dr
                dc = col - v.col
                if dc < 0:
                    dc = -dc
                self.nodes.append(float(dr + dc) / far)
