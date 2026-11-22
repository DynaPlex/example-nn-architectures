"""Candidate-token featurizer: one token per admissible next location.

The idea: a vehicle's decision is a choice among at most five cells (stay, up,
down, left, right), so give the network exactly those five cells as tokens and
let it score each one against the context. Compared with one token per grid
cell (`featurizer_cells.GridTokenFeaturizer`) this is twenty times fewer
tokens on the default grid, and nothing about it needs a grid: any routing
model with a small set of candidate moves can be featurized the same way.

Tensors, per sample:

  * `candidates` [5, 4] — in the fixed order stay/up/down/left/right, each
    token (row, col) of the target cell normalized to [0, 1], the weight of the
    order waiting there divided by the vehicle capacity (0 = none), and an
    in-grid flag. An off-grid move is still written (the
    tensor is fixed-size) with the vehicle's own coordinates and the flag at 0;
    the network does not score it and the validity mask would reject it anyway.
  * `orders` [max_orders, 3] + `orders_count` — the waiting orders, (row, col)
    normalized and weight / vehicle capacity, padded past the count. The model
    never holds more than `max_orders` orders, so nothing is left out.
  * `vehicles` [n_vehicles, 4] — (row, col, capacity_left) normalized plus an
    is-focal flag marking the deciding vehicle.
  * `depot` [1, 2] — the depot location.

Pairs with `networks.move_token_factory` and the CELL action encoding: the network
scores the five candidate tokens and scatters the scores into the grid-cell
action vector using the candidates' own coordinates.

`write_features` compiles into the engine (DynaML subset).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from dynaplex.modelling import (
    Dtype,
    FixedObjectWriter,
    TensorSpec,
    VariableObjectWriter,
    featurizer,
)

from mdp import State, VehicleRoutingMDP

N_CANDIDATES: Final[int] = 5


@featurizer
@dataclass(slots=True)
class MoveTokenFeaturizer:
    """Candidate tokens plus order/vehicle/depot context (module docstring)."""
    mdp: Final[VehicleRoutingMDP]
    candidates: Final[FixedObjectWriter]
    orders: Final[VariableObjectWriter]
    vehicles: Final[FixedObjectWriter]
    depot: Final[FixedObjectWriter]

    def _candidate(self, state: State, row: int, col: int, in_grid: bool) -> None:
        gm1 = float(self.mdp.grid_size - 1)
        self.candidates.begin_object()
        self.candidates.append(float(row) / gm1)
        self.candidates.append(float(col) / gm1)
        order = state.grid[row][col]
        if order is None:
            self.candidates.append(0.0)
        else:
            self.candidates.append(float(order.weight) / float(self.mdp.capacity))
        if in_grid:
            self.candidates.append(1.0)
        else:
            self.candidates.append(0.0)

    def write_features(self, state: State) -> None:
        g = self.mdp.grid_size
        n = len(state.vehicles)
        gm1 = float(g - 1)
        cap = float(self.mdp.capacity)
        v = state.vehicles[state.current_vehicle]

        # candidate tokens: stay, up, down, left, right (off-grid: own cell, flag 0)
        self._candidate(state, v.row, v.col, True)
        if v.row > 0:
            self._candidate(state, v.row - 1, v.col, True)
        else:
            self._candidate(state, v.row, v.col, False)
        if v.row < g - 1:
            self._candidate(state, v.row + 1, v.col, True)
        else:
            self._candidate(state, v.row, v.col, False)
        if v.col > 0:
            self._candidate(state, v.row, v.col - 1, True)
        else:
            self._candidate(state, v.row, v.col, False)
        if v.col < g - 1:
            self._candidate(state, v.row, v.col + 1, True)
        else:
            self._candidate(state, v.row, v.col, False)

        # order tokens: one per waiting order, in the order of the state's list
        for order in state.orders:
            self.orders.begin_object()
            self.orders.append(float(order.row) / gm1)
            self.orders.append(float(order.col) / gm1)
            self.orders.append(float(order.weight) / cap)

        # vehicle tokens: position + remaining capacity + is-focal flag
        for i in range(n):
            veh = state.vehicles[i]
            self.vehicles.begin_object()
            self.vehicles.append(float(veh.row) / gm1)
            self.vehicles.append(float(veh.col) / gm1)
            self.vehicles.append(float(veh.capacity_left) / cap)
            if i == state.current_vehicle:
                self.vehicles.append(1.0)
            else:
                self.vehicles.append(0.0)

        # the depot token
        self.depot.begin_object()
        self.depot.append(float(self.mdp.depot.row) / gm1)
        self.depot.append(float(self.mdp.depot.col) / gm1)

    # Hand-written because `orders` is variable-count: its maximum (max_orders)
    # is a modelling fact no probe state can reveal. Keys = writer field names,
    # in declaration order; `orders_count` right after `orders`.
    def spec(self) -> dict:
        return {
            "candidates": TensorSpec(Dtype.float32, (N_CANDIDATES, 4)),
            "orders": TensorSpec(Dtype.float32, (self.mdp.max_orders, 3)),
            "orders_count": TensorSpec(Dtype.int64, ()),
            "vehicles": TensorSpec(Dtype.float32, (self.mdp.n_vehicles, 4)),
            "depot": TensorSpec(Dtype.float32, (1, 2)),
        }
