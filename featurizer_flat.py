"""State featurizer: permutation-invariant count maps (WvJ 2026-08-12 rework).

Earlier variants followed the paper's per-vehicle one-hot location blocks
(psi1) and suffered from the induced symmetry: (V-1) * G^2 near-identical
columns that an MLP has to treat as interchangeable. This version collapses
the fleet into the state's `vehicle_count` grid — one integer per cell, ALL
vehicles included (the deciding one too) — which makes vehicle permutations
literally the same feature vector:

  * vehicle-count grid (G^2 entries)     — how many vehicles at each cell,
  * waiting-order grid (G^2 entries)     — the paper's psi2, as the weight
                                           waiting on each cell (0 = none),
  * remaining capacity per vehicle (V)   — round-robin: the DECIDING vehicle's
                                           capacity first, then the others in
                                           index order, wrapping.

Width: 2*G^2 + V (205 for the default 10x10 grid with 5 vehicles).

Because moves apply immediately (no pending buffer), the grids a later decider
sees already include the moves taken earlier in the same period.

`write_features` compiles into the engine (DynaML subset), so it sticks to
for-range loops, index arithmetic and `float()` casts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from dynaplex.modelling import GlobalStateWriter, featurizer

from mdp import State, VehicleRoutingMDP


@featurizer
@dataclass(slots=True)
class VehicleFeaturizer:
    """Count-map features (see module docstring)."""
    mdp: Final[VehicleRoutingMDP]
    v: Final[GlobalStateWriter]

    def write_features(self, state: State) -> None:
        g = self.mdp.grid_size
        n = len(state.vehicles)
        # vehicle-count grid (all vehicles, deciding one included)
        for row in range(g):
            for col in range(g):
                self.v.append(float(state.vehicle_count[row, col]))
        # waiting-order grid (psi2): the weight waiting on each cell
        for row in range(g):
            for col in range(g):
                order = state.grid[row][col]
                if order is None:
                    self.v.append(0.0)
                else:
                    self.v.append(float(order.weight))
        # capacities — deciding vehicle first, then the others, wrapping
        for k in range(n):
            idx = state.current_vehicle + k
            if idx >= n:
                idx = idx - n
            self.v.append(float(state.vehicles[idx].capacity_left))
