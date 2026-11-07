"""The engine-side SAT featurizer: the same tokens as `sat_tokenizer`, written
into dense preallocated slabs instead of derived by reflection.

This is the bridge. `sat_tokenizer` walks a dataclass tree with `get_type_hints`
and dynamic `getattr`, which the DynaPlex interpreter cannot transpile, so it can
never run inside `write_features`. But its reflection is a COMPILE-TIME activity:
once `sat_schema.TOKEN_SCHEMA` is fixed, the token layout is fixed. This module
is that layout expressed statically — one writer field per token type — so it
compiles into the engine and runs inside the worker worlds at DCL scale.

What the static layout buys, beyond speed: every piece of integer metadata the
seq dict carried disappears, because it becomes derivable from position.

    seq dict field      | here
    --------------------|--------------------------------------------------
    type_ids            | which TENSOR the token lives in
    is_group_root       | "lives in `moves`"
    action_group_ids    | arange(num_actions) — moves token i IS action i
    intra_group_pos     | all zero (VRP actions are single-token)
    indices_by_type     | the fixed concat order in networks_sat.forward
    num_action_groups   | num_actions, a constant
    variable counts     | `orders_count`, the ONE genuinely dynamic int

That matters because writer slabs are float32-only (`features.py`), so int
metadata could not have been stored anyway without an ugly cast — and
`nn.Embedding`/`scatter_add_` need Long. The static layout removes the problem
rather than working around it.

Column order and normalisation are transcribed from `sat_schema.TOKEN_SCHEMA`,
which is the naming authority for BOTH sides. `test_featurizers_sat.py` asserts
the two agree exactly, column for column, on real states — that parity test is
the only thing standing between this file and silent column-order drift, which
is invisible to every shape check.

`write_features` stays inside the DynaML subset: while/for-range loops, if/else,
index arithmetic, float() casts, and one helper method (`_nearest`, the same
shape as the proven `MyopicPolicy._nearest`). No `or` in conditions, no tuples,
no comprehensions.
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

from mdp import ActionEncoding, State, VehicleRoutingMDP


@featurizer
@dataclass(slots=True)
class SATFeaturizer:
    """Structured-action tokens for `networks_sat.sat_agile_net`.

    Five token tensors, in `TOKEN_SCHEMA` order. Field names ARE the spec keys
    and the `forward(batch)` keys:

      * `summary`  [1, 4]            — one context token: system load + focal
                                        vehicle (VRPGlobal)
      * `depot`    [1, 2]            — the warehouse token (DepotTok)
      * `vehicles` [n_vehicles, 4]   — one per vehicle, feature 3 = is-focal
      * `orders`   [max_orders, 2]   — one per WAITING order, row-major scan,
                                        + `orders_count` companion
      * `moves`    [num_actions, 6]  — one per ACTION, in action-index order

    THE load-bearing convention, inherited from `GridTokenFeaturizer`: `moves`
    token i describes action i. The head scores each moves token, so the [B,
    num_actions] output is already in action order and the ordinary validity
    mask crops it downstream (masked CE in training, masked argmax when acting).
    The network never masks itself, and must never emit -inf or NaN — a NaN would
    poison the masked argmax.

    Works under either encoding: DIRECTIONAL writes 5 move tokens,
    CELL writes grid_size**2 in row-major order. DIRECTIONAL is strongly
    preferred — under CELL the `moves` slab is 20x larger for actions that are
    inadmissible 95% of the time, and the raw `action_index` column (0..99)
    sits badly beside columns normalised to [0, 1].
    """
    mdp: Final[VehicleRoutingMDP]
    summary: Final[FixedObjectWriter]
    depot: Final[FixedObjectWriter]
    vehicles: Final[FixedObjectWriter]
    orders: Final[VariableObjectWriter]
    moves: Final[FixedObjectWriter]

    def _nearest(self, state: State, row: int, col: int) -> int:
        """Manhattan distance to the nearest waiting order, or a big sentinel.

        Same body as `MyopicPolicy._nearest` (mdp.py:343-361) — which is what
        proves this transpiles — and the same body as
        `sat_schema.nearest_order_distance`, which is what the parity test
        compares against.
        """
        g = self.mdp.grid_size
        best = 1000000000
        r = 0
        while r < g:
            c = 0
            while c < g:
                if state.has_order[r, c]:
                    dr = r - row
                    if dr < 0:
                        dr = -dr
                    dc = c - col
                    if dc < 0:
                        dc = -dc
                    if dr + dc < best:
                        best = dr + dc
                c = c + 1
            r = r + 1
        return best

    def write_features(self, state: State) -> None:
        g = self.mdp.grid_size
        n = len(state.vehicles)
        gm1 = float(g - 1)
        cap = float(self.mdp.capacity)
        # normaliser for nearest-order distance, and the value used when nothing
        # is waiting: nothing is maximally far, so the column stays in [0, 1]
        # with no sentinel leaking through
        dmax = float(2 * (g - 1))
        focal = state.vehicles[state.current_vehicle]

        # --- summary token: system load + the focal vehicle ------------------
        self.summary.begin_object()
        self.summary.append(float(state.n_orders) / float(self.mdp.max_orders))
        self.summary.append(float(focal.row) / gm1)
        self.summary.append(float(focal.col) / gm1)
        self.summary.append(float(focal.capacity_left) / cap)

        # --- the warehouse/dropoff token -------------------------------------
        self.depot.begin_object()
        self.depot.append(float(self.mdp.depot.row) / gm1)
        self.depot.append(float(self.mdp.depot.col) / gm1)

        # --- vehicle tokens: position + capacity + is-focal flag -------------
        i = 0
        while i < n:
            veh = state.vehicles[i]
            self.vehicles.begin_object()
            self.vehicles.append(float(veh.row) / gm1)
            self.vehicles.append(float(veh.col) / gm1)
            self.vehicles.append(float(veh.capacity_left) / cap)
            if i == state.current_vehicle:
                self.vehicles.append(1.0)
            else:
                self.vehicles.append(0.0)
            i = i + 1

        # --- order tokens: one per waiting order, ROW-MAJOR scan -------------
        for row in range(g):
            for col in range(g):
                if state.has_order[row, col]:
                    self.orders.begin_object()
                    self.orders.append(float(row) / gm1)
                    self.orders.append(float(col) / gm1)

        # --- action tokens: COMPLETE and in action-index order ---------------
        if self.mdp.action_encoding == ActionEncoding.CELL:
            # cell a IS action a = (a // g, a % g); always on-grid
            a = 0
            while a < g * g:
                self._write_move(state, a, a // g, a % g, 1.0, gm1, dmax)
                a = a + 1
        else:
            # DIRECTIONAL: 0=stay, 1=up, 2=down, 3=left, 4=right — the same
            # decode as modify_state_with_action (mdp.py:206-215)
            k = 0
            while k < 5:
                row = focal.row
                col = focal.col
                if k == 1:
                    row = row - 1
                elif k == 2:
                    row = row + 1
                elif k == 3:
                    col = col - 1
                elif k == 4:
                    col = col + 1
                # An off-grid target is CLAMPED back to the vehicle's own cell
                # and flagged, never encoded as an out-of-range coordinate: the
                # net must emit finite, in-range features for actions the mask
                # will reject anyway. Written as four separate ifs rather than
                # one `or` chain to stay inside the compilable subset.
                on_grid = 1.0
                if row < 0:
                    on_grid = 0.0
                if row >= g:
                    on_grid = 0.0
                if col < 0:
                    on_grid = 0.0
                if col >= g:
                    on_grid = 0.0
                if on_grid == 0.0:
                    row = focal.row
                    col = focal.col
                self._write_move(state, k, row, col, on_grid, gm1, dmax)
                k = k + 1

    def _write_move(self, state: State, action_index: int, row: int, col: int,
                    on_grid: float, gm1: float, dmax: float) -> None:
        """One action token: the key, the target cell, and its desirability.

        `nearest_order_dist` is deliberately the myopic benchmark's own decision
        statistic, so the net starts at least as informed as the 3.096 policy.
        """
        self.moves.begin_object()
        self.moves.append(float(action_index))
        self.moves.append(float(row) / gm1)
        self.moves.append(float(col) / gm1)
        self.moves.append(on_grid)
        if state.has_order[row, col]:
            self.moves.append(1.0)
        else:
            self.moves.append(0.0)
        if state.n_orders == 0:
            self.moves.append(1.0)
        else:
            self.moves.append(float(self._nearest(state, row, col)) / dmax)

    def spec(self) -> dict:
        """The size declaration. Hand-written because `orders` is variable-count
        and no probe state can reveal its maximum (`VariableObjectCounter.spec`
        refuses without one) — `max_orders` is a modelling fact.

        Keys must equal the writer field names in declaration order, with each
        variable field immediately followed by its `_count` companion. Widths
        come from `sat_schema.TOKEN_SCHEMA`; `test_featurizers_sat` asserts they
        still agree.
        """
        return {
            "summary": TensorSpec(Dtype.float32, (1, 4)),
            "depot": TensorSpec(Dtype.float32, (1, 2)),
            "vehicles": TensorSpec(Dtype.float32, (self.mdp.n_vehicles, 4)),
            "orders": TensorSpec(Dtype.float32, (self.mdp.max_orders, 2)),
            "orders_count": TensorSpec(Dtype.int64, ()),
            "moves": TensorSpec(Dtype.float32, (self.mdp.num_actions, 6)),
        }
