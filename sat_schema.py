"""The VRP observation schema: the ONE source of truth for SAT token layout.

`sat_tokenizer` derives token layout by reflecting over dataclass fields, which
is a CPython-only capability (see that module's docstring). The engine-side
featurizer must reproduce the same tokens by a completely different mechanism —
`write_features` appending floats into preallocated float32 slabs, transpiled to
C++. Two independent implementations of one layout is the setup where silent
COLUMN-ORDER DRIFT lives: swap two `append` lines and every shape still
validates, every test still passes, and the network quietly trains on garbage.

So this module is the naming and ordering authority for both. `TOKEN_SCHEMA`
declares, per token type: its writer kind, its per-sample maximum token count as
an expression over the MDP, and its column names IN ORDER. Then:

  * Phase 1 (CPython): the dataclasses below carry exactly those columns as
    fields in exactly that order; `sat_tokenizer` reads them by reflection.
  * Phase 2 (engine): `featurizers_sat.write_features` appends exactly those
    columns in exactly that order, and `spec()` is read off `max_count`.

`validate_schema()` asserts the dataclass field order still equals the declared
column order, so the drift is caught at import rather than in a loss curve.

What this deliberately does NOT do: share one expression language between the
two sides. A column's *value* is computed from a Python object on one side and
by DynaML index arithmetic on the other; pretending one evaluator serves both
would cost more than the drift it prevents. The table pins names, order, widths
and counts — which is where drift actually happens — and the two bodies are
written from it, guarded by the parity test.

Encoding note
-------------
DIRECTIONAL is the recommended encoding and the default here. Two reasons:

  * cost — CELL writes G*G = 100 action tokens where at most 5 are ever
    admissible, so both the slab (2,400 B/row against 120) and the O(N^2)
    attention over action nodes pay 20x for nothing;
  * conditioning — `action_index` is an `ActionKey` field, so it is a real
    MDP action id AND (unavoidably) a feature column. Under DIRECTIONAL it is
    a small ordinal 0..4; under CELL it is a raw 0..99 sitting next to columns
    normalised to [0, 1].

CELL is still supported, because the sample sets and the 3.324/pointer-net
baselines are cell-encoded.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

from mdp import ActionEncoding, State, VehicleRoutingMDP
from sat_markers import Action, ActionKey, GlobalState

# ============================================================================
# TOKEN DATACLASSES — field order IS column order (enforced by validate_schema)
# ============================================================================


@dataclass(slots=True)
class VRPGlobal(GlobalState):
    """One context token summarising the decision: system load + the focal vehicle.

    Redundant with the `is_focal` flag on VehicleTok by design — the context
    encoder mean-pools its tokens, so a single summary token gives the pooled
    state vector a direct path to the quantities that drive the decision instead
    of requiring it to reconstruct them from the pool.
    """
    n_orders_frac: float
    focal_row: float
    focal_col: float
    focal_cap: float


@dataclass(slots=True)
class DepotTok:
    """The warehouse/dropoff token (mdp.depot, constant at (0, 0))."""
    row: float
    col: float


@dataclass(slots=True)
class VehicleTok:
    """One token per vehicle. `is_focal` marks state.current_vehicle.

    Same four columns as TokenFeaturizer's `vehicles` tensor
    (featurizers_tokens.py:57-67), in the same order, so the two are directly
    comparable in the parity test.
    """
    row: float
    col: float
    cap_left: float
    is_focal: float


@dataclass(slots=True)
class OrderTok:
    """One token per WAITING order, row-major scan of state.has_order.

    Variable count: this is the only token type whose population changes with
    the state, hence the `_count` companion on the engine side.
    """
    row: float
    col: float


@dataclass(slots=True)
class MoveAction(Action):
    """One token per ACTION, written completely and in action-index order.

    `action_index` is the load-bearing contract of the whole integration: it is
    annotated `ActionKey[int]`, so `sat_actions.resolve_action` returns it as the
    key for `mdp.modify_state_with_action`, and because the action list is built
    complete and in order, group g == action g. Every downstream consumer (DCL
    labels, the validity mask, masked argmax) assumes logit position == action
    index, so this must never become a compacted valid-only list.

    Columns after the key describe the TARGET cell of the move. Off-grid targets
    are clamped to the vehicle's own cell and flagged by `on_grid=0.0` rather
    than encoded as out-of-range coordinates: the network must emit finite logits
    for inadmissible actions (a NaN would poison the masked argmax), and the
    validity mask crops them downstream.

    `nearest_order_dist` is deliberately the myopic benchmark's decision
    statistic — it makes the net at least as informed as the 3.096 policy, and
    `MyopicPolicy._nearest` (mdp.py:343-361) already proves the computation
    transpiles, so this column survives into Phase 2 unchanged.
    """
    action_index: ActionKey[int]
    row: float
    col: float
    on_grid: float
    has_order: float
    nearest_order_dist: float


@dataclass(slots=True)
class SATObservation:
    """The observation root. Skipped as a token (`skip_root=True`) — it has no
    primitive fields of its own, only children.

    Every field is populated on EVERY state, including `orders` when no order is
    waiting. That is not cosmetic: `sat_tokenizer` registers a token type the
    first time it sees an instance, and the registry is closed thereafter, so a
    type that is empty on the first state and populated later would raise. An
    empty list is a zero count, never an absent type.
    """
    state: VRPGlobal
    depot: DepotTok
    vehicles: list[VehicleTok]
    orders: list[OrderTok]
    actions: list[MoveAction]


# ============================================================================
# THE SCHEMA TABLE
# ============================================================================


@dataclass(frozen=True, slots=True)
class TokenSpec:
    """One token type's declaration, shared by both phases.

    Attributes:
        name: the engine-side writer FIELD NAME and spec key. Spec keys are not
            free-form — they must equal the writer field names in declaration
            order, each variable field followed by its `<name>_count` companion.
        dataclass: the CPython token type whose fields carry `columns`.
        kind: "fixed" -> FixedObjectWriter [n, width];
            "variable" -> VariableObjectWriter [max, width] + int64 count.
        max_count: an expression over `mdp` giving the per-sample token
            capacity. A string, not a number, because the generated/hand-written
            `spec()` must stay independent of any particular MDP instance — and
            because a variable field's maximum is a MODELLING fact that no probe
            state can reveal (VariableObjectCounter.spec refuses without it).
        columns: column names in order. Must equal the dataclass field order.
        role: "context" tokens feed the state encoder; "action" tokens become
            the action nodes and must be in action-index order.
    """
    name: str
    dataclass: type
    kind: Literal["fixed", "variable"]
    max_count: str
    columns: tuple[str, ...]
    role: Literal["context", "action"]

    @property
    def width(self) -> int:
        return len(self.columns)


TOKEN_SCHEMA: tuple[TokenSpec, ...] = (
    # `name` is the Phase 2 writer FIELD NAME, so it must be a clean identifier
    # that does not shadow a builtin — hence "summary" rather than "globals".
    TokenSpec(
        name="summary", dataclass=VRPGlobal, kind="fixed", max_count="1",
        columns=("n_orders_frac", "focal_row", "focal_col", "focal_cap"),
        role="context",
    ),
    TokenSpec(
        name="depot", dataclass=DepotTok, kind="fixed", max_count="1",
        columns=("row", "col"),
        role="context",
    ),
    TokenSpec(
        name="vehicles", dataclass=VehicleTok, kind="fixed",
        max_count="mdp.n_vehicles",
        columns=("row", "col", "cap_left", "is_focal"),
        role="context",
    ),
    TokenSpec(
        name="orders", dataclass=OrderTok, kind="variable",
        max_count="mdp.max_orders",
        columns=("row", "col"),
        role="context",
    ),
    TokenSpec(
        name="moves", dataclass=MoveAction, kind="fixed",
        max_count="mdp.num_actions",
        columns=("action_index", "row", "col", "on_grid", "has_order",
                 "nearest_order_dist"),
        role="action",
    ),
)


def validate_schema() -> None:
    """Assert the table and the dataclasses still agree. Called at import.

    This is the guard against column-order drift: renaming or reordering a
    dataclass field without updating the table (or vice versa) fails here,
    loudly, instead of silently permuting feature columns.
    """
    seen: set[str] = set()
    n_action = 0
    for ts in TOKEN_SCHEMA:
        assert ts.name not in seen, f"duplicate token name {ts.name!r}"
        seen.add(ts.name)
        actual = tuple(f.name for f in fields(ts.dataclass))
        assert actual == ts.columns, (
            f"{ts.dataclass.__name__} field order {actual} != schema columns "
            f"{ts.columns} for token {ts.name!r} — column-order drift")
        if ts.role == "action":
            n_action += 1
            assert issubclass(ts.dataclass, Action), (
                f"{ts.dataclass.__name__} is the action token type but does not "
                f"subclass Action, so the tokenizer would not open a group")
            assert ts.columns[0] == "action_index", (
                "the action key must be the first column so it is unmissable")
        else:
            assert not issubclass(ts.dataclass, Action), (
                f"{ts.dataclass.__name__} is a context token but subclasses "
                f"Action — every instance would open its own action group")
    assert n_action == 1, f"expected exactly one action token type, got {n_action}"


validate_schema()


def resolve_max_counts(mdp: VehicleRoutingMDP) -> dict[str, int]:
    """Evaluate each token type's `max_count` expression against `mdp`.

    Phase 2's `spec()` is derived from this. Kept as expression strings in the
    table (rather than numbers here) so the table stays MDP-instance-independent.
    """
    return {ts.name: int(eval(ts.max_count, {"mdp": mdp})) for ts in TOKEN_SCHEMA}


# ============================================================================
# CPYTHON OBSERVATION BUILDER (Phase 1; the parity reference for Phase 2)
# ============================================================================

# DIRECTIONAL move deltas, in action-index order: 0=stay, 1=up, 2=down,
# 3=left, 4=right. Mirrors mdp.modify_state_with_action:206-215 and the
# candidate order MyopicPolicy uses.
_MOVES: tuple[tuple[int, int], ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def nearest_order_distance(state: State, grid_size: int, row: int, col: int) -> int:
    """Manhattan distance from (row, col) to the nearest waiting order.

    Transcribed from MyopicPolicy._nearest (mdp.py:343-361) — while loops and
    index arithmetic only, so the same body transpiles for Phase 2. Returns a
    large sentinel when no order is waiting; callers normalise it away.
    """
    best = 1000000000
    r = 0
    while r < grid_size:
        c = 0
        while c < grid_size:
            if state.has_order[r, c]:
                dr = r - row
                if dr < 0:
                    dr = -dr
                dc = c - col
                if dc < 0:
                    dc = -dc
                if dr + dc < best:
                    best = dr + dc
            c += 1
        r += 1
    return best


def build_sat_observation(state: State, mdp: VehicleRoutingMDP) -> SATObservation:
    """Tokenise one decision state into the observation tree.

    Normalisation matches featurizers_tokens.py exactly (positions by
    `grid_size - 1`, capacity by `capacity`) so the Phase 2 parity test can
    compare column-for-column.

    Called only at AWAIT_ACTION: `state.current_vehicle` is the decision cursor,
    and the action set is the focal vehicle's admissible moves.
    """
    g = mdp.grid_size
    gm1 = float(g - 1)
    cap = float(mdp.capacity)
    # max Manhattan distance on the grid — the normaliser for nearest-order
    # distance, and the value used when nothing is waiting (nothing is maximally
    # far, which keeps the column in [0, 1] with no sentinel leaking through)
    dmax = float(2 * (g - 1))

    focal = state.vehicles[state.current_vehicle]

    globals_tok = VRPGlobal(
        n_orders_frac=float(state.n_orders) / float(mdp.max_orders),
        focal_row=float(focal.row) / gm1,
        focal_col=float(focal.col) / gm1,
        focal_cap=float(focal.capacity_left) / cap,
    )

    depot_tok = DepotTok(row=float(mdp.depot.row) / gm1,
                         col=float(mdp.depot.col) / gm1)

    vehicle_toks: list[VehicleTok] = []
    for i in range(mdp.n_vehicles):
        veh = state.vehicles[i]
        vehicle_toks.append(VehicleTok(
            row=float(veh.row) / gm1,
            col=float(veh.col) / gm1,
            cap_left=float(veh.capacity_left) / cap,
            is_focal=1.0 if i == state.current_vehicle else 0.0,
        ))

    # order tokens: row-major scan, matching TokenFeaturizer.write_features
    order_toks: list[OrderTok] = []
    for row in range(g):
        for col in range(g):
            if state.has_order[row, col]:
                order_toks.append(OrderTok(row=float(row) / gm1,
                                           col=float(col) / gm1))

    actions = _build_actions(state, mdp, g, gm1, dmax)

    return SATObservation(state=globals_tok, depot=depot_tok,
                          vehicles=vehicle_toks, orders=order_toks,
                          actions=actions)


def _build_actions(state: State, mdp: VehicleRoutingMDP, g: int, gm1: float,
                   dmax: float) -> list[MoveAction]:
    """The action tokens: COMPLETE and in action-index order (group g == action g).

    DIRECTIONAL: 5 tokens; an off-grid move keeps the vehicle's own cell as its
    target and sets `on_grid=0.0`.
    CELL: g*g tokens in row-major order — cell i is action i, the same
    convention GridTokenFeaturizer relies on (featurizers_tokens.py:208).
    """
    focal = state.vehicles[state.current_vehicle]
    actions: list[MoveAction] = []

    if mdp.action_encoding == ActionEncoding.DIRECTIONAL:
        for k, (dr, dc) in enumerate(_MOVES):
            row = focal.row + dr
            col = focal.col + dc
            on_grid = 0 <= row < g and 0 <= col < g
            if not on_grid:
                # clamp to the vehicle's own cell: finite, in-range features for
                # an action the mask will reject anyway
                row, col = focal.row, focal.col
            actions.append(_move_action(state, k, row, col, gm1, dmax,
                                        1.0 if on_grid else 0.0, g))
    else:
        for a in range(g * g):
            actions.append(_move_action(state, a, a // g, a % g, gm1, dmax,
                                        1.0, g))
    return actions


def _move_action(state: State, action_index: int, row: int, col: int,
                 gm1: float, dmax: float, on_grid: float, g: int) -> MoveAction:
    d = nearest_order_distance(state, g, row, col)
    return MoveAction(
        action_index=action_index,
        row=float(row) / gm1,
        col=float(col) / gm1,
        on_grid=on_grid,
        has_order=1.0 if state.has_order[row, col] else 0.0,
        nearest_order_dist=1.0 if state.n_orders == 0 else float(d) / dmax,
    )
