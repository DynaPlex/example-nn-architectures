"""Token featurizer for the dot-attention network (WvJ design 2026-08-14).

Three object types, each its own named tensor:

  * `orders`  [max_orders, 2] + `orders_count` [B] — one token per waiting
    order (row-major scan of the order grid), features (row, col) normalized
    to [0, 1]. Padded past the count; the attention mask is derived torch-side
    as `arange(max_orders) < orders_count`.
  * `vehicles` [n_vehicles, 4] — one token per vehicle, features
    (row, col, capacity_left) normalized plus an is-focal flag marking the
    deciding vehicle (state.current_vehicle).
  * `depot` [1, 2] — the warehouse/dropoff location token.

`write_features` compiles into the engine (DynaML subset): for-range loops,
index arithmetic, float() casts.
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


@featurizer
@dataclass(slots=True)
class TokenFeaturizer:
    """Order/vehicle/depot tokens (module docstring)."""
    mdp: Final[VehicleRoutingMDP]
    orders: Final[VariableObjectWriter]
    vehicles: Final[FixedObjectWriter]
    depot: Final[FixedObjectWriter]

    def write_features(self, state: State) -> None:
        g = self.mdp.grid_size
        n = len(state.vehicles)
        gm1 = float(g - 1)
        cap = float(self.mdp.capacity)

        # order tokens: one per waiting order, row-major scan
        for row in range(g):
            for col in range(g):
                if state.has_order[row, col]:
                    self.orders.begin_object()
                    self.orders.append(float(row) / gm1)
                    self.orders.append(float(col) / gm1)

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

        # the warehouse/dropoff token
        self.depot.begin_object()
        self.depot.append(float(self.mdp.depot.row) / gm1)
        self.depot.append(float(self.mdp.depot.col) / gm1)

    # The observation spec: the featurizer's SIZE DECLARATION — one TensorSpec
    # per output tensor, per-sample shapes (no batch axis; the consumer prepends
    # it). The KEYS ARE NOT FREE-FORM: each key must be exactly a writer FIELD
    # NAME above (`orders`, `vehicles`, `depot`), in declaration order — that
    # shared name is what ties the spec entry to its writer field, its Holder
    # slab, its x.<name>.npy file and its forward(batch) key. The one generated
    # exception: a VariableObjectWriter field named `orders` produces TWO
    # tensors, so its `orders_count` companion must be listed right after it
    # (dtype int64, scalar per sample). All of this is validated — a wrong or
    # missing key fails loudly at sizing time.
    #
    # This is also the one method @featurizer could NOT generate here: a
    # variable-count field's max (max_orders) is a modelling fact that no probe
    # state can reveal, so it is declared by hand.
    def spec(self) -> dict:
        return {
            "orders": TensorSpec(Dtype.float32, (self.mdp.max_orders, 2)),
            "orders_count": TensorSpec(Dtype.int64, ()),
            "vehicles": TensorSpec(Dtype.float32, (self.mdp.n_vehicles, 4)),
            "depot": TensorSpec(Dtype.float32, (1, 2)),
        }


# ============================================================================
# What @featurizer generated for TokenFeaturizer (and you never write)
# ============================================================================
#
# How a featurizer sits in the web — the full round trip:
#
#   1. SIZE.     A consumer (DCL, the comparer's NN harness, gym) asks the
#                featurizer for its spec(): tensor names, dtypes, per-sample
#                shapes. The featurizer is the ONLY party that knows sizes.
#   2. ALLOCATE. CPython sizes the slabs from that spec — one zeroed
#                [batch, *sample_shape] numpy array per tensor (batch = slot
#                count, sample count, minibatch...; the featurizer never
#                cares) — and packs them into the generated Holder: a plain
#                dataclass, one Final ndarray field per tensor, same names.
#   3. INSTALL.  The holder is handed back via install(holder): each writer
#                field is constructed AROUND its slab(s), reading its sizes
#                off the array shapes (so the spec's numbers are the writers'
#                numbers by construction). The holder is also the object that
#                crosses into the engine — the worker worlds write into the
#                very same backings CPython allocated.
#   4. WRITE.    Per observed state the driver calls reset(row) — re-pointing
#                every writer at that row — then write_features(state) (the
#                one method you truly author), then finish(), which validates
#                the row and, for a variable field, records the object count
#                into `<field>_count` and zero-fills the padded tail.
#   5. CONSUME.  The filled slabs flow on as a dict keyed by the same names —
#                into SampleSet files (x.orders.npy, x.orders_count.npy, ...),
#                gym observations, and the network factory's forward(batch).
#
# For steps 3 and 4, @featurizer synthesized the following members on this
# class — shown verbatim (inspect.getsource shows the same); hand-write any of
# them only to override:
#
# --- the Holder (attached as TokenFeaturizer.Holder) -------------------------
#
# @dataclass(slots=True)
# class TokenFeaturizerHolder:
#     orders:       Final[Array3D[np.float32]]   # [B, max_orders, 2]
#     orders_count: Final[Array1D[np.int64]]     # [B]   <- generated companion
#     vehicles:     Final[Array3D[np.float32]]   # [B, n_vehicles, 4]
#     depot:        Final[Array3D[np.float32]]   # [B, 1, 2]
#
# One field per TENSOR (not per writer field): the VariableObjectWriter field
# `orders` contributes both its object slab and its `orders_count` companion.
# All-Final ndarray fields make the holder a BUNDLE: a binding-fixed shell
# over shared array backings, usable on both sides of the engine boundary.
#
# --- install: wrap the writers around the slabs ------------------------------
#
# def install(self, h: TokenFeaturizerHolder) -> None:
#     self.orders = VariableObjectWriter(h.orders, h.orders_count,
#                                        h.orders.shape[1], h.orders.shape[2])
#     self.vehicles = FixedObjectWriter(h.vehicles, h.vehicles.shape[1], h.vehicles.shape[2])
#     self.depot = FixedObjectWriter(h.depot, h.depot.shape[1], h.depot.shape[2])
#
# Note the sizes are READ OFF THE ARRAYS (shape[1], shape[2]) — the allocator
# sized those arrays from spec(), so writer sizes and spec cannot disagree.
#
# --- reset / finish: the per-row field walks ---------------------------------
#
# def reset(self, row: int) -> None:
#     self.orders.reset(row)
#     self.vehicles.reset(row)
#     self.depot.reset(row)
#
# def finish(self) -> None:
#     self.orders.finish()     # writes orders_count[row]; zero-fills the tail
#     self.vehicles.finish()   # asserts exactly n_vehicles complete objects
#     self.depot.finish()
#
# --- spec --------------------------------------------------------------------
#
# Would also be synthesized (sizing-by-counting on a probe state) if every
# field were probe-sizable; here the variable `orders` field makes that
# impossible, so spec() above is hand-written — see its comment.
# ============================================================================


@featurizer
@dataclass(slots=True)
class GridTokenFeaturizer:
    """Grid-token variant (WvJ alternative design): every LOCATION is a token.

    Three fixed-size object types — `cells` [G*G, 3] (row, col normalized +
    order presence), `vehicles` [V, 4] (as TokenFeaturizer, is-focal flag
    included), `depot` [1, 2]. No variable counts, no padding: all tokens are
    always real.

    THE load-bearing convention: cell tokens are written in ROW-MAJOR CELL
    ORDER, so token i describes cell i = (i // G, i % G) — exactly action i of
    the CELL action encoding. A pointer head that scores each cell token
    therefore returns logits already in action order, and the ordinary
    action-validity mask (masked CE in training, masked argmax when acting)
    crops them to the deciding vehicle's neighborhood downstream — the network
    never handles masking itself. Pairs with `networks.attention_pointer_net`
    and the default CELL-encoded MDP.

    No spec() is declared: every field is fixed-size, so @featurizer
    synthesizes it (sizing-by-counting on a probe state).
    """
    mdp: Final[VehicleRoutingMDP]
    cells: Final[FixedObjectWriter]
    vehicles: Final[FixedObjectWriter]
    depot: Final[FixedObjectWriter]

    def write_features(self, state: State) -> None:
        g = self.mdp.grid_size
        n = len(state.vehicles)
        gm1 = float(g - 1)
        cap = float(self.mdp.capacity)

        # cell tokens: ROW-MAJOR — token index == cell-action index
        for row in range(g):
            for col in range(g):
                self.cells.begin_object()
                self.cells.append(float(row) / gm1)
                self.cells.append(float(col) / gm1)
                if state.has_order[row, col]:
                    self.cells.append(1.0)
                else:
                    self.cells.append(0.0)

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

        # the warehouse/dropoff token
        self.depot.begin_object()
        self.depot.append(float(self.mdp.depot.row) / gm1)
        self.depot.append(float(self.mdp.depot.col) / gm1)
