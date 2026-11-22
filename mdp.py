"""Stylized dynamic vehicle routing on a grid — the "1-step decision" variant.

The model follows Akkerman, Mes & van Jaarsveld (2025), "A comparison of
reinforcement learning policies for dynamic vehicle routing problems with
stochastic customer requests", Computers & Industrial Engineering 200, 110747
(`akkermans_model.pdf` in this repository; Section 3, Fig. 2a), with two
deliberate departures:

  * a cell can hold MORE THAN ONE unit of demand. An ORDER is a cell with demand
    waiting on it, and its WEIGHT is the number of units. At most `max_orders`
    cells hold an order at the same time; demand itself is never refused. While
    fewer than `max_orders` cells are occupied, a unit arrives on a cell drawn
    uniformly from the whole grid (it starts an order there, or adds to the one
    that is there). At the cap it is added to one of the existing orders, drawn
    uniformly. So `max_orders` bounds what a policy has to look at, not the
    demand;
  * a vehicle whose capacity is exhausted takes no decisions: it is skipped in
    the decision cycle and walks a fixed shortest path to the depot at (0, 0) —
    first up, then left — where its capacity resets.

Dynamics per period:
  1. Every vehicle with remaining capacity chooses, in index order, its NEXT
     LOCATION (vehicles may share a cell) and MOVES IMMEDIATELY — later
     deciders in the same period observe the updated positions.
  2. Event: full vehicles take their automatic return step;
     a vehicle standing on an order with capacity left takes as much of its
     weight as it can carry (lowest index first); vehicles at the depot reset
     to full capacity; with probability `order_probability` one unit of demand
     arrives (see above) — arriving under a vehicle with capacity means it is
     served on the spot. Every unit still waiting then costs `holding_cost`
     this period.

Infinite horizon; the natural objective is average cost per period.

Actions — TWO encodings of the same decision (`ActionEncoding`, a constructor
choice; the dynamics are identical, only the action's NAME differs):

  * `CELL` (default): as in the paper's eq. (3a), an action is a GRID CELL —
    the vehicle's next location. Action `a` means "move to
    (a // grid_size, a % grid_size)", so there are grid_size**2 actions, of
    which validity masking admits only the current cell (stay) and its in-grid
    neighbors — at most 5. Encoding decisions as locations keeps the action
    space spatially aligned with the state (a policy network can score every
    cell's desirability and let the mask crop the scores to the deciding
    vehicle's neighborhood).
  * `DIRECTIONAL`: an action is a MOVE of the deciding vehicle — the 5 actions
    0..4 = stay/up/down/left/right, with off-grid moves masked out. This is
    the encoding the paper's NNPFA actually ran (iterative per-vehicle moves),
    and the natural head for a network that reads the deciding vehicle (it
    emits 5 logits directly — no scattering into cell space).

Because a directional action reduces to a target cell, both encodings share
every line of transition logic; the dispatch is a thin decode at the top of
`modify_state_with_action` (and the mirrored encode in `write_action_validity`
and `MyopicPolicy`). Same seeds => bit-identical trajectories and costs across
encodings (regression-tested). The encoding is part of the MDP's structural
identity: DCL/comparer runs land in separate experiment workdirs, and
`num_actions` is grid_size**2 or 5 accordingly.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dynaplex.modelling import (
    Array2D,
    HorizonType,
    StateCategory,
    TrajectoryContext,
    Validity,
    const_dataclass,
)


class ActionEncoding(Enum):
    """How an action names the deciding vehicle's move (module docstring):
    CELL = target grid cell (grid_size**2 actions, paper eq. 3a);
    DIRECTIONAL = stay/up/down/left/right (5 actions, the NNPFA encoding)."""
    CELL = 1
    DIRECTIONAL = 2


# DIRECTIONAL action indices, also the candidate order used by MyopicPolicy:
# 0 = stay, 1 = up (row-1), 2 = down (row+1), 3 = left (col-1), 4 = right (col+1)


@dataclass(slots=True)
class Vehicle:
    row: int
    col: int
    capacity_left: int


@dataclass(slots=True)
class Order:
    """The demand waiting on one cell: `weight` units at (row, col). `slot` is
    the order's index in `State.orders`, so it can be removed without a search."""
    row: int
    col: int
    weight: int
    slot: int


@dataclass(slots=True)
class State:
    """The waiting demand is held twice, and both views name the SAME Order
    objects: `orders` lists them (at most `max_orders`), `grid[r][c]` is the
    order on that cell or None. `total_weight` is the sum of their weights, so
    the cost is O(1). vehicle_count[r, c] mirrors the fleet positions (every
    move is a -1/+1 pair) — the permutation-invariant feature map.
    current_vehicle is the decision cursor; only vehicles with
    capacity_left > 0 appear in the decision cycle."""
    vehicles: list[Vehicle]
    orders: list[Order]
    grid: list[list[Order | None]]
    total_weight: int
    vehicle_count: Array2D[np.int64]
    current_vehicle: int
    category: StateCategory


@const_dataclass(slots=True)
class Depot:
    row: int
    col: int


@const_dataclass(init=False, slots=True)
class VehicleRoutingMDP:
    """See the module docstring. Infinite horizon, average cost per period."""

    grid_size: int
    n_vehicles: int
    capacity: int
    max_orders: int
    order_probability: float
    holding_cost: float
    action_encoding: ActionEncoding
    depot: Depot
    num_actions: int
    horizon_type: HorizonType

    def __init__(self, grid_size: int = 10, n_vehicles: int = 5, capacity: int = 5,
                 max_orders: int = 20, order_probability: float = 0.5,
                 holding_cost: float = 1.0,
                 action_encoding: ActionEncoding = ActionEncoding.CELL):
        assert 1 <= max_orders <= grid_size * grid_size
        self.grid_size = grid_size
        self.n_vehicles = n_vehicles
        self.capacity = capacity
        self.max_orders = max_orders
        self.order_probability = order_probability
        self.holding_cost = holding_cost
        self.action_encoding = action_encoding
        self.depot = Depot(0, 0)
        if action_encoding == ActionEncoding.CELL:
            self.num_actions = grid_size * grid_size
        else:
            self.num_actions = 5
        self.horizon_type = HorizonType.INFINITE

    # ---- helpers ----------------------------------------------------------

    def _advance_cursor(self, state: State, start: int) -> None:
        i = start
        while i < self.n_vehicles:
            if state.vehicles[i].capacity_left > 0:
                state.current_vehicle = i
                state.category = StateCategory.AWAIT_ACTION
                return
            i += 1
        state.current_vehicle = 0
        state.category = StateCategory.AWAIT_EVENT

    def _arrival_cell(self, state: State, u: float) -> int:
        """The cell (row * grid_size + col) a unit of demand arrives on, from one
        uniform draw `u` in [0, 1): below the cap any cell of the grid, at the
        cap the cell of one of the existing orders. One draw in both cases, so
        two rollouts that differ in whether the cap is reached keep using the
        same random numbers afterwards."""
        g = self.grid_size
        if len(state.orders) < self.max_orders:
            cell = int(u * float(g * g))
            if cell >= g * g:
                cell = g * g - 1
            return cell
        k = int(u * float(len(state.orders)))
        if k >= len(state.orders):
            k = len(state.orders) - 1
        order = state.orders[k]
        return order.row * g + order.col

    def _add_unit(self, state: State, row: int, col: int) -> None:
        order = state.grid[row][col]
        if order is None:
            new_order = Order(row=row, col=col, weight=1, slot=len(state.orders))
            state.orders.append(new_order)
            state.grid[row][col] = new_order
        else:
            order.weight += 1
        state.total_weight += 1

    def _remove_order(self, state: State, order: Order) -> None:
        # the last order of the list takes the place of the one that leaves
        last = state.orders[-1]
        state.orders[order.slot] = last
        last.slot = order.slot
        state.orders.pop()
        state.grid[order.row][order.col] = None

    # ---- MDP contract -----------------------------------------------------

    def get_initial_state(self, context: TrajectoryContext) -> State:
        g = self.grid_size
        vehicles: list[Vehicle] = []
        vehicle_count = np.zeros((g, g), dtype=np.int64)
        i = 0
        while i < self.n_vehicles:
            v = Vehicle(row=context.rng.choice(g),
                        col=context.rng.choice(g),
                        capacity_left=self.capacity)
            vehicle_count[v.row, v.col] += 1
            vehicles.append(v)
            i += 1
        grid: list[list[Order | None]] = []
        for _r in range(g):
            line: list[Order | None] = []
            for _c in range(g):
                line.append(None)
            grid.append(line)
        orders: list[Order] = []
        state = State(
            vehicles=vehicles,
            orders=orders,
            grid=grid,
            total_weight=0,
            vehicle_count=vehicle_count,
            current_vehicle=0,
            category=StateCategory.AWAIT_ACTION,
        )
        # 30 initial units of demand, placed by the arrival rule: on distinct
        # cells until the cap is reached, on the existing orders after that
        for _k in range(30):
            cell = self._arrival_cell(state, context.rng.random())
            self._add_unit(state, cell // g, cell % g)
        self._advance_cursor(state, 0)
        return state

    def modify_state_with_action(self, state: State, context: TrajectoryContext,
                                 action: int) -> None:
        # the move happens IMMEDIATELY (no pending buffer): later deciders in the
        # same period see the updated position and count grid.
        # The ONLY encoding-dependent step is decoding the action to the target
        # cell — everything after is shared between the encodings.
        v = state.vehicles[state.current_vehicle]
        g = self.grid_size
        row = v.row
        col = v.col
        if self.action_encoding == ActionEncoding.CELL:
            row = action // g
            col = action % g
        else:
            # DIRECTIONAL: 0=stay, 1=up, 2=down, 3=left, 4=right
            if action == 1:
                row -= 1
            elif action == 2:
                row += 1
            elif action == 3:
                col -= 1
            elif action == 4:
                col += 1
        state.vehicle_count[v.row, v.col] -= 1
        v.row = row
        v.col = col
        state.vehicle_count[v.row, v.col] += 1
        self._advance_cursor(state, state.current_vehicle + 1)

    def modify_state_with_event(self, state: State, context: TrajectoryContext) -> None:
        # 1. deciding vehicles already moved at decision time; full vehicles take
        #    their automatic depot-return step (up, then left) now
        i = 0
        while i < self.n_vehicles:
            v = state.vehicles[i]
            if v.capacity_left == 0:
                state.vehicle_count[v.row, v.col] -= 1
                if v.row > 0:
                    v.row -= 1
                elif v.col > 0:
                    v.col -= 1
                state.vehicle_count[v.row, v.col] += 1
            i += 1
        # 2. serving: lowest index first; a vehicle takes all it can carry
        i = 0
        while i < self.n_vehicles:
            v = state.vehicles[i]
            if v.capacity_left > 0:
                order = state.grid[v.row][v.col]
                if order is not None:
                    take = order.weight
                    if take > v.capacity_left:
                        take = v.capacity_left
                    order.weight -= take
                    v.capacity_left -= take
                    state.total_weight -= take
                    if order.weight == 0:
                        self._remove_order(state, order)
            i += 1
        # 3. depot reset
        i = 0
        while i < self.n_vehicles:
            v = state.vehicles[i]
            if v.row == self.depot.row and v.col == self.depot.col:
                v.capacity_left = self.capacity
            i += 1
        # 4. arrival of one unit of demand (served on the spot when it lands
        #    under a vehicle with capacity). Both numbers are drawn in every
        #    period, whether or not a unit arrives and wherever it lands, so
        #    every period uses the same amount of randomness.
        u_arrival = context.rng.random()
        u_place = context.rng.random()
        if u_arrival < self.order_probability:
            g = self.grid_size
            cell = self._arrival_cell(state, u_place)
            row = cell // g
            col = cell % g
            server = -1
            i = 0
            while i < self.n_vehicles:
                v = state.vehicles[i]
                if server < 0 and v.row == row and v.col == col and v.capacity_left > 0:
                    server = i
                i += 1
            if server >= 0:
                state.vehicles[server].capacity_left -= 1
            else:
                self._add_unit(state, row, col)
        # 5. waiting costs, clock, next decision round
        context.cumulative_cost += self.holding_cost * float(state.total_weight)
        context.time_elapsed += 1
        self._advance_cursor(state, 0)

    def write_action_validity(self, state: State, valid: Validity) -> None:
        # Admissible moves: stay and the in-grid neighbors (the paper's B_{i,v}).
        # CELL: mark those cells in the grid_size**2 mask, everything else False.
        # DIRECTIONAL: the same five possibilities as a 5-mask (off-grid False).
        v = state.vehicles[state.current_vehicle]
        g = self.grid_size
        if self.action_encoding == ActionEncoding.CELL:
            a = 0
            while a < g * g:
                valid.set(a, False)
                a += 1
            cell = v.row * g + v.col
            valid.set(cell, True)
            if v.row > 0:
                valid.set(cell - g, True)
            if v.row < g - 1:
                valid.set(cell + g, True)
            if v.col > 0:
                valid.set(cell - 1, True)
            if v.col < g - 1:
                valid.set(cell + 1, True)
        else:
            valid.set(0, True)
            valid.set(1, v.row > 0)
            valid.set(2, v.row < g - 1)
            valid.set(3, v.col > 0)
            valid.set(4, v.col < g - 1)


# ---- benchmark policies ---------------------------------------------------


@const_dataclass(slots=True)
class MyopicPolicy:
    """The paper's myopic benchmark (Appendix C.1): move toward the closest
    waiting order — each admissible next location is valued by its distance to
    the nearest order. Between locations that are equally close to an order,
    the one whose nearest order is heaviest wins; what is still tied is broken
    uniformly at random. With no orders in the system the vehicle stays put.

    Encoding-agnostic: the CHOICE (a candidate move 0..4, drawn identically
    under both encodings — same policy_rng stream) is made first, and only
    `_encode` translates it to the MDP's action space. Trajectories are
    therefore bit-identical across encodings given the same seed."""

    mdp: VehicleRoutingMDP

    def _encode(self, state: State, k: int) -> int:
        """The action index for candidate move k (0=stay, 1=up, 2=down,
        3=left, 4=right) of the deciding vehicle, in the MDP's encoding."""
        if self.mdp.action_encoding == ActionEncoding.DIRECTIONAL:
            return k
        g = self.mdp.grid_size
        v = state.vehicles[state.current_vehicle]
        cell = v.row * g + v.col
        if k == 1:
            return cell - g
        if k == 2:
            return cell + g
        if k == 3:
            return cell - 1
        if k == 4:
            return cell + 1
        return cell

    def _nearest(self, state: State, row: int, col: int) -> int:
        """The value of standing on (row, col), lower is better: the distance to
        the nearest order, and between equal distances the heavier order. One
        number: distance * 1000000 - weight of the heaviest order that near."""
        best = 1000000
        heaviest = 0
        for order in state.orders:
            dr = order.row - row
            if dr < 0:
                dr = -dr
            dc = order.col - col
            if dc < 0:
                dc = -dc
            if dr + dc < best:
                best = dr + dc
                heaviest = order.weight
            elif dr + dc == best and order.weight > heaviest:
                heaviest = order.weight
        return best * 1000000 - heaviest

    def get_action(self, state: State, context: TrajectoryContext) -> int:
        g = self.mdp.grid_size
        v = state.vehicles[state.current_vehicle]
        if len(state.orders) == 0:
            return self._encode(state, 0)
        # candidate next locations in fixed order: stay, up, down, left, right
        d0 = self._nearest(state, v.row, v.col)
        d1 = 2000000000000
        d2 = 2000000000000
        d3 = 2000000000000
        d4 = 2000000000000
        if v.row > 0:
            d1 = self._nearest(state, v.row - 1, v.col)
        if v.row < g - 1:
            d2 = self._nearest(state, v.row + 1, v.col)
        if v.col > 0:
            d3 = self._nearest(state, v.row, v.col - 1)
        if v.col < g - 1:
            d4 = self._nearest(state, v.row, v.col + 1)
        best = d0
        if d1 < best:
            best = d1
        if d2 < best:
            best = d2
        if d3 < best:
            best = d3
        if d4 < best:
            best = d4
        ties = 0
        if d0 == best:
            ties += 1
        if d1 == best:
            ties += 1
        if d2 == best:
            ties += 1
        if d3 == best:
            ties += 1
        if d4 == best:
            ties += 1
        pick = context.policy_rng.choice(ties)
        if d0 == best:
            if pick == 0:
                return self._encode(state, 0)
            pick -= 1
        if d1 == best:
            if pick == 0:
                return self._encode(state, 1)
            pick -= 1
        if d2 == best:
            if pick == 0:
                return self._encode(state, 2)
            pick -= 1
        if d3 == best:
            if pick == 0:
                return self._encode(state, 3)
            pick -= 1
        return self._encode(state, 4)
