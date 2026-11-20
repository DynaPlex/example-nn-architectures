"""Dynamics invariants, the deliberate departures from the paper (order cap,
automatic depot return, arrival redraw / on-the-spot serving), and the engine
integration via PolicyComparer."""
import numpy as np
import pytest

import dynaplex as dp
from dynaplex.models import RandomPolicy
from dynaplex.modelling import StateCategory, ValidityWriter

from mdp import ActionEncoding, MyopicPolicy, State, Vehicle, VehicleRoutingMDP


def _counts(vehicles, g):
    counts = np.zeros((g, g), dtype=np.int64)
    for v in vehicles:
        counts[v.row, v.col] += 1
    return counts


def _mask(mdp, state):
    mask = np.zeros((1, mdp.num_actions), dtype=np.bool_)
    mdp.write_action_validity(state, ValidityWriter(mask, 0))
    return mask[0]


def _step_period(mdp, policy, state, ctx):
    """Drive one full period: all pending decisions, then the event."""
    while state.category == StateCategory.AWAIT_ACTION:
        a = policy.get_action(state, ctx)
        assert _mask(mdp, state)[a]
        mdp.modify_state_with_action(state, ctx, a)
    mdp.modify_state_with_event(state, ctx)


def test_static_validation():
    with pytest.raises(AssertionError):
        VehicleRoutingMDP(grid_size=3, max_orders=10)
    VehicleRoutingMDP(grid_size=3, max_orders=9)  # boundary is allowed


def test_dynamics_invariants_under_myopic():
    mdp = VehicleRoutingMDP()
    policy = MyopicPolicy(mdp)
    ctx = dp.modelling.new_context(mdp, 7)
    state = mdp.get_initial_state(ctx)
    g = mdp.grid_size
    for _ in range(400):
        prev = [(v.row, v.col) for v in state.vehicles]
        prev_cost = ctx.cumulative_cost
        _step_period(mdp, policy, state, ctx)
        # counter mirrors the grid, cap respected
        assert state.n_orders == int(state.has_order.sum())
        # count grid mirrors the fleet exactly
        assert np.array_equal(state.vehicle_count, _counts(state.vehicles, g))
        assert 0 <= state.n_orders <= mdp.max_orders
        for i, v in enumerate(state.vehicles):
            assert 0 <= v.row < g and 0 <= v.col < g
            assert 0 <= v.capacity_left <= mdp.capacity
            # one Manhattan step per period, at most
            assert abs(v.row - prev[i][0]) + abs(v.col - prev[i][1]) <= 1
            # a vehicle standing at the depot has been reset (reset happens
            # after moving, and arrivals cannot undo it past 0 while it is full)
            if (v.row, v.col) == (mdp.depot.row, mdp.depot.col):
                assert v.capacity_left >= mdp.capacity - 1
        # the decision cursor only ever points at a vehicle with capacity
        if state.category == StateCategory.AWAIT_ACTION:
            assert state.vehicles[state.current_vehicle].capacity_left > 0
        assert ctx.cumulative_cost >= prev_cost
    assert ctx.time_elapsed == 400


def test_full_vehicle_walks_up_then_left_and_resets():
    mdp = VehicleRoutingMDP(grid_size=5, n_vehicles=1, capacity=4,
                            max_orders=10, order_probability=0.0)
    ctx = dp.modelling.new_context(mdp, 0)
    vehicles = [Vehicle(row=2, col=3, capacity_left=0)]
    state = State(
        vehicles=vehicles,
        has_order=np.zeros((5, 5), dtype=np.bool_),
        vehicle_count=_counts(vehicles, 5),
        n_orders=0,
        current_vehicle=0,
        category=StateCategory.AWAIT_EVENT,  # full vehicle: nothing to decide
    )
    expected = [(1, 3), (0, 3), (0, 2), (0, 1), (0, 0)]
    for row, col in expected:
        assert state.category == StateCategory.AWAIT_EVENT
        mdp.modify_state_with_event(state, ctx)
        v = state.vehicles[0]
        assert (v.row, v.col) == (row, col)
    # capacity reset at the depot, so the vehicle decides again
    assert state.vehicles[0].capacity_left == mdp.capacity
    assert state.category == StateCategory.AWAIT_ACTION


def test_order_cap_binds():
    mdp = VehicleRoutingMDP(grid_size=6, n_vehicles=1, max_orders=3,
                            order_probability=1.0)
    policy = RandomPolicy(mdp)
    ctx = dp.modelling.new_context(mdp, 11)
    state = mdp.get_initial_state(ctx)
    assert state.n_orders <= 3  # initial draw respects the cap
    hit_cap = False
    for _ in range(200):
        _step_period(mdp, policy, state, ctx)
        assert state.n_orders <= 3
        hit_cap = hit_cap or state.n_orders == 3
    assert hit_cap  # with p=1 and one vehicle the cap actually binds


def test_arrival_redraw_and_on_the_spot_serving():
    # 2x2 grid, orders everywhere except under the vehicle: the only cell an
    # arrival can land on (after redraws) is the vehicle's, so it must be
    # served on the spot every period — n_orders stays put, capacity drains.
    mdp = VehicleRoutingMDP(grid_size=2, n_vehicles=1, capacity=3, max_orders=4,
                            order_probability=1.0)
    has_order = np.ones((2, 2), dtype=np.bool_)
    has_order[1, 1] = False
    vehicles = [Vehicle(row=1, col=1, capacity_left=3)]
    state = State(
        vehicles=vehicles,
        has_order=has_order,
        vehicle_count=_counts(vehicles, 2),
        n_orders=3,
        current_vehicle=0,
        category=StateCategory.AWAIT_ACTION,
    )
    ctx = dp.modelling.new_context(mdp, 5)
    stay = 1 * 2 + 1  # the vehicle's own cell (1,1) as a cell-index action
    for expected_cap in (2, 1):
        mdp.modify_state_with_action(state, ctx, stay)
        mdp.modify_state_with_event(state, ctx)
        assert state.vehicles[0].capacity_left == expected_cap
        assert state.n_orders == 3
        assert not state.has_order[1, 1]


def test_validity_masks_grid_edges():
    mdp = VehicleRoutingMDP(grid_size=4, n_vehicles=1, max_orders=10)
    vehicles = [Vehicle(row=0, col=0, capacity_left=5)]
    state = State(
        vehicles=vehicles,
        has_order=np.zeros((4, 4), dtype=np.bool_),
        vehicle_count=_counts(vehicles, 4),
        n_orders=0,
        current_vehicle=0,
        category=StateCategory.AWAIT_ACTION,
    )
    def admissible_cells(state):
        m = _mask(mdp, state).reshape(4, 4)
        return {(r, c) for r in range(4) for c in range(4) if m[r, c]}

    # corner (0,0): stay, down, right
    assert admissible_cells(state) == {(0, 0), (1, 0), (0, 1)}
    state.vehicles[0].row, state.vehicles[0].col = 2, 2  # interior: all 5
    assert admissible_cells(state) == {(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)}
    state.vehicles[0].row, state.vehicles[0].col = 3, 1  # bottom edge: no down
    assert admissible_cells(state) == {(3, 1), (2, 1), (3, 0), (3, 2)}


def test_comparer_runs_engine_and_is_reproducible():
    mdp = VehicleRoutingMDP()
    comparer = dp.PolicyComparer(mdp, number_of_trajectories=32, warmup_time=32,
                              horizon=128, seed=1, workers=2)
    a = comparer.assess(MyopicPolicy(mdp))
    b = comparer.assess(MyopicPolicy(mdp))
    assert np.array_equal(a.returns, b.returns)
    assert a.backend == "engine"
    # myopic beats random moving, comfortably
    r = comparer.assess(RandomPolicy(mdp))
    assert a.returns.mean() < r.returns.mean()


# ---- the two action encodings (mdp.ActionEncoding) -------------------------


def _snapshot(state):
    return ([(v.row, v.col, v.capacity_left) for v in state.vehicles],
            state.has_order.tobytes(), state.n_orders)


def test_encodings_are_bit_identical():
    """Only the action NAMES differ between the encodings: with the same seed,
    myopic drives byte-identical trajectories and costs under both."""
    mdp_c = VehicleRoutingMDP(action_encoding=ActionEncoding.CELL)
    mdp_d = VehicleRoutingMDP(action_encoding=ActionEncoding.DIRECTIONAL)
    assert mdp_c.num_actions == 100 and mdp_d.num_actions == 5
    ctx_c = dp.modelling.new_context(mdp_c, 11)
    ctx_d = dp.modelling.new_context(mdp_d, 11)
    s_c = mdp_c.get_initial_state(ctx_c)
    s_d = mdp_d.get_initial_state(ctx_d)
    for _ in range(200):
        _step_period(mdp_c, MyopicPolicy(mdp_c), s_c, ctx_c)
        _step_period(mdp_d, MyopicPolicy(mdp_d), s_d, ctx_d)
        assert _snapshot(s_c) == _snapshot(s_d)
        assert ctx_c.cumulative_cost == ctx_d.cumulative_cost


def test_directional_mask_maps_onto_cell_mask():
    """For every decision state along an episode, the DIRECTIONAL mask's valid
    moves name exactly the CELL mask's valid cells."""
    mdp_c = VehicleRoutingMDP(action_encoding=ActionEncoding.CELL)
    mdp_d = VehicleRoutingMDP(action_encoding=ActionEncoding.DIRECTIONAL)
    g = mdp_c.grid_size
    ctx = dp.modelling.new_context(mdp_c, 5)
    state = mdp_c.get_initial_state(ctx)
    policy = MyopicPolicy(mdp_c)
    checked = 0
    for _ in range(50):
        while state.category == StateCategory.AWAIT_ACTION:
            v = state.vehicles[state.current_vehicle]
            cell = v.row * g + v.col
            moves = {0: cell, 1: cell - g, 2: cell + g, 3: cell - 1, 4: cell + 1}
            dir_mask = _mask(mdp_d, state)      # same State class, either mdp reads it
            cell_mask = _mask(mdp_c, state)
            assert {moves[k] for k in range(5) if dir_mask[k]} == \
                set(np.flatnonzero(cell_mask))
            checked += 1
            mdp_c.modify_state_with_action(state, ctx, policy.get_action(state, ctx))
        mdp_c.modify_state_with_event(state, ctx)
    assert checked > 100


def test_directional_encoding_compiles_in_engine():
    """The enum dispatch in modify_state_with_action / write_action_validity
    compiles; myopic evaluates identically under both encodings (common seeds)."""
    kw = dict(number_of_trajectories=16, warmup_time=16, horizon=64, seed=2, workers=2)
    mdp_c = VehicleRoutingMDP()
    a_c = dp.PolicyComparer(mdp_c, **kw).assess(MyopicPolicy(mdp_c))
    mdp_d = VehicleRoutingMDP(action_encoding=ActionEncoding.DIRECTIONAL)
    a_d = dp.PolicyComparer(mdp_d, **kw).assess(MyopicPolicy(mdp_d))
    assert a_d.backend == "engine"
    assert np.array_equal(a_c.returns, a_d.returns)
