"""Tests for the SAT schema + automatic tokenizer (Phase 1).

The load-bearing contracts here are the ones the engine bridge will inherit:

  * the token layout is STATIC — a closed type universe with fixed per-type
    widths, so that a `spec()` can be declared once (Phase 2);
  * group g == action g — logit position equals action index, which everything
    downstream (validity mask, masked argmax, DCL labels) assumes;
  * the schema table and the token dataclasses agree, so the CPython tokenizer
    and the engine-side `write_features` cannot drift apart in column order.
"""
import copy
from dataclasses import dataclass, fields

import numpy as np
import pytest
import torch

from sat_compat import make_context

from mdp import ActionEncoding, MyopicPolicy, VehicleRoutingMDP
import sat_schema
from sat_actions import resolve_action
from sat_markers import Action, GlobalState
from sat_schema import (
    TOKEN_SCHEMA,
    SATObservation,
    build_sat_observation,
    nearest_order_distance,
    resolve_max_counts,
    validate_schema,
)
from sat_tokenizer import SequenceBuilder


def _mdp(encoding=ActionEncoding.DIRECTIONAL):
    return VehicleRoutingMDP(grid_size=6, n_vehicles=3, capacity=3,
                            max_orders=12, action_encoding=encoding)


def _states(mdp, n, seed=0):
    """n successive AWAIT_ACTION states under the myopic policy."""
    ctx = make_context(mdp, seed)
    policy = MyopicPolicy(mdp)
    state = mdp.get_initial_state(ctx)
    out = []
    while len(out) < n:
        out.append(copy.deepcopy(state))
        mdp.modify_state_with_action(state, ctx, policy.get_action(state, ctx))
        while state.category.name == "AWAIT_EVENT":
            mdp.modify_state_with_event(state, ctx)
    return out


# ---------------------------------------------------------------------------
# Schema lint — the static-layout preconditions
# ---------------------------------------------------------------------------


def test_schema_table_matches_dataclasses():
    validate_schema()   # raises on column-order drift
    for ts in TOKEN_SCHEMA:
        assert tuple(f.name for f in fields(ts.dataclass)) == ts.columns
        assert ts.width == len(ts.columns)


def test_schema_declares_exactly_the_observation_token_types():
    """Every token type the walk can emit must be declared, and vice versa.

    This is the closed-universe precondition stated as a test: if the
    observation root can reach a dataclass the table does not list, Phase 2 has
    no writer field for it and `spec()` would be incomplete.
    """
    mdp = _mdp()
    sb = SequenceBuilder(skip_root=True)
    sb.build(build_sat_observation(_states(mdp, 1)[0], mdp))
    assert set(sb.type_names) == {ts.dataclass.__name__ for ts in TOKEN_SCHEMA}


def test_declared_widths_match_tokenizer_raw_dims():
    """The table's widths are what Phase 2's spec() will use — they must be
    what the tokenizer actually produces."""
    mdp = _mdp()
    sb = SequenceBuilder(skip_root=True)
    sb.build(build_sat_observation(_states(mdp, 1)[0], mdp))
    by_name = {ts.dataclass.__name__: ts.width for ts in TOKEN_SCHEMA}
    assert sb.raw_dims == by_name


def test_no_str_field_reachable_from_observation():
    """A str column is unrepresentable downstream (float32 slabs, no DynaML
    strings), so the tokenizer rejects it. Assert none is reachable."""
    seen: set[type] = set()

    def walk(tp):
        if tp in seen or not hasattr(tp, "__dataclass_fields__"):
            return
        seen.add(tp)
        from typing import get_args, get_type_hints
        for name, ftype in get_type_hints(tp).items():
            assert ftype is not str, f"{tp.__name__}.{name} is a str"
            for arg in get_args(ftype) or ():
                walk(arg)
            walk(ftype)

    walk(SATObservation)
    assert SATObservation in seen


def test_type_registry_stays_closed_over_many_states():
    """The registry must close on the first build and never grow.

    This is what makes a static spec possible. It works only because
    `build_sat_observation` emits every token type on every state — an empty
    order list is a zero count, not an absent type. Regression guard: an
    initial state has orders, so a bug that dropped empty-list types would only
    show up later, exactly where it is hardest to see.
    """
    mdp = _mdp()
    sb = SequenceBuilder(skip_root=True)
    first = None
    for state in _states(mdp, 60):
        r = sb.build(build_sat_observation(state, mdp))
        if first is None:
            first = list(r["type_names"])
        assert list(r["type_names"]) == first


def test_zero_orders_still_emits_every_type():
    """The degenerate state: no waiting orders at all."""
    mdp = _mdp()
    state = _states(mdp, 1)[0]
    state.has_order[:] = False
    state.n_orders = 0
    obs = build_sat_observation(state, mdp)
    assert obs.orders == []
    # nearest-order distance has no sentinel leaking through
    assert all(a.nearest_order_dist == 1.0 for a in obs.actions)
    sb = SequenceBuilder(skip_root=True)
    r = sb.build(obs)
    # OrderTok contributes no tokens, but must not break the walk
    assert "OrderTok" not in r["type_features"]
    assert r["num_action_groups"] == mdp.num_actions


# ---------------------------------------------------------------------------
# The group -> action contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding",
                         [ActionEncoding.DIRECTIONAL, ActionEncoding.CELL])
def test_group_index_equals_action_index(encoding):
    mdp = _mdp(encoding)
    sb = SequenceBuilder(skip_root=True)
    for state in _states(mdp, 25):
        r = sb.build(build_sat_observation(state, mdp))
        assert r["num_action_groups"] == mdp.num_actions
        keys = [resolve_action(r["_actions"], g)
                for g in range(r["num_action_groups"])]
        assert keys == list(range(mdp.num_actions))
        # group ids on the root tokens are 0..num_actions-1 in order
        roots = r["action_group_ids"][r["is_group_root"]]
        assert roots.tolist() == list(range(mdp.num_actions))


@pytest.mark.parametrize("encoding",
                         [ActionEncoding.DIRECTIONAL, ActionEncoding.CELL])
def test_every_admissible_action_has_a_token(encoding):
    """The mask crops the tokens, so the tokens must be a superset of the mask.

    If the action list were ever compacted to the valid set, a masked argmax
    over logit positions would silently select the wrong action.
    """
    mdp = _mdp(encoding)
    ctx = make_context(mdp, 0)
    sb = SequenceBuilder(skip_root=True)
    for state in _states(mdp, 25):
        r = sb.build(build_sat_observation(state, mdp))
        mdp.write_action_validity(state, ctx.valid)
        valid = np.flatnonzero(np.asarray(ctx.valid.arr[0]))
        assert valid.size >= 1
        assert valid.max() < r["num_action_groups"]


def test_directional_off_grid_actions_are_flagged_and_clamped():
    """Off-grid moves must stay finite and in-range, flagged by on_grid=0.

    The network must emit finite logits for inadmissible actions — a NaN or an
    out-of-range coordinate would poison the masked argmax downstream — so the
    target is clamped to the vehicle's own cell rather than encoded as -1.
    """
    mdp = _mdp(ActionEncoding.DIRECTIONAL)
    state = _states(mdp, 1)[0]
    # park the focal vehicle in the top-left corner: up and left are off-grid
    v = state.vehicles[state.current_vehicle]
    state.vehicle_count[v.row, v.col] -= 1
    v.row, v.col = 0, 0
    state.vehicle_count[0, 0] += 1

    obs = build_sat_observation(state, mdp)
    on_grid = [a.on_grid for a in obs.actions]
    assert on_grid == [1.0, 0.0, 1.0, 0.0, 1.0]   # stay, up, down, left, right
    for a in obs.actions:
        assert 0.0 <= a.row <= 1.0 and 0.0 <= a.col <= 1.0
        assert np.isfinite([a.row, a.col, a.nearest_order_dist]).all()
    # the two off-grid moves were clamped onto the vehicle's own cell
    assert (obs.actions[1].row, obs.actions[1].col) == (0.0, 0.0)
    assert (obs.actions[3].row, obs.actions[3].col) == (0.0, 0.0)


def test_cell_tokens_are_in_row_major_action_order():
    """CELL: action a IS cell (a // G, a % G) — the same convention
    GridTokenFeaturizer relies on."""
    mdp = _mdp(ActionEncoding.CELL)
    g = mdp.grid_size
    obs = build_sat_observation(_states(mdp, 1)[0], mdp)
    gm1 = float(g - 1)
    for a, act in enumerate(obs.actions):
        assert act.action_index == a
        assert act.row == pytest.approx(float(a // g) / gm1)
        assert act.col == pytest.approx(float(a % g) / gm1)


# ---------------------------------------------------------------------------
# Feature correctness
# ---------------------------------------------------------------------------


def test_normalisation_matches_token_featurizer_conventions():
    """Positions over grid_size-1, capacity over capacity, is_focal one-hot —
    the same conventions featurizers_tokens.py uses, so the Phase 2 parity test
    can compare column-for-column."""
    mdp = _mdp()
    state = _states(mdp, 1)[0]
    obs = build_sat_observation(state, mdp)
    gm1 = float(mdp.grid_size - 1)

    assert len(obs.vehicles) == mdp.n_vehicles
    for i, tok in enumerate(obs.vehicles):
        veh = state.vehicles[i]
        assert tok.row == pytest.approx(veh.row / gm1)
        assert tok.col == pytest.approx(veh.col / gm1)
        assert tok.cap_left == pytest.approx(veh.capacity_left / mdp.capacity)
        assert tok.is_focal == (1.0 if i == state.current_vehicle else 0.0)
    assert sum(t.is_focal for t in obs.vehicles) == 1.0

    # order tokens: one per waiting order, row-major
    rows, cols = np.nonzero(state.has_order)
    assert len(obs.orders) == state.n_orders == len(rows)
    for tok, r, c in zip(obs.orders, rows, cols):
        assert tok.row == pytest.approx(r / gm1)
        assert tok.col == pytest.approx(c / gm1)


def test_nearest_order_distance_matches_myopic_policy():
    """The column is the myopic benchmark's own statistic — assert it against
    MyopicPolicy._nearest, the implementation Phase 2 will transcribe."""
    mdp = _mdp()
    policy = MyopicPolicy(mdp)
    for state in _states(mdp, 10):
        for row in range(mdp.grid_size):
            for col in range(mdp.grid_size):
                assert (nearest_order_distance(state, mdp.grid_size, row, col)
                        == policy._nearest(state, row, col))


def test_all_features_are_finite_and_bounded():
    mdp = _mdp()
    sb = SequenceBuilder(skip_root=True)
    for state in _states(mdp, 30):
        r = sb.build(build_sat_observation(state, mdp))
        for tname, feats in r["type_features"].items():
            assert torch.isfinite(feats).all(), tname
            # every column is a normalised quantity or a flag, except the raw
            # action key (column 0 of the action token)
            body = feats[:, 1:] if tname == "MoveAction" else feats
            assert (body >= 0.0).all() and (body <= 1.0).all(), tname


def test_build_is_deterministic():
    mdp = _mdp()
    state = _states(mdp, 1)[0]
    sb = SequenceBuilder(skip_root=True)
    a = sb.build(build_sat_observation(state, mdp))
    b = sb.build(build_sat_observation(state, mdp))
    assert a["seq_len"] == b["seq_len"]
    assert torch.equal(a["type_ids"], b["type_ids"])
    assert torch.equal(a["action_group_ids"], b["action_group_ids"])
    for k in a["type_features"]:
        assert torch.equal(a["type_features"][k], b["type_features"][k])


def test_max_counts_bound_the_actual_token_counts():
    """Phase 2's spec() is these numbers — a state exceeding one would overflow
    a writer, so they must be true bounds, not typical values."""
    mdp = _mdp()
    caps = resolve_max_counts(mdp)
    sb = SequenceBuilder(skip_root=True)
    for state in _states(mdp, 60):
        r = sb.build(build_sat_observation(state, mdp))
        for ts in TOKEN_SCHEMA:
            feats = r["type_features"].get(ts.dataclass.__name__)
            n = 0 if feats is None else feats.shape[0]
            assert n <= caps[ts.name], f"{ts.name}: {n} > {caps[ts.name]}"


# ---------------------------------------------------------------------------
# The hardening guards
# ---------------------------------------------------------------------------


def test_str_field_is_rejected_at_plan_time():
    @dataclass(slots=True)
    class Bad(GlobalState):
        label: str

    @dataclass(slots=True)
    class Obs:
        g: Bad

    with pytest.raises(TypeError, match="str is not a valid token feature"):
        SequenceBuilder().build(Obs(Bad("depot")))


def test_late_discovered_type_raises_instead_of_growing_the_registry():
    """A type absent from the first sample and present later must be an error.

    Upstream grew the registry here and rebuilt the optimizer mid-run. Under a
    frozen observation spec there is no such move — the slab does not exist.
    """
    @dataclass(slots=True)
    class G(GlobalState):
        a: float

    @dataclass(slots=True)
    class Late:
        z: float

    @dataclass(slots=True)
    class A(Action):
        i: int

    @dataclass(slots=True)
    class Obs:
        g: G
        late: list[Late]
        actions: list[A]

    sb = SequenceBuilder(skip_root=True)
    sb.build(Obs(G(1.0), [], [A(0)]))
    with pytest.raises(TypeError, match="type universe must be closed"):
        sb.build(Obs(G(1.0), [Late(9.0)], [A(0)]))


def test_varying_feature_width_raises():
    """Same type name, different width across builds — unrepresentable."""
    @dataclass(slots=True)
    class G(GlobalState):
        a: float

    @dataclass(slots=True)
    class A(Action):
        i: int

    @dataclass(slots=True)
    class Obs:
        g: G
        actions: list[A]

    sb = SequenceBuilder(skip_root=True)
    sb.build(Obs(G(1.0), [A(0)]))
    sb._raw_dims["A"] = 7      # simulate a registry/width disagreement
    with pytest.raises(TypeError, match="static spec"):
        sb.build(Obs(G(1.0), [A(0)]))


def test_validate_schema_catches_column_order_drift(monkeypatch):
    """Reordering a dataclass field without updating the table must fail."""
    broken = sat_schema.TokenSpec(
        name="vehicles", dataclass=sat_schema.VehicleTok, kind="fixed",
        max_count="mdp.n_vehicles",
        columns=("col", "row", "cap_left", "is_focal"),   # row/col swapped
        role="context",
    )
    monkeypatch.setattr(sat_schema, "TOKEN_SCHEMA", (broken,))
    with pytest.raises(AssertionError, match="column-order drift"):
        validate_schema()
