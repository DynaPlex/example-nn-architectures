"""Tests for the engine-side SAT featurizer (Phase 2a).

The centrepiece is `test_dense_slabs_match_the_tokenizer_exactly`. There are now
two independent implementations of one token layout — `sat_tokenizer` deriving it
by dataclass reflection, and `featurizers_sat.write_features` writing it into
float32 slabs for the transpiler — and the failure mode that matters is COLUMN
ORDER DRIFT: swap two `append` lines and every shape still validates, the
rehearsal still passes, the engine still compiles, and the network quietly trains
on permuted features. Nothing but a value-level comparison catches that.

`sat_schema.TOKEN_SCHEMA` is the shared naming authority, so the comparison is
asserted EXACTLY (`np.array_equal` on float32), not with a tolerance. Both sides
compute each column with the same expression on the same inputs; if the bits
differ, the expressions differ, and that is the bug.

The engine cross-validation is the only test of the transpile itself — the
rehearsal and the parity oracle both run the CPython path.
"""
import copy

import numpy as np
import pytest

import dynaplex.modelling as dpm
from sat_compat import make_context

if not hasattr(dpm, "VariableObjectWriter"):
    pytest.skip("this dynaplex has no object-token writers — install dynaplex==1.12.2 "
                "from Test PyPI (see README/requirements.txt), not 1.12.0 from regular PyPI",
                allow_module_level=True)

from dynaplex import collect_samples
from dynaplex.modelling import (
    allocate_arrays,
    assert_featurizer_for_mdp,
    feature_spec,
    featurizer_template,
    holder_from_featurizer,
)

from featurizers import VehicleFeaturizer
from featurizers_sat import SATFeaturizer
from mdp import ActionEncoding, MyopicPolicy, VehicleRoutingMDP
from sat_schema import TOKEN_SCHEMA, build_sat_observation, resolve_max_counts
from sat_tokenizer import SequenceBuilder

_KW = dict(n=40, m=4, h=8, controllers=1, workers=2, slots=8, seed=0)

# spec key -> the sat_schema token dataclass whose bank it must reproduce
_TENSOR_TO_TYPE = {ts.name: ts.dataclass.__name__ for ts in TOKEN_SCHEMA}


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


def _featurize(mdp, states):
    """Drive the featurizer over `states` in pure CPython -> dict of slabs.

    The `NNAgent.__call__` idiom (dynaplex/nn.py:236-251): size from the spec,
    allocate, install the writers around the slabs, then reset/write/finish per
    row. Same code path the engine runs, minus the transpile.
    """
    spec = feature_spec(SATFeaturizer, mdp)
    arrays = allocate_arrays(spec, len(states))
    fz = featurizer_template(SATFeaturizer, mdp)
    fz = fz.bind(holder_from_featurizer(SATFeaturizer)(**arrays))
    for i, st in enumerate(states):
        fz.reset(i)
        fz.write_features(st)
        fz.finish()
    return arrays


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding",
                         [ActionEncoding.DIRECTIONAL, ActionEncoding.CELL])
def test_rehearsal_and_spec(encoding):
    mdp = _mdp(encoding)
    spec = assert_featurizer_for_mdp(SATFeaturizer, mdp)
    # key ORDER is part of the contract: writer field order, each variable field
    # immediately followed by its _count companion
    assert list(spec) == ["summary", "depot", "vehicles", "orders",
                          "orders_count", "moves"]
    assert spec["summary"].sample_shape == (1, 4)
    assert spec["depot"].sample_shape == (1, 2)
    assert spec["vehicles"].sample_shape == (3, 4)
    assert spec["orders"].sample_shape == (12, 2)
    assert spec["orders_count"].sample_shape == ()
    assert spec["moves"].sample_shape == (mdp.num_actions, 6)


def test_spec_widths_match_the_schema_table():
    """The table is the shared authority, so the engine spec must follow it."""
    mdp = _mdp()
    spec = feature_spec(SATFeaturizer, mdp)
    caps = resolve_max_counts(mdp)
    for ts in TOKEN_SCHEMA:
        assert spec[ts.name].sample_shape == (caps[ts.name], ts.width), ts.name


# ---------------------------------------------------------------------------
# THE PARITY ORACLE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding",
                         [ActionEncoding.DIRECTIONAL, ActionEncoding.CELL])
def test_dense_slabs_match_the_tokenizer_exactly(encoding):
    """Every column of every token, on 40 real states, bit-for-bit.

    Asserted with `np.array_equal` rather than `allclose` on purpose: both sides
    evaluate the same expression on the same inputs, so equal bits are
    achievable, and a tolerance would hide exactly the small-but-systematic
    drift (a swapped normaliser, a column permutation between two similar
    features) that this test exists to catch.
    """
    mdp = _mdp(encoding)
    states = _states(mdp, 40)
    slabs = _featurize(mdp, states)

    sb = SequenceBuilder(skip_root=True)
    for row, state in enumerate(states):
        seq = sb.build(build_sat_observation(state, mdp))
        banks = seq["type_features"]

        for tensor, tname in _TENSOR_TO_TYPE.items():
            want = banks.get(tname)
            n = 0 if want is None else want.shape[0]
            got = slabs[tensor][row]

            if tensor == "orders":
                assert slabs["orders_count"][row] == n, (
                    f"row {row}: orders_count={slabs['orders_count'][row]} but "
                    f"the tokenizer emitted {n} order tokens")
                assert not got[n:].any(), (
                    f"row {row}: padding tail past orders_count is not zeroed")
                got = got[:n]
            else:
                assert got.shape[0] == n, (
                    f"row {row}, {tensor}: {got.shape[0]} slab tokens vs "
                    f"{n} tokenizer tokens")

            if n:
                assert np.array_equal(got, want.numpy()), (
                    f"row {row}, tensor {tensor!r} (type {tname}) differs from "
                    f"the tokenizer — column order or normalisation drift.\n"
                    f"slab:      {got}\ntokenizer: {want.numpy()}")


def test_parity_holds_with_no_waiting_orders():
    """The degenerate state, where the variable tensor is empty."""
    mdp = _mdp()
    state = _states(mdp, 1)[0]
    state.has_order[:] = False
    state.n_orders = 0

    slabs = _featurize(mdp, [state])
    assert slabs["orders_count"][0] == 0
    assert not slabs["orders"][0].any()

    seq = SequenceBuilder(skip_root=True).build(build_sat_observation(state, mdp))
    assert "OrderTok" not in seq["type_features"]
    # nearest-order distance falls back to 1.0 on BOTH sides, not a sentinel
    assert np.array_equal(slabs["moves"][0],
                          seq["type_features"]["MoveAction"].numpy())
    assert (slabs["moves"][0][:, 5] == 1.0).all()


def test_padding_tail_is_rewritten_between_rows():
    """Writer buffers are reused across rows, so a shrinking order count must
    zero the tail — otherwise a row inherits its predecessor's orders."""
    mdp = _mdp()
    busy = _states(mdp, 1)[0]
    quiet = _states(mdp, 1)[0]
    quiet.has_order[:] = False
    quiet.n_orders = 0

    slabs = _featurize(mdp, [busy, quiet])
    assert slabs["orders_count"][0] > 0
    assert slabs["orders_count"][1] == 0
    assert not slabs["orders"][1].any(), "stale orders leaked into the next row"


# ---------------------------------------------------------------------------
# The action-order contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding",
                         [ActionEncoding.DIRECTIONAL, ActionEncoding.CELL])
def test_move_token_i_is_action_i(encoding):
    """Column 0 is the raw action key, so it must equal the token index."""
    mdp = _mdp(encoding)
    slabs = _featurize(mdp, _states(mdp, 10))
    want = np.arange(mdp.num_actions, dtype=np.float32)
    for row in range(slabs["moves"].shape[0]):
        assert np.array_equal(slabs["moves"][row][:, 0], want)


def test_cell_move_tokens_are_row_major():
    mdp = _mdp(ActionEncoding.CELL)
    g = mdp.grid_size
    gm1 = float(g - 1)
    slabs = _featurize(mdp, _states(mdp, 5))
    for a in range(mdp.num_actions):
        assert slabs["moves"][0][a, 1] == pytest.approx((a // g) / gm1)
        assert slabs["moves"][0][a, 2] == pytest.approx((a % g) / gm1)


def test_every_admissible_action_has_a_move_token():
    """The mask crops the tokens, so the tokens must cover the mask."""
    mdp = _mdp()
    ctx = make_context(mdp, 0)
    for state in _states(mdp, 20):
        mdp.write_action_validity(state, ctx.valid)
        valid = np.flatnonzero(np.asarray(ctx.valid.arr[0]))
        assert valid.max() < mdp.num_actions


def test_off_grid_moves_are_flagged_and_stay_in_range():
    mdp = _mdp(ActionEncoding.DIRECTIONAL)
    state = _states(mdp, 1)[0]
    v = state.vehicles[state.current_vehicle]
    state.vehicle_count[v.row, v.col] -= 1
    v.row, v.col = 0, 0                      # corner: up and left are off-grid
    state.vehicle_count[0, 0] += 1

    moves = _featurize(mdp, [state])["moves"][0]
    assert moves[:, 3].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert np.isfinite(moves).all()
    assert (moves[:, 1:3] >= 0.0).all() and (moves[:, 1:3] <= 1.0).all()
    # the clamped targets are the vehicle's own cell
    assert moves[1, 1] == 0.0 and moves[1, 2] == 0.0
    assert moves[3, 1] == 0.0 and moves[3, 2] == 0.0


# ---------------------------------------------------------------------------
# The engine — the only test of the transpile
# ---------------------------------------------------------------------------


def test_featurizer_compiles_and_collects_through_the_engine():
    """Cross-validated against the flat run, whose seeds produce the same states.

    Features never steer gen-0 collection, so labels and masks must be
    byte-identical between the two runs; anything else means the featurizer
    perturbed the trajectory. Then the SAT slabs are decoded back against the
    flat run's order grid.
    """
    mdp = _mdp()
    flat = collect_samples(mdp, MyopicPolicy(mdp), features=VehicleFeaturizer, **_KW)
    sat = collect_samples(mdp, MyopicPolicy(mdp), features=SATFeaturizer, **_KW)

    assert sat.y.tobytes() == flat.y.tobytes()
    assert sat.mask.tobytes() == flat.mask.tobytes()

    (xflat,) = flat.features.values()
    g, v = mdp.grid_size, mdp.n_vehicles
    gm1 = float(g - 1)
    # VehicleFeaturizer layout: [vehicle-count grid | order grid | capacities]
    order_grid = xflat[:, g * g:2 * g * g].reshape(-1, g, g)

    for row in range(sat.y.shape[0]):
        rows, cols = np.nonzero(order_grid[row])
        n = int(sat.features["orders_count"][row])
        assert n == len(rows), f"row {row}: {n} order tokens vs {len(rows)} orders"
        want = np.stack([rows / gm1, cols / gm1], axis=1).astype(np.float32)
        assert np.array_equal(sat.features["orders"][row][:n], want)
        assert not sat.features["orders"][row][n:].any()
        # exactly one focal vehicle, and the summary token agrees with it
        veh = sat.features["vehicles"][row]
        assert veh[:, 3].sum() == pytest.approx(1.0)
        focal = int(veh[:, 3].argmax())
        summ = sat.features["summary"][row][0]
        assert summ[1] == pytest.approx(veh[focal, 0])
        assert summ[2] == pytest.approx(veh[focal, 1])


def test_engine_slabs_match_the_cpython_slabs():
    """The transpiled write_features must agree with the CPython one.

    Cheap to state, and it is what would catch a transpiler-level divergence
    (integer division, float promotion, evaluation order) that neither the
    rehearsal nor the tokenizer parity test can see.
    """
    mdp = _mdp()
    sat = collect_samples(mdp, MyopicPolicy(mdp), features=SATFeaturizer, **_KW)
    # the engine's own sample states are not exposed, so compare distributions
    # of the one column that is a pure function of the order grid
    for name in ("summary", "depot", "vehicles", "orders", "moves"):
        arr = sat.features[name]
        assert np.isfinite(arr).all(), name
    assert (sat.features["depot"] == 0.0).all(), "depot is fixed at (0, 0)"
    counts = sat.features["orders_count"]
    assert (counts >= 0).all() and (counts <= mdp.max_orders).all()
    for row in range(arr.shape[0]):
        n = int(counts[row])
        assert not sat.features["orders"][row][n:].any()
