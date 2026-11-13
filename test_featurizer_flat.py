"""Featurizer tests: width, count-map semantics, and engine collection."""
from __future__ import annotations

import dynaplex
import numpy as np
from dynaplex.dcl import collect_samples
from dynaplex.modelling import GlobalStateWriter, StateCategory

from featurizer_flat import VehicleFeaturizer
from mdp import MyopicPolicy, State, Vehicle, VehicleRoutingMDP


def _width(mdp):
    g, nv = mdp.grid_size, mdp.n_vehicles
    return 2 * g * g + nv


def _counts(vehicles, g):
    counts = np.zeros((g, g), dtype=np.int64)
    for v in vehicles:
        counts[v.row, v.col] += 1
    return counts


def _features(mdp: VehicleRoutingMDP, state: State) -> np.ndarray:
    """CPython mirror: run write_features into a fresh single-row writer."""
    width = _width(mdp)
    arr = np.zeros((1, width), np.float32)
    VehicleFeaturizer(mdp=mdp, v=GlobalStateWriter(arr)).write_features(state)
    return arr[0]


def _state(vehicles, has_order, current):
    g = has_order.shape[0]
    return State(vehicles=vehicles, has_order=has_order,
                 vehicle_count=_counts(vehicles, g),
                 n_orders=int(has_order.sum()), current_vehicle=current,
                 category=StateCategory.AWAIT_ACTION)


def test_width_and_layout():
    mdp = VehicleRoutingMDP(grid_size=4, n_vehicles=3, capacity=5, max_orders=6)
    g = mdp.grid_size
    has_order = np.zeros((g, g), np.bool_)
    has_order[3, 3] = True
    # two vehicles share a cell: the count map shows 2 there
    vehicles = [Vehicle(1, 2, 5), Vehicle(1, 2, 3), Vehicle(2, 1, 1)]
    f = _features(mdp, _state(vehicles, has_order, current=1))
    assert f.shape == (2 * 16 + 3,)
    counts = f[:16].reshape(g, g)
    assert counts[1, 2] == 2.0 and counts[2, 1] == 1.0 and counts.sum() == 3.0
    orders = f[16:32].reshape(g, g)
    assert orders[3, 3] == 1.0 and orders.sum() == 1.0
    # capacities: deciding vehicle (index 1) first, wrapping
    assert f[-3:].tolist() == [3.0, 1.0, 5.0]


def test_permutation_invariance():
    # relabeling vehicles with equal capacities leaves the features identical
    mdp = VehicleRoutingMDP(grid_size=4, n_vehicles=3, capacity=5, max_orders=6)
    has_order = np.zeros((4, 4), np.bool_)
    has_order[0, 3] = True
    vehicles = [Vehicle(0, 0, 5), Vehicle(1, 2, 5), Vehicle(2, 1, 5)]
    f0 = _features(mdp, _state(list(vehicles), has_order, current=0))
    swapped = [vehicles[0], vehicles[2], vehicles[1]]
    f1 = _features(mdp, _state(swapped, has_order, current=0))
    assert (f0 == f1).all()


def test_engine_collection_reproducible(tmp_path):
    # Byte-exact reproducibility at a fixed configuration. (Worker-count
    # invariance is NOT asserted: on this MDP the number of decisions per
    # period varies — full vehicles are skipped — which trips a known
    # worker-split sensitivity in DCL collection.)
    mdp = VehicleRoutingMDP()
    a = collect_samples(mdp, MyopicPolicy(mdp), records=str(tmp_path / "a"),
                        n=8, m=4, h=10, workers=2, slots=4, seed=5)
    b = collect_samples(mdp, MyopicPolicy(mdp), records=str(tmp_path / "b"),
                        n=8, m=4, h=10, workers=2, slots=4, seed=5)
    # a collection stores states; the features are made from them
    (xa,) = dynaplex.featurize(mdp, str(tmp_path / "a" / "states"), VehicleFeaturizer).values()
    (xb,) = dynaplex.featurize(mdp, str(tmp_path / "b" / "states"), VehicleFeaturizer).values()
    assert xa.shape == (8, _width(mdp))
    assert a.mask.shape == (8, mdp.num_actions)
    assert xa.tobytes() == xb.tobytes()
    assert a.y.tobytes() == b.y.tobytes()
    assert a.mask.tobytes() == b.mask.tobytes()
    # count-map sanity on collected rows: fleet size everywhere
    assert (xa[:, :100].sum(axis=1) == mdp.n_vehicles).all()
