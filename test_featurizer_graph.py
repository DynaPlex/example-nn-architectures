"""GridGraphFeaturizer + gin_net: node order, node features, logits per node."""
import numpy as np
import pytest

import dynaplex as dp
from dynaplex.modelling import assert_featurizer_for_mdp

from featurizer_flat import VehicleFeaturizer
from featurizer_graph import GridGraphFeaturizer
from mdp import MyopicPolicy, VehicleRoutingMDP


def _mdp():
    return VehicleRoutingMDP(grid_size=6, n_vehicles=3, capacity=3, max_orders=12)


_KW = dict(n=40, m=4, h=8, controllers=1, workers=2, slots=8, seed=0)


def test_spec_is_synthesized_one_node_per_cell():
    spec = assert_featurizer_for_mdp(GridGraphFeaturizer, _mdp())
    assert [(k, ts.sample_shape) for k, ts in spec.items()] == [("nodes", (36, 5))]


def test_node_features_match_flat_run(tmp_path):
    mdp = _mdp()
    g = mdp.grid_size
    # one collection; both featurizers are applied to the states it stored
    graph_run = dp.collect_samples(mdp, MyopicPolicy(mdp), records=str(tmp_path / "records"), **_KW)
    states = str(tmp_path / "records" / "states")
    (xflat,) = dp.featurize(mdp, states, VehicleFeaturizer).values()
    nodes = dp.featurize(mdp, states, GridGraphFeaturizer)["nodes"]

    # exactly one focal node per state, and it is the one admissible "stay" cell:
    # the focal cell is always valid, and the valid set is its neighbourhood
    focal = nodes[:, :, 0]
    assert (focal.sum(axis=1) == 1.0).all()
    focal_cell = focal.argmax(axis=1)
    assert graph_run.mask[np.arange(len(graph_run)), focal_cell].all()
    # distance feature is zero exactly at the focal node
    dist = nodes[:, :, 4]
    assert (dist.argmin(axis=1) == focal_cell).all()
    assert (dist[np.arange(len(graph_run)), focal_cell] == 0.0).all()
    # other-vehicle counts: the flat run's vehicle-count grid minus the focal one
    counts = xflat[:, :g * g].copy()
    counts[np.arange(len(graph_run)), focal_cell] -= 1
    assert np.allclose(nodes[:, :, 1] * (mdp.n_vehicles - 1), counts)
    # order presence equals the flat run's order grid, node for cell
    assert np.array_equal(nodes[:, :, 2], xflat[:, g * g:2 * g * g])
    # the depot flag marks cell (0, 0) only
    assert (nodes[:, 0, 3] == 1.0).all() and not nodes[:, 1:, 3].any()


def test_gin_net_one_logit_per_node():
    torch = pytest.importorskip("torch")
    from dynaplex.modelling import feature_spec
    from networks import gin_net

    mdp = _mdp()
    spec = feature_spec(GridGraphFeaturizer, mdp)
    net = gin_net(spec, mdp.num_actions, d_model=16, n_layers=2)
    B, n = 4, mdp.num_actions
    out = net({"nodes": torch.rand(B, n, 5)})
    assert out.shape == (B, n) and torch.isfinite(out).all()
    out.sum().backward()
    with pytest.raises(AssertionError, match="CELL"):
        gin_net(spec, 5)


def test_gin_message_passing_is_local_and_readout_is_global():
    """Two layers reach two hops: perturbing a node changes the node
    embeddings within Manhattan distance 2 and nowhere else. The readout then
    mixes in the graph mean, so every logit changes."""
    torch = pytest.importorskip("torch")
    from dynaplex.modelling import feature_spec
    from networks import gin_net

    mdp = _mdp()
    g = mdp.grid_size
    net = gin_net(feature_spec(GridGraphFeaturizer, mdp), mdp.num_actions,
                  d_model=16, n_layers=2).eval()
    x = torch.rand(1, g * g, 5)
    x2 = x.clone()
    r0, c0 = 3, 3
    x2[0, r0 * g + c0] += 1.0
    with torch.no_grad():
        changed = (net.embed(x2) != net.embed(x)).any(dim=-1)[0]
        logits_changed = (net({"nodes": x2}) != net({"nodes": x}))[0]
    for r in range(g):
        for c in range(g):
            assert changed[r * g + c] == (abs(r - r0) + abs(c - c0) <= 2)
    assert logits_changed.all()
