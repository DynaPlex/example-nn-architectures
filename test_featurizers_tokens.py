"""Smoke tests for TokenFeaturizer (the dot-attention token bundle)."""
import numpy as np
import pytest

from dynaplex import collect_samples
from dynaplex.modelling import assert_featurizer_for_mdp

from featurizers import VehicleFeaturizer
from featurizers_tokens import TokenFeaturizer
from mdp import MyopicPolicy, VehicleRoutingMDP


def _mdp():
    return VehicleRoutingMDP(grid_size=6, n_vehicles=3, capacity=3, max_orders=12)


_KW = dict(n=40, m=4, h=8, controllers=1, workers=2, slots=8, seed=0)


def test_rehearsal_and_spec():
    spec = assert_featurizer_for_mdp(TokenFeaturizer, _mdp())
    assert list(spec) == ["orders", "orders_count", "vehicles", "depot"]
    assert spec["orders"].sample_shape == (12, 2)
    assert spec["vehicles"].sample_shape == (3, 4)


def test_tokens_match_flat_run():
    # the count-map run is the reference: same seeds => same states, and its
    # order-grid block tells us exactly which order tokens each row must hold
    mdp = _mdp()
    flat_run = collect_samples(mdp, MyopicPolicy(mdp), features=VehicleFeaturizer, **_KW)
    tok_run = collect_samples(mdp, MyopicPolicy(mdp), features=TokenFeaturizer, **_KW)

    # features never steer gen-0 collection => labels/masks byte-identical
    assert tok_run.y.tobytes() == flat_run.y.tobytes()
    assert tok_run.mask.tobytes() == flat_run.mask.tobytes()
    (xflat,) = flat_run.features.values()

    # order tokens mirror the order grid: true counts, right coords, clean tails
    g = mdp.grid_size
    orders, counts = tok_run.features["orders"], tok_run.features["orders_count"]
    order_grid = xflat[:, g * g:2 * g * g].reshape(-1, g, g)
    for r in range(len(tok_run)):
        rr, cc = np.nonzero(order_grid[r])
        assert counts[r] == len(rr)
        assert np.allclose(orders[r, :len(rr), 0] * (g - 1), rr)
        assert np.allclose(orders[r, :len(rr), 1] * (g - 1), cc)
        assert not orders[r, len(rr):].any()

    # exactly one focal vehicle per decision state
    assert (tok_run.features["vehicles"][:, :, 3].sum(axis=1) == 1.0).all()


# ---- GridTokenFeaturizer + pointer head (the grid-token variant) -----------

from featurizers_tokens import GridTokenFeaturizer  # noqa: E402


def test_grid_tokens_spec_is_synthesized():
    # all-fixed fields => spec() is generated (sizing-by-counting on the probe)
    assert GridTokenFeaturizer.spec.__module__.startswith("dynaplex._synth.")
    spec = assert_featurizer_for_mdp(GridTokenFeaturizer, _mdp())
    assert [(k, ts.sample_shape) for k, ts in spec.items()] == \
        [("cells", (36, 3)), ("vehicles", (3, 4)), ("depot", (1, 2))]


def test_grid_tokens_row_major_order_matches_actions():
    """THE load-bearing convention: cell token i must describe cell-action i."""
    mdp = _mdp()
    flat_run = collect_samples(mdp, MyopicPolicy(mdp), features=VehicleFeaturizer, **_KW)
    grid_run = collect_samples(mdp, MyopicPolicy(mdp), features=GridTokenFeaturizer, **_KW)
    assert grid_run.y.tobytes() == flat_run.y.tobytes()
    assert grid_run.mask.tobytes() == flat_run.mask.tobytes()

    g = mdp.grid_size
    cells = grid_run.features["cells"]
    # coordinates of token i decode to (i // g, i % g) — in every row
    rows = np.round(cells[:, :, 0] * (g - 1)).astype(int)
    cols = np.round(cells[:, :, 1] * (g - 1)).astype(int)
    assert (rows * g + cols == np.arange(g * g)[None, :]).all()
    # order-presence feature equals the flat run's order grid, token-for-cell
    (xflat,) = flat_run.features.values()
    assert np.array_equal(cells[:, :, 2], xflat[:, g * g:2 * g * g])


def test_pointer_net_scores_cells_in_action_order():
    torch = pytest.importorskip("torch")
    from networks import attention_pointer_net
    from dynaplex.modelling import feature_spec

    mdp = _mdp()
    spec = feature_spec(GridTokenFeaturizer, mdp)
    net = attention_pointer_net(spec, mdp.num_actions, d_model=32, n_layers=1, n_heads=2)
    B, g = 4, mdp.grid_size
    batch = {"cells": torch.rand(B, g * g, 3), "vehicles": torch.rand(B, 3, 4),
             "depot": torch.rand(B, 1, 2)}
    out = net(batch)
    assert out.shape == (B, g * g) and torch.isfinite(out).all()
    out.sum().backward()
    # wrong action count (e.g. a DIRECTIONAL mdp) is rejected loudly
    with pytest.raises(AssertionError, match="CELL"):
        attention_pointer_net(spec, 5)
