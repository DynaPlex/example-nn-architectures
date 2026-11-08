"""MoveTokenFeaturizer + move_token_net: candidate order, scatter into cells,
and the candidate-interaction switch."""
import numpy as np
import pytest

from dynaplex import collect_samples
from dynaplex.modelling import assert_featurizer_for_mdp

from featurizer_flat import VehicleFeaturizer
from featurizer_moves import MoveTokenFeaturizer
from mdp import MyopicPolicy, VehicleRoutingMDP


def _mdp():
    return VehicleRoutingMDP(grid_size=6, n_vehicles=3, capacity=3, max_orders=12)


_KW = dict(n=40, m=4, h=8, controllers=1, workers=2, slots=8, seed=0)


def test_spec():
    spec = assert_featurizer_for_mdp(MoveTokenFeaturizer, _mdp())
    assert [(k, ts.sample_shape) for k, ts in spec.items()] == [
        ("candidates", (5, 4)), ("orders", (12, 2)), ("orders_count", ()),
        ("vehicles", (3, 4)), ("depot", (1, 2))]


def test_candidates_are_exactly_the_valid_cells():
    mdp = _mdp()
    g = mdp.grid_size
    flat_run = collect_samples(mdp, MyopicPolicy(mdp), features=VehicleFeaturizer, **_KW)
    run = collect_samples(mdp, MyopicPolicy(mdp), features=MoveTokenFeaturizer, **_KW)
    assert run.y.tobytes() == flat_run.y.tobytes()
    assert run.mask.tobytes() == flat_run.mask.tobytes()
    cand = run.features["candidates"]
    rows = np.round(cand[:, :, 0] * (g - 1)).astype(int)
    cols = np.round(cand[:, :, 1] * (g - 1)).astype(int)
    in_grid = cand[:, :, 3] > 0.5
    (xflat,) = flat_run.features.values()
    order_grid = xflat[:, g * g:2 * g * g]
    for i in range(len(run)):
        # the in-grid candidates are exactly the mask's valid cells
        cells = set((rows[i, in_grid[i]] * g + cols[i, in_grid[i]]).tolist())
        assert cells == set(np.flatnonzero(run.mask[i]).tolist())
        # stay = candidate 0; up/down/left/right at offsets from it
        r0, c0 = rows[i, 0], cols[i, 0]
        for k, (dr, dc) in enumerate(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))):
            if in_grid[i, k]:
                assert (rows[i, k], cols[i, k]) == (r0 + dr, c0 + dc)
                assert cand[i, k, 2] == order_grid[i, rows[i, k] * g + cols[i, k]]
            else:
                assert (rows[i, k], cols[i, k]) == (r0, c0)
    # exactly one focal vehicle, standing on candidate 0
    veh = run.features["vehicles"]
    assert (veh[:, :, 3].sum(axis=1) == 1.0).all()
    focal = veh[:, :, 3].argmax(axis=1)
    assert np.allclose(veh[np.arange(len(run)), focal, 0], cand[:, 0, 0])
    assert np.allclose(veh[np.arange(len(run)), focal, 1], cand[:, 0, 1])


def _batch(torch, mdp, B=3):
    g = mdp.grid_size
    cand = torch.zeros(B, 5, 4)
    # vehicle at (2, 2): all five candidates in-grid, written like the featurizer
    for k, (dr, dc) in enumerate(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))):
        cand[:, k, 0] = (2 + dr) / (g - 1)
        cand[:, k, 1] = (2 + dc) / (g - 1)
        cand[:, k, 3] = 1.0
    cand[0, 4, 3] = 0.0                       # sample 0: "right" is off-grid
    cand[0, 4, 0:2] = cand[0, 0, 0:2]
    return {"candidates": cand, "orders": torch.rand(B, mdp.max_orders, 2),
            "orders_count": torch.tensor([3, 0, mdp.max_orders]),
            "vehicles": torch.rand(B, mdp.n_vehicles, 4), "depot": torch.rand(B, 1, 2)}


def test_move_token_net_scatters_scores_into_candidate_cells():
    torch = pytest.importorskip("torch")
    from dynaplex.modelling import feature_spec
    from networks import move_token_net

    mdp = _mdp()
    g = mdp.grid_size
    net = move_token_net(feature_spec(MoveTokenFeaturizer, mdp), mdp.num_actions,
                        d_model=16, n_layers=1, n_heads=2)
    batch = _batch(torch, mdp)
    out = net(batch)
    assert out.shape == (3, g * g) and torch.isfinite(out).all()
    out.sum().backward()
    scored = out > -1e8
    cells = {(2 + dr) * g + (2 + dc) for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))}
    assert set(torch.nonzero(scored[1]).flatten().tolist()) == cells
    assert set(torch.nonzero(scored[0]).flatten().tolist()) == cells - {2 * g + 3}
    with pytest.raises(AssertionError, match="square"):
        move_token_net(feature_spec(MoveTokenFeaturizer, mdp), 5)


@pytest.mark.parametrize("training", [False, True])
def test_interact_false_scores_candidates_independently(training):
    """With interact=False, changing one candidate's features leaves the other
    candidates' scores untouched (in eval AND train mode — the encoder's eval
    fast path must honour the mask). With interact=True they move."""
    torch = pytest.importorskip("torch")
    from dynaplex.modelling import feature_spec
    from networks import move_token_net

    mdp = _mdp()
    g = mdp.grid_size
    spec = feature_spec(MoveTokenFeaturizer, mdp)
    batch = _batch(torch, mdp)
    stay = 2 * g + 2
    up = 1 * g + 2
    for interact, expect_change in ((False, False), (True, True)):
        torch.manual_seed(0)
        net = move_token_net(spec, mdp.num_actions, d_model=16, n_layers=1, n_heads=2,
                            interact=interact).train(training)
        torch.manual_seed(1)                                   # same dropout draws
        base = net(batch).detach()
        bumped = {k: v.clone() for k, v in batch.items()}
        bumped["candidates"][:, 1, 2] = 1.0 - bumped["candidates"][:, 1, 2]   # flip "up"'s order flag
        torch.manual_seed(1)
        out = net(bumped).detach()
        assert not torch.allclose(out[:, up], base[:, up])          # "up" itself moves
        changed = not torch.allclose(out[:, stay], base[:, stay], atol=1e-6)
        assert changed == expect_change, (interact, training)
