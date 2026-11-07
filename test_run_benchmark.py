"""Tests for the architecture benchmark: the statistics, the validity gates and
the harness wiring.

The statistics are the part worth testing hardest. A wrong p-value is invisible —
it produces a plausible number, the report renders, and the conclusion is wrong
in a way no shape check or smoke test would catch. So every estimator here is
checked against a case whose answer is known by hand or by symmetry, and the
gates are checked by INJECTING the failure they exist to catch.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

import benchmark_config as bc
from benchmark_stats import (
    bootstrap_diff_ci,
    cohens_d,
    holm,
    iqm,
    permutation_test,
    probability_of_improvement,
)


# ---------------------------------------------------------------------------
# permutation test
# ---------------------------------------------------------------------------

def test_permutation_test_matches_a_hand_computed_case():
    """3 vs 3, completely separated. C(6,3) = 20 splits; the observed labelling
    and its mirror are the only two with |diff| that extreme, so p = 2/20."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 11.0, 12.0])
    diff, p, exact = permutation_test(a, b)
    assert exact
    assert diff == pytest.approx(-9.0)
    assert p == pytest.approx(2 / 20)


def test_permutation_test_floor_is_two_over_n_choose_k():
    """The smallest attainable exact p. This is the whole reason the
    pre-registration puts the floor at K=5: at K=4 it is 2/70 = 0.029, which
    cannot survive a Holm correction over two contrasts."""
    for k, expected in ((4, 2 / math.comb(8, 4)), (5, 2 / math.comb(10, 5))):
        a = np.arange(k, dtype=float)
        b = np.arange(k, dtype=float) + 1000.0
        _, p, exact = permutation_test(a, b)
        assert exact
        assert p == pytest.approx(expected)


def test_permutation_test_is_symmetric_and_identical_groups_give_p_one():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    diff, p, _ = permutation_test(a, b)
    assert diff == pytest.approx(0.0)
    assert p == pytest.approx(1.0)   # every split is at least as extreme as zero

    x = np.array([1.0, 2.0, 5.0])
    y = np.array([7.0, 9.0, 11.0])
    d1, p1, _ = permutation_test(x, y)
    d2, p2, _ = permutation_test(y, x)
    assert d1 == pytest.approx(-d2)
    assert p1 == pytest.approx(p2)


def test_permutation_test_falls_back_to_monte_carlo_and_never_returns_zero():
    a = np.arange(12, dtype=float)
    b = np.arange(12, dtype=float) + 100.0
    _, p, exact = permutation_test(a, b, max_exact=100, resamples=2000)
    assert not exact
    assert p > 0.0        # the add-one correction; a Monte-Carlo 0 is not a p-value
    assert p < 0.01


# ---------------------------------------------------------------------------
# effect size, IQM, probability of improvement
# ---------------------------------------------------------------------------

def test_iqm_drops_the_tails():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0])
    assert iqm(x) == pytest.approx(np.mean([2.0, 3.0, 4.0, 5.0]))
    assert x.mean() > 14.0          # the outlier owns the mean
    assert iqm(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)   # n<4 -> mean


def test_bootstrap_ci_brackets_the_point_estimate_and_excludes_zero_when_separated():
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98])
    b = np.array([5.0, 5.1, 4.9, 5.05, 4.95, 5.0, 5.02, 4.98])
    point, lo, hi = bootstrap_diff_ci(a, b, resamples=2000, rng_seed=1)
    assert point == pytest.approx(a.mean() - b.mean())
    assert lo <= point <= hi
    assert hi < 0.0                  # a is clearly lower-cost


def test_bootstrap_ci_includes_zero_for_identical_distributions():
    """Two arms drawn from the same values must not produce a CI that excludes
    zero. Built as the same multiset in a different order so the null is exact
    rather than a lucky draw."""
    a = np.array([2.9, 3.1, 2.8, 3.2, 3.0, 2.95, 3.05, 3.15])
    b = a[::-1].copy()
    point, lo, hi = bootstrap_diff_ci(a, b, resamples=4000, rng_seed=2)
    assert point == pytest.approx(0.0)
    assert lo < 0.0 < hi


def test_probability_of_improvement_is_lower_is_better():
    a = np.array([1.0, 2.0, 3.0])          # cheaper => better
    b = np.array([4.0, 5.0, 6.0])
    point, lo, hi = probability_of_improvement(a, b, resamples=1000, rng_seed=3)
    assert point == pytest.approx(1.0)
    assert lo <= point <= hi

    point, _, _ = probability_of_improvement(b, a, resamples=1000, rng_seed=3)
    assert point == pytest.approx(0.0)


def test_probability_of_improvement_counts_ties_as_half():
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 3.0])
    # pairs: (1,2) win, (1,3) win, (2,2) tie, (2,3) win -> (3 + 0.5) / 4
    point, _, _ = probability_of_improvement(a, b, resamples=200, rng_seed=0)
    assert point == pytest.approx(3.5 / 4)


def test_cohens_d_sign_and_zero_variance():
    a = np.array([1.0, 1.0, 1.0, 1.0])
    b = np.array([2.0, 2.0, 2.0, 2.0])
    assert cohens_d(a, b) == float("-inf")   # separated with no within-group spread
    assert cohens_d(b, a) == float("inf")    # and the sign must flip
    assert cohens_d(a, a) == 0.0


# ---------------------------------------------------------------------------
# multiplicity
# ---------------------------------------------------------------------------

def test_holm_adjustment():
    adj = holm({"a": 0.01, "b": 0.04})
    assert adj["a"] == pytest.approx(0.02)    # smallest x m
    assert adj["b"] == pytest.approx(0.04)    # next x (m-1)


def test_holm_is_monotone():
    """The step-down can produce a raw-order inversion; the adjusted values must
    still be non-decreasing in the raw p, or a larger p could 'pass' where a
    smaller one failed."""
    adj = holm({"a": 0.03, "b": 0.031, "c": 0.9})
    assert adj["a"] <= adj["b"] <= adj["c"]
    assert all(v <= 1.0 for v in adj.values())


def test_holm_preserves_key_order():
    keys = ["z", "a", "m"]
    adj = holm({k: 0.01 for k in keys})
    assert list(adj) == keys


# ---------------------------------------------------------------------------
# config integrity
# ---------------------------------------------------------------------------

def test_every_arm_holds_the_trunk_knobs_equal():
    """The whole design rests on 'only the triple varies'. A stray d_model in one
    arm would make the comparison a capacity comparison instead."""
    for arm in bc.ALL_ARMS.values():
        for k, v in bc.TRUNK.items():
            assert arm.net_kwargs[k] == v, f"{arm.name} overrides {k}"


def test_budget_tags_are_distinct_so_workdirs_cannot_collide():
    """DCL resumes when `agent_gen1` exists at the workdir, so two budgets sharing
    a tag would make the second silently reload the first's agent and report it
    under its own recipe. `default` and `long` share n/m/h and differ only in the
    training recipe -- exactly the case an n/m/h-derived tag cannot distinguish."""
    tags = [b.tag for b in bc.BUDGETS.values()]
    assert len(tags) == len(set(tags)), tags
    assert bc.BUDGETS["default"].collect == bc.BUDGETS["long"].collect
    assert bc.BUDGETS["default"].train != bc.BUDGETS["long"].train
    assert bc.BUDGETS["default"].tag != bc.BUDGETS["long"].tag


def test_arm_factories_are_importable_dotted_paths():
    """dp.Net stores a dotted path, not a live module, so a cold node can rebuild
    the agent. A typo here surfaces only after the collection has been paid for."""
    import importlib
    for arm in bc.ALL_ARMS.values():
        mod_name, _, attr = arm.factory.rpartition(".")
        mod = importlib.import_module(mod_name)
        assert callable(getattr(mod, attr)), arm.factory


def test_pointer_arm_is_cell_and_the_others_are_directional():
    from mdp import ActionEncoding
    assert bc.ARMS["pointer"].encoding is ActionEncoding.CELL
    assert bc.ARMS["sat"].encoding is ActionEncoding.DIRECTIONAL
    assert bc.ARMS["tokens"].encoding is ActionEncoding.DIRECTIONAL


# ---------------------------------------------------------------------------
# the network-side feature ablation
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
dpm = pytest.importorskip("dynaplex.modelling")
if not hasattr(dpm, "VariableObjectWriter"):
    pytest.skip("needs dynaplex >= 1.12.2 (Test PyPI)", allow_module_level=True)


def _sat_batch(batch: int = 4, num_actions: int = 5, max_orders: int = 7):
    from dynaplex.modelling import Dtype, TensorSpec
    spec = {
        "summary": TensorSpec(Dtype.float32, (1, 4)),
        "depot": TensorSpec(Dtype.float32, (1, 2)),
        "vehicles": TensorSpec(Dtype.float32, (3, 4)),
        "orders": TensorSpec(Dtype.float32, (max_orders, 2)),
        "orders_count": TensorSpec(Dtype.int64, ()),
        "moves": TensorSpec(Dtype.float32, (num_actions, 6)),
    }
    g = torch.Generator().manual_seed(7)
    b = {
        "summary": torch.rand(batch, 1, 4, generator=g),
        "depot": torch.rand(batch, 1, 2, generator=g),
        "vehicles": torch.rand(batch, 3, 4, generator=g),
        "orders": torch.rand(batch, max_orders, 2, generator=g),
        "orders_count": torch.full((batch,), 3, dtype=torch.long),
        "moves": torch.rand(batch, num_actions, 6, generator=g),
    }
    return spec, b


def test_geometry_action_features_ignore_the_hand_crafted_columns():
    """`action_features="geometry"` must be EXACTLY equivalent to a featurizer
    that never wrote has_order / nearest_order_dist — that equivalence is what
    lets the ablation share the full net's sample set instead of needing its own
    collection (contra MODULE_MAP.md:301-303)."""
    from networks_sat import sat_agile_net
    spec, batch = _sat_batch()
    torch.manual_seed(0)
    net = sat_agile_net(spec, 5, action_features="geometry").eval()
    with torch.no_grad():
        before = net(batch)
        scrambled = dict(batch)
        moves = batch["moves"].clone()
        moves[..., 4:] = torch.randn_like(moves[..., 4:]) * 100.0
        scrambled["moves"] = moves
        after = net(scrambled)
    assert torch.equal(before, after)


def test_full_action_features_do_depend_on_those_columns():
    """The control for the test above: with the default the same perturbation
    must move the output, or the ablation would be vacuous."""
    from networks_sat import sat_agile_net
    spec, batch = _sat_batch()
    torch.manual_seed(0)
    net = sat_agile_net(spec, 5).eval()
    with torch.no_grad():
        before = net(batch)
        scrambled = dict(batch)
        moves = batch["moves"].clone()
        moves[..., 4:] = torch.randn_like(moves[..., 4:]) * 100.0
        scrambled["moves"] = moves
        after = net(scrambled)
    assert not torch.equal(before, after)


def test_geometry_columns_receive_no_gradient():
    """The other half of the equivalence: zeroed inputs must not train their
    weights either, or the ablated net would still have capacity tied to those
    features."""
    from networks_sat import sat_agile_net, _ACTION_GEOMETRY_COLS
    spec, batch = _sat_batch()
    torch.manual_seed(0)
    net = sat_agile_net(spec, 5, action_features="geometry")
    net(batch).sum().backward()
    grad = net.proj["moves"].weight.grad
    assert grad is not None
    assert torch.count_nonzero(grad[:, _ACTION_GEOMETRY_COLS:]) == 0
    assert torch.count_nonzero(grad[:, :_ACTION_GEOMETRY_COLS]) > 0


def test_action_features_default_is_unchanged_behaviour():
    """Existing agents must keep loading: the knob adds no parameters and no
    state_dict keys."""
    from networks_sat import sat_agile_net
    spec, _ = _sat_batch()
    torch.manual_seed(0)
    a = sat_agile_net(spec, 5)
    torch.manual_seed(0)
    b = sat_agile_net(spec, 5, action_features="geometry")
    assert set(a.state_dict()) == set(b.state_dict())


# ---------------------------------------------------------------------------
# validity gates, by injecting the failure each one exists to catch
# ---------------------------------------------------------------------------

def _row(arm: str, seed: int, mean: float, *, myopic_returns=(3.1, 3.0, 3.2),
         degeneracy=0.3, stopped_early=True, y_sha="aaa", mask_sha="bbb"):
    return {
        "schema": "benchmark-result-v1", "budget": "quick", "budget_tag": "n1m1h1",
        "collect": {}, "train": {}, "comparer": {}, "dcl_seed": 0,
        "comparer_seed": 42, "workers": 4, "platform": "test", "torch": "x",
        "numpy": "y", "arm": arm, "seed": seed, "encoding": "DIRECTIONAL",
        "featurizer": "f", "network": "n", "net_kwargs": {},
        "sample_digest": {"n": 10, "num_actions": 5, "y_sha": y_sha,
                          "mask_sha": mask_sha},
        "mean": mean, "error": 0.02, "delta_mean": mean - 3.1,
        "delta_error": 0.03, "returns": [mean] * 3,
        "degeneracy_share": degeneracy, "degeneracy_hist": [1, 0, 0, 0, 0],
        "stopped_early": stopped_early, "epochs_run": 5, "epochs_max": 40,
        "best_val_loss": 0.5, "val_regret": 0.1, "train_seconds": 1.0,
        "num_parameters": 100,
        "anchors": {"myopic": {"mean": 3.1, "error": 0.03,
                               "returns": list(myopic_returns)},
                    "random": {"mean": 52.0, "error": 0.4,
                               "returns": [52.0, 52.0, 52.0]}},
    }


def _dataset(**kw):
    rows = []
    for seed in range(5):
        rows.append(_row("sat", seed, 2.5 + 0.05 * seed, **kw))
        rows.append(_row("tokens", seed, 3.4 + 0.05 * seed, **kw))
        rows.append(_row("pointer", seed, 3.6 + 0.05 * seed, **kw))
    return rows


def _gates(rows, anchors=bc.REFERENCE_ANCHORS):
    import analyze_benchmark as ab
    return {g["name"].split(" (")[0]: g
            for g in ab.check_gates(rows, ab.by_arm(rows), anchors)}


def test_gates_all_pass_on_a_clean_dataset():
    g = _gates(_dataset())
    assert g["CRN splice"]["ok"] is True
    assert g["anchor reproduction"]["ok"] is True
    assert g["degeneracy"]["ok"] is True
    assert g["seed count"]["ok"] is True


def test_crn_splice_gate_catches_a_mismatched_comparer_call():
    """The failure this exists for: someone evaluates one arm in a separate call
    with a different seed, and the per-trajectory returns silently stop being
    paired. Nothing else in the pipeline would notice."""
    rows = _dataset()
    rows[0]["anchors"]["myopic"]["returns"] = [9.9, 9.9, 9.9]
    g = _gates(rows)
    assert g["CRN splice"]["ok"] is False
    detail = g["CRN splice"]["detail"]
    # the ONE odd record must be named, not the fourteen that agree with each other
    assert "sat_s0" in detail
    assert "tokens_s1" not in detail


def test_anchor_gate_catches_a_drifted_comparer_config():
    rows = _dataset()
    for r in rows:
        r["anchors"]["myopic"]["mean"] = 4.4
    g = _gates(rows)
    assert g["anchor reproduction"]["ok"] is False


def test_anchor_gate_is_not_checkable_when_the_budget_has_no_reference():
    """The quick budget evaluates at horizon 200, where random reads ~40 against
    the reference table 52.8 -- correct, not a failure. The gate must report
    not-checkable rather than invent a range and fail every quick run."""
    rows = _dataset()
    for r in rows:
        r["anchors"]["random"]["mean"] = 40.0
    assert _gates(rows, anchors=None)["anchor reproduction"]["ok"] is None
    assert _gates(rows)["anchor reproduction"]["ok"] is False   # but it checks when it can
    assert bc.BUDGETS["quick"].anchors is None
    assert bc.BUDGETS["default"].anchors == bc.REFERENCE_ANCHORS


def test_degeneracy_gate_catches_a_collapsed_arm():
    rows = _dataset()
    rows[1]["degeneracy_share"] = 0.98
    g = _gates(rows)
    assert g["degeneracy"]["ok"] is False


def test_data_matching_gate_is_advisory_and_notices_differing_labels():
    rows = [r for r in _dataset() if r["arm"] in ("sat", "tokens")]
    for r in rows:
        if r["arm"] == "tokens":
            r["sample_digest"]["y_sha"] = "different"
    g = _gates(rows)
    gate = g["data matching"]
    assert gate["ok"] is False
    assert gate["advisory"] is True          # informs the write-up, does not block


def test_balance_gate_catches_unequal_or_too_few_seeds():
    rows = [r for r in _dataset() if not (r["arm"] == "sat" and r["seed"] == 4)]
    assert _gates(rows)["seed count"]["ok"] is False
    rows = [r for r in _dataset() if r["seed"] < 3]
    assert _gates(rows)["seed count"]["ok"] is False


# ---------------------------------------------------------------------------
# the decision rule and the report
# ---------------------------------------------------------------------------

def test_decision_rule_passes_on_a_clearly_separated_dataset():
    import analyze_benchmark as ab
    rows = _dataset()
    contrasts, _ = ab.analyse(ab.by_arm(rows))
    for c in contrasts.values():
        assert c["passes"] is True
        assert c["separated"] is True
        assert c["mean_diff"] < 0


def test_decision_rule_fails_when_the_arms_overlap():
    """The rule must be able to say 'not established'. If it cannot, it is not a
    test, it is a formality."""
    import analyze_benchmark as ab
    rows = []
    for seed in range(5):
        rows.append(_row("sat", seed, 3.0 + 0.3 * (seed % 2)))
        rows.append(_row("tokens", seed, 3.1 + 0.3 * ((seed + 1) % 2)))
        rows.append(_row("pointer", seed, 3.05 + 0.3 * (seed % 2)))
    contrasts, _ = ab.analyse(ab.by_arm(rows))
    assert not all(c["passes"] for c in contrasts.values())


def test_report_withholds_a_verdict_when_a_blocking_gate_fails():
    import analyze_benchmark as ab
    rows = _dataset()
    rows[0]["anchors"]["myopic"]["returns"] = [9.9, 9.9, 9.9]   # break gate 1
    arms = ab.by_arm(rows)
    gates = ab.check_gates(rows, arms, bc.REFERENCE_ANCHORS)
    contrasts, costs = ab.analyse(arms)
    report = ab.render("quick", rows, arms, gates, contrasts, costs)
    assert "**No verdict.**" in report
    assert "not evidence until the gates pass" in report


def test_report_renders_every_seed_not_just_summaries():
    import analyze_benchmark as ab
    rows = _dataset()
    arms = ab.by_arm(rows)
    report = ab.render("quick", rows, arms,
                       ab.check_gates(rows, arms, bc.REFERENCE_ANCHORS),
                       *ab.analyse(arms))
    for r in rows:
        assert f"{r['mean']:.3f}" in report
    assert "What this does not show" in report        # limitations always shipped


def test_results_round_trip_through_json(tmp_path, monkeypatch):
    """The harness persists results and skips arm-seeds that already have one;
    that is what makes --seeds extendable. Check the loader sees what the writer
    wrote."""
    import analyze_benchmark as ab
    monkeypatch.setattr(ab, "RESULTS_ROOT", str(tmp_path))
    d = tmp_path / "n1m1h1"
    d.mkdir()
    rows = _dataset()
    for r in rows:
        (d / f"{r['arm']}_seed{r['seed']}.json").write_text(json.dumps(r),
                                                            encoding="utf-8")
    (d / "summary.json").write_text("{}", encoding="utf-8")   # must be ignored
    loaded = ab.load("n1m1h1")
    assert len(loaded) == len(rows)
    assert {r["arm"] for r in loaded} == {"sat", "tokens", "pointer"}


# ---------------------------------------------------------------------------
# the degeneracy probe's own state distribution
# ---------------------------------------------------------------------------

def _probe_states(encoding, n_states=60, seed=5):
    """Re-walk exactly what `degeneracy` walks, returning the states."""
    import copy
    import numpy as np
    from sat_compat import make_context
    from dynaplex.modelling.features import ValidityWriter
    from mdp import VehicleRoutingMDP

    mdp = VehicleRoutingMDP(action_encoding=encoding)
    ctx = make_context(mdp, seed, eval=True)
    rng = np.random.default_rng(seed)
    state = mdp.get_initial_state(ctx)
    m = np.zeros((1, mdp.num_actions), dtype=np.bool_)
    vw = ValidityWriter(m, 0)
    states, took_invalid = [], 0
    while len(states) < n_states:
        states.append(copy.deepcopy(state))
        vw.reset(0)
        m[0, :] = False
        mdp.write_action_validity(state, vw)
        adm = np.flatnonzero(m[0])
        a = int(rng.choice(adm))
        if not m[0, a]:
            took_invalid += 1
        mdp.modify_state_with_action(state, ctx, a)
        while state.category.name == "AWAIT_EVENT":
            mdp.modify_state_with_event(state, ctx)
    return mdp, states, took_invalid


@pytest.mark.parametrize("encoding_name", ["CELL", "DIRECTIONAL"])
def test_degeneracy_probe_never_takes_an_inadmissible_action(encoding_name):
    """The original probe stepped with a hardcoded action 0. Under CELL that is
    cell (0,0) (mdp.py:203), inadmissible unless the vehicle is beside the depot --
    it was measured taking an invalid action in 5 of 60 probe states."""
    from mdp import ActionEncoding
    _, _, took_invalid = _probe_states(getattr(ActionEncoding, encoding_name))
    assert took_invalid == 0


@pytest.mark.parametrize("encoding_name", ["CELL", "DIRECTIONAL"])
def test_degeneracy_probe_visits_a_spread_of_states(encoding_name):
    """The real defect: with a hardcoded action 0 the fleet never moved and 60
    probe states covered ~5 distinct focal positions, so 'dominance' described the
    probe, not the network. A random ADMISSIBLE walk must actually move around."""
    from mdp import ActionEncoding
    _, states, _ = _probe_states(getattr(ActionEncoding, encoding_name))
    focal = {(s.vehicles[s.current_vehicle].row, s.vehicles[s.current_vehicle].col)
             for s in states}
    assert len(focal) >= 20, f"only {len(focal)} distinct focal positions in 60 states"


def test_degeneracy_probe_states_are_distinct_objects():
    """The MDP mutates State in place; without the deepcopy every entry is an
    alias of the final state and every agent scores 100% degenerate."""
    from mdp import ActionEncoding
    _, states, _ = _probe_states(ActionEncoding.DIRECTIONAL)
    assert len({id(s) for s in states}) == len(states)
    assert len({s.n_orders for s in states}) > 1


def test_budget_truncation_gate_blocks_when_most_seeds_hit_the_cap():
    """Promoted from advisory on 2026-08-22. It fired correctly at the default
    budget (7/8, 8/8, 7/8 capped) and the verdict was issued anyway; a 200-epoch
    probe then showed that verdict was an artifact of the cap."""
    rows = _dataset(stopped_early=False)
    g = _gates(rows)["budget truncation"]
    assert g["ok"] is False
    assert not g.get("advisory")


def test_budget_truncation_gate_tolerates_a_minority_of_capped_seeds():
    """One unlucky run hitting the cap is noise, not a truncated experiment."""
    rows = _dataset()
    rows[0]["stopped_early"] = False          # 1 of 5 sat seeds
    g = _gates(rows)["budget truncation"]
    assert g["ok"] is True
    assert g["advisory"] is True


def test_report_withholds_a_verdict_when_arms_are_truncated():
    import analyze_benchmark as ab
    rows = _dataset(stopped_early=False)
    arms = ab.by_arm(rows)
    gates = ab.check_gates(rows, arms, bc.REFERENCE_ANCHORS)
    report = ab.render("long", rows, arms, gates, *ab.analyse(arms))
    assert "**No verdict.**" in report


# ---------------------------------------------------------------------------
# censoring-aware statistics (epochs-to-threshold)
# ---------------------------------------------------------------------------

def test_rank_win_rate_handles_censored_values():
    """A seed that never reaches tau is +inf. It must order worse than every
    observed value, and two censored seeds must TIE rather than produce nan --
    `inf < inf` is False, `inf == inf` is True."""
    from benchmark_stats import rank_win_rate
    inf = float("inf")
    assert rank_win_rate(np.array([10.0]), np.array([inf])) == 1.0
    assert rank_win_rate(np.array([inf]), np.array([10.0])) == 0.0
    assert rank_win_rate(np.array([inf]), np.array([inf])) == 0.5
    assert not np.isnan(rank_win_rate(np.array([inf, 20.0]), np.array([inf, inf])))


def test_rank_win_rate_counts_ties_as_half():
    from benchmark_stats import rank_win_rate
    assert rank_win_rate(np.array([10.0, 20.0]),
                         np.array([20.0, 30.0])) == pytest.approx(3.5 / 4)


def test_permutation_test_ranks_reaches_the_exact_floor_on_separation():
    """Perfectly separated 5-vs-5 gives win rate 1.0 and the exact floor
    2/C(10,5) = 0.0079 -- the same floor that makes K=5 the design minimum."""
    from benchmark_stats import permutation_test_ranks
    a = np.array([10.0, 20.0, 20.0, 40.0, 40.0])
    b = np.array([80.0, 80.0, 160.0, 160.0, float("inf")])
    obs, p, exact = permutation_test_ranks(a, b)
    assert exact and obs == 1.0
    assert p == pytest.approx(2 / math.comb(10, 5))


def test_permutation_test_ranks_is_null_on_identical_groups():
    from benchmark_stats import permutation_test_ranks
    a = np.array([10.0, 20.0, 30.0, 40.0])
    obs, p, _ = permutation_test_ranks(a, a.copy())
    assert obs == pytest.approx(0.5)
    assert p == pytest.approx(1.0)


def test_censored_summary_never_collapses_censoring_into_a_number():
    """'5 of 8 seeds reached tau, median 40 among those' is the honest statement;
    a bare mean would silently drop or invent the censored ones."""
    from benchmark_stats import censored_summary
    inf = float("inf")
    s = censored_summary(np.array([20.0, 40.0, inf, inf]))
    assert s == {"n": 4, "reached": 2, "median": 30.0, "min": 20.0, "max": 40.0}
    none = censored_summary(np.array([inf, inf]))
    assert none["reached"] == 0 and none["median"] is None
