"""The snapshot trainer must be bit-for-bit `dynaplex.nn.train_network`.

`learning_curve.train_with_snapshots` reimplements someone else's training recipe
so it can snapshot mid-run. That is a second copy of a moving target, and the
failure mode is silent: a slightly different shuffle or a missing `masked_fill`
still trains, still converges, and produces a learning curve that is subtly not
the curve the real pipeline would have. So the equivalence is asserted BITWISE
against `train_network` itself, for every arm, rather than approximately.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
dpm = pytest.importorskip("dynaplex.modelling")
if not hasattr(dpm, "VariableObjectWriter"):
    pytest.skip("needs dynaplex >= 1.12.2 (Test PyPI)", allow_module_level=True)

import dynaplex as dp
from dynaplex.dcl import collect_samples
from dynaplex.nn import train_network

import benchmark_config as bc
from learning_curve import epochs_to_threshold, train_with_snapshots
from mdp import VehicleRoutingMDP


def _samples(arm, n=120, m=8, h=6, seed=0):
    mdp = VehicleRoutingMDP(action_encoding=arm.encoding)
    return collect_samples(mdp, dp.RandomPolicy(mdp), features=arm.featurizer,
                           n=n, m=m, h=h, slots=16, seed=seed, workers=1)


# `Net.build` draws weight init from AMBIENT global torch RNG, because
# `train_network` constructs the module BEFORE calling `torch.manual_seed`
# (dynaplex/nn.py:404 vs :408) -- the trap documented at MODULE_MAP.md:236-242 and
# worked around in run_benchmark.train_arm_seed. So comparing two training calls
# requires re-seeding the GLOBAL rng before each: without it the second call gets a
# different initialisation and the comparison measures init luck, not equivalence.
INIT_SEED = 12345


def _weights(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _assert_bitwise(a: dict, b: dict, label: str):
    assert set(a) == set(b), f"{label}: different parameter sets"
    for k in a:
        assert torch.equal(a[k], b[k]), (
            f"{label}: tensor {k!r} differs (max abs diff "
            f"{(a[k].float() - b[k].float()).abs().max().item():.3e})")


@pytest.mark.parametrize("arm_name", ["sat", "tokens", "pointer"])
def test_snapshot_matches_train_network_bitwise(arm_name):
    """THE load-bearing test. Each rung must equal a dedicated
    `train_network(epochs=E, patience=0)` run at the same seed, exactly."""
    arm = bc.ALL_ARMS[arm_name]
    ss = _samples(arm)
    ladder = [1, 2, 4]

    torch.manual_seed(INIT_SEED)
    points, history = train_with_snapshots(
        ss, network=arm.net(), ladder=ladder, features=arm.featurizer,
        lr=1e-4, seed=3, val_fraction=0.2)

    assert [p.epochs for p in points] == ladder
    assert len(history.train_loss) == max(ladder)

    for p in points:
        torch.manual_seed(INIT_SEED)
        ref = train_network(ss, network=arm.net(), features=arm.featurizer,
                            epochs=p.epochs, patience=0, lr=1e-4, seed=3,
                            val_fraction=0.2)
        _assert_bitwise(_weights(p.agent.module), _weights(ref.module),
                        f"{arm_name} rung {p.epochs}")


@pytest.mark.parametrize("arm_name", ["sat", "tokens"])
def test_snapshot_metrics_match_train_network(arm_name):
    arm = bc.ALL_ARMS[arm_name]
    ss = _samples(arm)
    torch.manual_seed(INIT_SEED)
    points, _ = train_with_snapshots(ss, network=arm.net(), ladder=[1, 3],
                                     features=arm.featurizer, lr=1e-4, seed=1,
                                     val_fraction=0.2)
    for p in points:
        torch.manual_seed(INIT_SEED)
        ref = train_network(ss, network=arm.net(), features=arm.featurizer,
                            epochs=p.epochs, patience=0, lr=1e-4, seed=1,
                            val_fraction=0.2)
        assert p.best_val_loss == pytest.approx(ref.info["best_val_loss"])
        assert p.val_regret == pytest.approx(ref.info["val_regret"], abs=1e-9)


def test_rungs_are_independent_objects_not_aliases():
    """A snapshot must freeze the weights at its rung. Storing a reference to the
    live `best_state` would make every rung show the FINAL policy -- the same
    aliasing trap the MDP's in-place State mutation sets (MODULE_MAP.md:255-261),
    and it would make every arm look like it converged instantly."""
    arm = bc.ALL_ARMS["sat"]
    ss = _samples(arm)
    points, _ = train_with_snapshots(ss, network=arm.net(), ladder=[1, 6],
                                     features=arm.featurizer, lr=1e-3, seed=0,
                                     val_fraction=0.2)
    early, late = _weights(points[0].agent.module), _weights(points[1].agent.module)
    assert points[0].agent.module is not points[1].agent.module
    differs = any(not torch.equal(early[k], late[k]) for k in early)
    assert differs, "rung 1 and rung 6 have identical weights; snapshots are aliased"


def test_history_is_recorded_every_epoch():
    arm = bc.ALL_ARMS["tokens"]
    ss = _samples(arm)
    _, h = train_with_snapshots(ss, network=arm.net(), ladder=[5],
                                features=arm.featurizer, seed=0, val_fraction=0.2)
    assert len(h.train_loss) == len(h.val_loss) == len(h.val_regret) == 5
    assert all(np.isfinite(x) for x in h.train_loss)
    assert all(np.isfinite(x) for x in h.val_loss)
    # best_epoch_at is non-decreasing by construction
    assert h.best_epoch_at == sorted(h.best_epoch_at)


# ---------------------------------------------------------------------------
# the threshold-crossing endpoint
# ---------------------------------------------------------------------------

def test_epochs_to_threshold_takes_the_first_crossing():
    assert epochs_to_threshold([5, 10, 20], [9.0, 2.9, 2.5], 3.0) == 10
    assert epochs_to_threshold([5, 10, 20], [2.9, 2.8, 2.5], 3.0) == 5


def test_epochs_to_threshold_censors_rather_than_guessing():
    """An arm that never reaches tau within the ladder returns None. It must not
    silently become max(ladder), which would understate how far off it was, nor be
    dropped, which would bias the comparison toward whichever arm fails more."""
    assert epochs_to_threshold([5, 10, 20], [9.0, 8.0, 7.0], 3.0) is None


def test_epochs_to_threshold_does_not_interpolate():
    """Only measured rungs may be returned -- 40 and 80 are budgets that were run;
    57 is not."""
    got = epochs_to_threshold([40, 80], [3.5, 2.5], 3.0)
    assert got == 80
