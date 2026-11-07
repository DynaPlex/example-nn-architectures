"""The frozen experiment definition for `run_benchmark.py` / `analyze_benchmark.py`.

Everything that must be IDENTICAL across arms lives here as data, so the harness
and the analysis read the same numbers and cannot drift apart. Nothing in this
module trains, collects or evaluates anything; importing it is free.

The comparison this parameterises is deliberately narrow: three (featurizer,
network, encoding) triples on one MDP at one budget, many training seeds. See
`PREREGISTRATION.md` for the endpoints, the tests and the decision rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import dynaplex as dp

from featurizers_sat import SATFeaturizer
from featurizers_tokens import GridTokenFeaturizer, TokenFeaturizer
from mdp import ActionEncoding


@dataclass(frozen=True)
class Arm:
    """One (featurizer, network, encoding) triple. Everything else is shared."""
    name: str
    featurizer: type
    factory: str                 # dotted path for dp.Net — never a live module
    encoding: ActionEncoding
    net_kwargs: dict
    blurb: str

    def net(self) -> "dp.Net":
        return dp.Net(self.factory, **self.net_kwargs)


# Transformer size knobs, held equal across all three arms. These are exactly the
# values the existing run scripts already use (train.py's paper budget and
# run_sat.py), so no arm is being re-tuned here.
TRUNK = dict(d_model=64, n_layers=2, n_heads=4)

ARMS: dict[str, Arm] = {
    "sat": Arm(
        name="sat",
        featurizer=SATFeaturizer,
        factory="networks_sat.sat_agile_net",
        encoding=ActionEncoding.DIRECTIONAL,
        net_kwargs=dict(TRUNK),
        blurb="SAT/AGILE: typed context tokens + one token per ACTION, "
              "complete-graph attention over [state ; actions]",
    ),
    "tokens": Arm(
        name="tokens",
        featurizer=TokenFeaturizer,
        factory="networks.attention_net",
        encoding=ActionEncoding.DIRECTIONAL,
        net_kwargs=dict(TRUNK, head="moves5"),
        blurb="order/vehicle/depot tokens, focal-vehicle readout -> 5 move logits",
    ),
    "pointer": Arm(
        name="pointer",
        featurizer=GridTokenFeaturizer,
        factory="networks.attention_pointer_net",
        encoding=ActionEncoding.CELL,
        net_kwargs=dict(TRUNK),
        blurb="all G*G cells as tokens, shared per-cell scoring head (pointer)",
    ),
}

# The optional fourth arm that closes the feature confound. Off by default: it
# widens the question from "is SAT better" to "why", which is `run_sat_ablation.py`'s
# job. Kept here so `--arms sat,tokens,pointer,sat_nofeat` just works.
OPTIONAL_ARMS: dict[str, Arm] = {
    "sat_nofeat": Arm(
        name="sat_nofeat",
        featurizer=SATFeaturizer,
        factory="networks_sat.sat_agile_net",
        encoding=ActionEncoding.DIRECTIONAL,
        net_kwargs=dict(TRUNK, action_features="geometry"),
        blurb="SAT/AGILE with has_order + nearest_order_dist zeroed at the "
              "network input - same sample set, features removed",
    ),
}

ALL_ARMS: dict[str, Arm] = {**ARMS, **OPTIONAL_ARMS}

DEFAULT_ARMS = ("sat", "tokens", "pointer")

# --- budgets ---------------------------------------------------------------
# `default` is the laptop-scale budget every arm is re-run at. It is NOT the
# baselines' published m=2000 / 200-epoch budget (that is ~2.7 h per arm-seed and
# does not fit a seed sweep on 16 cores), so the absolute numbers here will not
# line up with the 3.324 on record — which is the point: those numbers were not
# comparable to each other either. `paper` reproduces the published budget for a
# cluster.

@dataclass(frozen=True)
class Budget:
    name: str
    collect: dict                # -> dp.DCL(**collect)
    train: dict                  # -> dp.DCL(train=...)
    comparer: dict               # -> dp.PolicyComparer(**comparer)
    anchors: dict | None         # expected myopic/random ranges, or None
    tag_override: str | None = None

    @property
    def tag(self) -> str:
        """Workdir/results tag. Carries n/m/h because passing `workdir=` bypasses
        DCL's structural hashing and it resumes from whatever it finds — a --quick
        run followed by a full one at the same path would train on the quick run's
        samples while reporting the full budget (MODULE_MAP.md:243-246).

        n/m/h alone is NOT enough. Two budgets can share a collection and differ
        only in the training recipe, and DCL resumes on `agent_gen1` existing — so
        the second would silently reload the first's 40-epoch agent and report it
        as a 200-epoch result. `tag_override` is how such a budget claims its own
        namespace; `test_budget_tags_are_distinct_so_workdirs_cannot_collide`
        enforces that no two budgets share a tag."""
        if self.tag_override:
            return self.tag_override
        c = self.collect
        return f"n{c['n']}m{c['m']}h{c['h']}"


# The anchor check compares myopic and random against the values on record. Those
# are `warmup_time=100, horizon=500` numbers (myopic 3.096, random 52.8), so the
# check only means something where the comparer config matches. At the quick
# budget the transient dominates -- myopic still lands near 3.1 but random reads
# ~40 at horizon 200 -- so that budget declares the gate not-checkable rather than
# being given a fabricated range.
REFERENCE_ANCHORS = {"myopic": (3.0, 3.25), "random": (48.0, 58.0)}

BUDGETS: dict[str, Budget] = {
    "quick": Budget(
        name="quick",
        collect=dict(n=1000, m=25, h=15),
        train=dict(epochs=3, batch_size=64, lr=1e-4, patience=8),
        comparer=dict(number_of_trajectories=32, warmup_time=50, horizon=200),
        anchors=None,
    ),
    "default": Budget(
        name="default",
        collect=dict(n=10000, m=500, h=100),
        train=dict(epochs=40, batch_size=64, lr=1e-4, patience=8),
        comparer=dict(number_of_trajectories=256, warmup_time=100, horizon=500),
        anchors=dict(REFERENCE_ANCHORS),
    ),
    # Same collection as `default`, five times the training. The probe for "were
    # the baselines simply truncated?" — at the default budget 7/8 `pointer` and
    # 8/8 `tokens` runs hit the 40-epoch cap, so every arm was still improving
    # when it was stopped, and the published baseline config is 200 epochs.
    "long": Budget(
        name="long",
        collect=dict(n=10000, m=500, h=100),
        train=dict(epochs=200, batch_size=64, lr=1e-4, patience=15),
        comparer=dict(number_of_trajectories=256, warmup_time=100, horizon=500),
        anchors=dict(REFERENCE_ANCHORS),
        tag_override="n10000m500h100e200",
    ),
    "paper": Budget(
        name="paper",
        collect=dict(n=10000, m=2000, h=100),
        train=dict(epochs=200, batch_size=64, lr=1e-4, patience=15),
        comparer=dict(number_of_trajectories=256, warmup_time=100, horizon=500),
        anchors=dict(REFERENCE_ANCHORS),
    ),
}

# Seeds for collection and evaluation are FIXED — only the training seed varies.
# Sharing one collection across seeds means the seed sweep measures weight-init
# and shuffle variance on a fixed dataset, not collection variance. That is the
# right population for an architecture comparison, and it is what makes the
# `sat` and `tokens` arms data-matched (validity gate 3).
DCL_SEED = 0
COMPARER_SEED = 42

# Anchor policies, present in every comparer call. `myopic` doubles as the CRN
# splice probe (gate 1) and the reference-value check (gate 2).
#
# NOTE that `random` is NOT comparable across encodings: uniform-over-valid draws
# from the policy RNG differently under CELL and DIRECTIONAL, and it measured
# 40.04 vs 39.13 in the two arms of one smoke run. Only `myopic` may be used to
# pair the two comparer calls.
BENCHMARK = "myopic"
SPLICE_PROBE = "myopic"

# An arm-seed whose greedy policy puts more than this share of its mass on one
# action has collapsed to a constant policy; its cost reflects budget, not
# architecture (run_sat_ablation.py:84-95).
DEGENERACY_LIMIT = 0.9

RESULTS_ROOT = "benchmark_results"
WORKDIR_ROOT = "dynaplex_runs_benchmark"
