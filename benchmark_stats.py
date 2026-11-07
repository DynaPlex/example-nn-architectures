"""The statistics for the architecture benchmark. Numpy only, no scipy.

Kept in its own module so `test_run_benchmark.py` can check each estimator
against a hand-computable case without touching the experiment harness. There is
deliberately no new dependency: this repo has no `pyproject.toml` on purpose
(README.md:389), and everything here is a few lines of numpy.

THE ONE IDEA THAT SHAPES ALL OF IT. Two noise sources sit between an architecture
and a number: evaluation noise (which trajectories the comparer drew) and
training noise (weight init, shuffle order). On this problem the comparer's SE is
~0.02-0.03 while the spread across training seeds is ~0.2. Evaluation noise is an
order of magnitude smaller, so buying more comparer trajectories is nearly
worthless and buying more seeds is everything -- and the unit of analysis has to
be the SEED, not the trajectory. Every function here takes one value per seed.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def permutation_test(a: np.ndarray, b: np.ndarray,
                     max_exact: int = 200_000,
                     resamples: int = 100_000,
                     rng_seed: int = 0) -> tuple[float, float, bool]:
    """Two-sided permutation test on the difference of means, `a` minus `b`.

    Returns (observed difference, p-value, exact?). With K seeds per arm there
    are C(2K, K) splits; at K=8 that is 12,870, so the test is enumerated
    EXACTLY. Above `max_exact` splits it falls back to `resamples` random
    relabellings. The exact p-value counts the observed split itself, which is
    why the floor is 2/C(2K, K) rather than 0 -- at K=5 the smallest attainable
    p is 0.0079, which is the real reason K=5 is the floor for this design.

    A permutation test is used rather than Welch's t because K is small, the
    per-seed cost distribution is visibly skewed (a bad init produces a long
    right tail), and normality is not something this design can check.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("permutation_test needs at least one value per group")
    observed = float(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n_a, n = len(a), len(pool)
    total = _n_choose_k(n, n_a)

    if total <= max_exact:
        idx = np.arange(n)
        diffs = np.empty(total, dtype=float)
        for i, left in enumerate(combinations(idx, n_a)):
            sel = np.zeros(n, dtype=bool)
            sel[list(left)] = True
            diffs[i] = pool[sel].mean() - pool[~sel].mean()
        exact = True
    else:
        rng = np.random.default_rng(rng_seed)
        diffs = np.empty(resamples, dtype=float)
        for i in range(resamples):
            perm = rng.permutation(n)
            diffs[i] = pool[perm[:n_a]].mean() - pool[perm[n_a:]].mean()
        exact = False

    # `>=` with a tolerance: floating-point means of the same multiset can differ
    # in the last bit, and dropping the equal cases would bias p downward.
    tol = 1e-12 * max(1.0, abs(observed))
    p = float((np.abs(diffs) >= abs(observed) - tol).mean())
    if not exact:
        # add-one correction; a Monte-Carlo p of exactly 0 is not a p-value
        p = float((np.sum(np.abs(diffs) >= abs(observed) - tol) + 1) / (len(diffs) + 1))
    return observed, p, exact


def _n_choose_k(n: int, k: int) -> int:
    from math import comb
    return comb(n, k)


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, *, alpha: float = 0.05,
                      resamples: int = 20_000, rng_seed: int = 0,
                      statistic=None) -> tuple[float, float, float]:
    """(point estimate, lo, hi) for `statistic(a) - statistic(b)`.

    Percentile bootstrap, resampling each group independently (the groups are
    independent: different architectures consume weight-init RNG differently, so
    "seed 3" is a label, not a common random number). `statistic` defaults to the
    mean; pass `iqm` for the robust version.
    """
    stat = statistic if statistic is not None else np.mean
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(rng_seed)
    point = float(stat(a) - stat(b))
    draws = np.empty(resamples, dtype=float)
    for i in range(resamples):
        ra = a[rng.integers(0, len(a), len(a))]
        rb = b[rng.integers(0, len(b), len(b))]
        draws[i] = stat(ra) - stat(rb)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle 50 % of the values.

    Agarwal et al. (2021), "Deep RL at the Edge of the Statistical Precipice" --
    with a handful of seeds the sample mean is dominated by outliers and the
    median throws away too much, so the IQM is the recommended middle. Reported
    alongside the mean, never instead of it.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 4:
        return float(x.mean())
    lo = int(np.floor(n * 0.25))
    hi = int(np.ceil(n * 0.75))
    return float(x[lo:hi].mean())


def probability_of_improvement(a: np.ndarray, b: np.ndarray, *,
                               alpha: float = 0.05, resamples: int = 20_000,
                               rng_seed: int = 0) -> tuple[float, float, float]:
    """P(a beats b) for a LOWER-IS-BETTER metric, i.e. P(a < b) over independent
    seed draws, with a bootstrap CI. Ties count as half.

    This is the Mann-Whitney U statistic scaled to [0, 1]. It answers the
    question a mean difference cannot: "if I train each architecture once, how
    often does this one come out ahead?" 1.0 means the two seed sets do not
    overlap at all.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    def _poi(x, y):
        d = x[:, None] - y[None, :]
        return float((np.sum(d < 0) + 0.5 * np.sum(d == 0)) / d.size)

    point = _poi(a, b)
    rng = np.random.default_rng(rng_seed)
    draws = np.empty(resamples, dtype=float)
    for i in range(resamples):
        ra = a[rng.integers(0, len(a), len(a))]
        rb = b[rng.integers(0, len(b), len(b))]
        draws[i] = _poi(ra, rb)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment over a family of contrasts.

    Returns adjusted p-values (monotone, capped at 1.0), so the decision rule can
    stay "adjusted p < 0.05" regardless of how many contrasts the family has.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)   # enforce monotonicity
        out[name] = running
    return {k: out[k] for k in pvalues}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference with a pooled SD. Reported for sizing the
    NEXT experiment, not for the decision -- with K under ~10 it is badly biased."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    if pooled == 0:
        gap = a.mean() - b.mean()
        # sign matters: an infinite effect can still favour either arm
        return 0.0 if gap == 0 else float(np.sign(gap)) * float("inf")
    return float((a.mean() - b.mean()) / pooled)


# ---------------------------------------------------------------------------
# censoring-aware comparison, for "epochs to reach quality tau"
# ---------------------------------------------------------------------------
#
# A seed that never reaches tau within the ladder has NO epochs-to-threshold. It
# cannot be dropped (that biases toward whichever arm fails more often) and it
# cannot be set to max(ladder) (that pretends it just made it). Representing it as
# +inf keeps it in the sample, ordered worse than every observed value, which is
# exactly what is known about it. Means are then undefined, so every statistic
# below is rank-based.

def rank_win_rate(a: np.ndarray, b: np.ndarray) -> float:
    """P(a beats b) for a lower-is-better metric, ties counted as half.

    Identical to `probability_of_improvement`'s point estimate but written to be
    safe with +inf on both sides: `inf < inf` is False and `inf == inf` is True,
    so two censored seeds tie rather than producing nan.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d_lt = (a[:, None] < b[None, :]).sum()
    d_eq = (a[:, None] == b[None, :]).sum()
    return float((d_lt + 0.5 * d_eq) / (len(a) * len(b)))


def permutation_test_ranks(a: np.ndarray, b: np.ndarray,
                           max_exact: int = 200_000, resamples: int = 100_000,
                           rng_seed: int = 0) -> tuple[float, float, bool]:
    """Two-sided permutation test on `rank_win_rate`, i.e. on the Mann-Whitney U.

    Returns (observed win rate, p-value, exact?). Used instead of
    `permutation_test` whenever the values may be censored (+inf), where a
    difference of means is undefined. Under the null the win rate is 0.5, so
    "extreme" means far from 0.5 in either direction.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = rank_win_rate(a, b)
    pool = np.concatenate([a, b])
    n_a, n = len(a), len(pool)
    total = _n_choose_k(n, n_a)

    if total <= max_exact:
        idx = np.arange(n)
        stats = np.empty(total, dtype=float)
        for i, left in enumerate(combinations(idx, n_a)):
            sel = np.zeros(n, dtype=bool)
            sel[list(left)] = True
            stats[i] = rank_win_rate(pool[sel], pool[~sel])
        exact = True
    else:
        rng = np.random.default_rng(rng_seed)
        stats = np.empty(resamples, dtype=float)
        for i in range(resamples):
            perm = rng.permutation(n)
            stats[i] = rank_win_rate(pool[perm[:n_a]], pool[perm[n_a:]])
        exact = False

    tol = 1e-12
    extreme = np.abs(stats - 0.5) >= abs(observed - 0.5) - tol
    if exact:
        p = float(extreme.mean())
    else:
        p = float((extreme.sum() + 1) / (len(stats) + 1))
    return observed, p, exact


def censored_summary(x: np.ndarray) -> dict:
    """Reached-count and the quantiles among those that reached, for reporting.

    Never collapses a censored sample to a single number -- "5 of 8 seeds reached
    tau, median 40 epochs among those" is the honest statement, and it is what a
    reader needs to interpret the win rate.
    """
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    return {
        "n": int(len(x)),
        "reached": int(len(finite)),
        "median": (float(np.median(finite)) if len(finite) else None),
        "min": (float(finite.min()) if len(finite) else None),
        "max": (float(finite.max()) if len(finite) else None),
    }
