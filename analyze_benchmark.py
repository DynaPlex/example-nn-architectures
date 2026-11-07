"""Read `benchmark_results/`, check the validity gates, run the pre-registered
tests, and emit the results table.

This script is the only thing allowed to state a verdict, and it refuses to state
one if a gate fails. The gates and the decision rule are fixed in
`PREREGISTRATION.md` BEFORE the run, so what happens here is confirmatory rather
than a search for a favourable framing.

Usage:
    python analyze_benchmark.py                     # the default budget
    python analyze_benchmark.py --budget quick
    python analyze_benchmark.py --out RESULTS.md
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np

from benchmark_config import (
    BUDGETS,
    DEGENERACY_LIMIT,
    RESULTS_ROOT,
    SPLICE_PROBE,
)
from benchmark_stats import (
    bootstrap_diff_ci,
    cohens_d,
    holm,
    iqm,
    permutation_test,
    probability_of_improvement,
)

# The pre-registered primary contrasts: SAT against each original attention net.
# `(challenger, baseline)`; the metric is cost per period, lower is better.
PRIMARY_CONTRASTS = (("sat", "tokens"), ("sat", "pointer"))
ALPHA = 0.05


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(budget_tag: str) -> list[dict]:
    pattern = os.path.join(RESULTS_ROOT, budget_tag, "*.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        if os.path.basename(path) == "summary.json":
            continue
        with open(path, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    return rows


def by_arm(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["arm"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["seed"])
    return out


# --------------------------------------------------------------------------
# validity gates
# --------------------------------------------------------------------------

def check_gates(rows: list[dict], arms: dict[str, list[dict]],
                anchors: dict | None = None) -> list[dict]:
    """Every gate returns a dict with `name`, `ok`, `detail`. A failed gate blocks
    the verdict; it never silently rewrites the numbers."""
    gates: list[dict] = []

    # --- gate 1: CRN splice ------------------------------------------------
    # The CELL-encoded pointer arm needs its own comparer call, and seeds added
    # later are evaluated in later calls. Both are only legitimate if separate
    # calls land on the SAME episodes. myopic is in every call, so its
    # per-trajectory returns are the probe: element-wise identical everywhere, or
    # the results are not one paired frame and cannot be pooled.
    # Group by the returns vector rather than comparing everything to row 0: with
    # a single-reference check one odd row makes every OTHER row look wrong, which
    # points the reader at the wrong arm. The majority group is taken as the
    # reference and the minorities are named.
    groups: dict[str, list[str]] = {}
    for r in rows:
        got = np.asarray(r["anchors"][SPLICE_PROBE]["returns"], dtype=float)
        key = hashlib.sha256(got.tobytes()).hexdigest()[:12]
        groups.setdefault(key, []).append(f"{r['arm']}_s{r['seed']}")
    if len(groups) <= 1:
        detail = f"all {len(rows)} comparer records agree element-wise"
    else:
        majority = max(groups, key=lambda k: len(groups[k]))
        odd = {k: sorted(v) for k, v in groups.items() if k != majority}
        detail = (f"{len(groups)} distinct myopic return vectors; majority "
                  f"{majority} has {len(groups[majority])} records, odd ones out: "
                  f"{odd}")
    gates.append({
        "name": "CRN splice (myopic returns identical across all comparer calls)",
        "ok": len(groups) <= 1,
        "detail": detail,
    })

    # --- gate 2: anchor reproduction --------------------------------------
    # The reference values (myopic 3.096, random 52.8) are warmup-100/horizon-500
    # numbers, so this only means something when the comparer config matches.
    # `Budget.anchors is None` (the quick budget) declares it not-checkable rather
    # than inventing a range: at horizon 200 the transient dominates and random
    # reads ~40, which is correct, not a failure.
    if not anchors:
        gates.append({
            "name": "anchor reproduction (myopic/random match the reference table)",
            "ok": None,
            "detail": "not checkable: this budget evaluates at a comparer config "
                      "the reference values were not measured at",
        })
    else:
        bad = []
        for r in rows:
            for name, bounds in anchors.items():
                lo, hi = bounds
                m = float(r["anchors"][name]["mean"])
                if not (lo <= m <= hi):
                    bad.append(f"{r['arm']}_s{r['seed']}:{name}={m:.3f}")
        gates.append({
            "name": "anchor reproduction (myopic/random match the reference table)",
            "ok": not bad,
            "detail": ("myopic and random in range in every call" if not bad
                       else "out of range: " + ", ".join(sorted(set(bad)))),
        })

    # --- gate 3: data matching --------------------------------------------
    # sat and tokens collect under the same DIRECTIONAL MDP with the same DCL
    # seed, so their labels should be identical and only the features differ.
    digs = {a: (v[0].get("sample_digest") or {}) for a, v in arms.items()}
    d_sat, d_tok = digs.get("sat"), digs.get("tokens")
    if d_sat and d_tok:
        same = (d_sat.get("y_sha") == d_tok.get("y_sha")
                and d_sat.get("mask_sha") == d_tok.get("mask_sha"))
        detail = ("sat and tokens fit identical states and labels; sampling "
                  "noise cancels between them" if same else
                  f"labels DIFFER (sat {d_sat.get('y_sha')} vs tokens "
                  f"{d_tok.get('y_sha')}) -- the arms are budget-matched but NOT "
                  f"data-matched; say so in the write-up")
    else:
        same, detail = None, "sat and/or tokens missing; not checkable"
    gates.append({
        "name": "data matching (sat vs tokens fit the same labels)",
        "ok": same, "detail": detail, "advisory": True,
    })

    # --- gate 4: degeneracy -----------------------------------------------
    degen = [f"{r['arm']}_s{r['seed']}={r['degeneracy_share']:.0%}"
             for r in rows if r.get("degeneracy_share", 0) > DEGENERACY_LIMIT]
    gates.append({
        "name": f"degeneracy (no arm-seed above {DEGENERACY_LIMIT:.0%} greedy dominance)",
        "ok": not degen,
        "detail": ("no arm-seed collapsed to a constant policy" if not degen
                   else "collapsed, scores reflect budget not architecture: "
                        + ", ".join(degen)),
    })

    # --- gate 5: budget truncation ----------------------------------------
    # BLOCKING since 2026-08-22. This was advisory, it fired correctly at the
    # default budget (7/8, 8/8, 7/8 capped), and the verdict was issued anyway --
    # which a 200-epoch probe then showed to be an artifact of the cap. An
    # advisory gate cannot guard an assumption the whole comparison rests on.
    # Threshold: more than half an arm's seeds capped means that arm was still
    # improving when it was stopped, so its cost measures the budget.
    capped = {}
    for arm, v in arms.items():
        n_cap = sum(1 for r in v if not r.get("stopped_early"))
        if n_cap:
            capped[arm] = f"{n_cap}/{len(v)}"
    blocking = {a: c for a, c in capped.items()
                if int(c.split("/")[0]) * 2 > int(c.split("/")[1])}
    gates.append({
        "name": "budget truncation (arms converged rather than hitting the epoch cap)",
        "ok": not blocking,
        "detail": ("every run early-stopped on patience" if not capped
                   else ("some runs capped, but under half per arm: " + str(capped)
                         if not blocking
                         else "MOST seeds hit the epoch cap, so these arms were "
                              "still improving and their cost measures the BUDGET, "
                              "not the architecture: " + str(blocking)
                              + " -- raise epochs (see --budget long)")),
        "advisory": bool(capped) and not blocking,
    })

    # --- gate 6: balance ---------------------------------------------------
    counts = {a: len(v) for a, v in arms.items()}
    balanced = len(set(counts.values())) == 1 and min(counts.values(), default=0) >= 5
    gates.append({
        "name": "seed count (equal across arms, at least 5)",
        "ok": balanced,
        "detail": str(counts) + ("" if balanced else
                                 "  -- K<5 cannot reach p<0.05 in an exact "
                                 "permutation test, and unequal K unbalances it"),
    })
    return gates


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyse(arms: dict[str, list[dict]]) -> tuple[dict, dict]:
    costs = {a: np.array([r["mean"] for r in v], dtype=float)
             for a, v in arms.items()}
    contrasts: dict[str, dict] = {}
    raw_p: dict[str, float] = {}

    for challenger, baseline in PRIMARY_CONTRASTS:
        if challenger not in costs or baseline not in costs:
            continue
        a, b = costs[challenger], costs[baseline]
        key = f"{challenger} vs {baseline}"
        diff, p, exact = permutation_test(a, b)
        mean_pt, mean_lo, mean_hi = bootstrap_diff_ci(a, b)
        iqm_pt, iqm_lo, iqm_hi = bootstrap_diff_ci(a, b, statistic=iqm)
        poi, poi_lo, poi_hi = probability_of_improvement(a, b)
        contrasts[key] = {
            "challenger": challenger, "baseline": baseline,
            "n_challenger": len(a), "n_baseline": len(b),
            "mean_diff": mean_pt, "mean_ci": [mean_lo, mean_hi],
            "iqm_diff": iqm_pt, "iqm_ci": [iqm_lo, iqm_hi],
            "p_raw": p, "p_exact": exact,
            "prob_improvement": poi, "poi_ci": [poi_lo, poi_hi],
            "cohens_d": cohens_d(a, b),
            "worst_challenger": float(a.max()), "best_baseline": float(b.min()),
            "separated": bool(a.max() < b.min()),
        }
        raw_p[key] = p

    adjusted = holm(raw_p) if raw_p else {}
    for key, adj in adjusted.items():
        c = contrasts[key]
        c["p_holm"] = adj
        # THE PRE-REGISTERED DECISION RULE, applied mechanically.
        c["passes"] = bool(adj < ALPHA and c["mean_ci"][1] < 0.0
                           and c["poi_ci"][0] > 0.5)
    return contrasts, {a: c for a, c in costs.items()}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def render(budget_name: str, rows: list[dict], arms: dict[str, list[dict]],
           gates: list[dict], contrasts: dict, costs: dict) -> str:
    L: list[str] = []
    w = L.append
    b = rows[0]
    w(f"# Architecture benchmark: SAT/AGILE vs the original attention nets")
    w("")
    w(f"Budget `{budget_name}` ({b['budget_tag']}): "
      f"collect `{b['collect']}`, train `{b['train']}`, comparer `{b['comparer']}`.")
    w(f"DCL seed {b['dcl_seed']} (shared by every arm), comparer seed "
      f"{b['comparer_seed']}. torch {b['torch']}, numpy {b['numpy']}, "
      f"{b['workers']} workers.")
    w("")
    w("Metric is **mean cost per period, lower is better**. The unit of analysis "
      "is the **training seed**: the comparer's trajectory SE is ~0.02 while the "
      "spread across seeds is ~0.2, so a single-seed number carries roughly ten "
      "times more uncertainty than its own error bar suggests.")
    w("")

    # --- gates -------------------------------------------------------------
    w("## Validity gates")
    w("")
    w("| gate | status | detail |")
    w("|---|---|---|")
    for g in gates:
        if g["ok"] is None:
            status = "n/a"
        elif g["ok"]:
            status = "pass"
        else:
            status = "ADVISORY" if g.get("advisory") else "**FAIL**"
        w(f"| {g['name']} | {status} | {g['detail']} |")
    w("")

    # --- per-seed table ----------------------------------------------------
    w("## Per-seed results")
    w("")
    w("Raw values, never only summaries -- the spread is the finding.")
    w("")
    w("| arm | " + " | ".join(f"seed {r['seed']}" for r in next(iter(arms.values())))
      + " | mean | IQM | sd | min | max |")
    w("|---" * (2 + len(next(iter(arms.values()))) + 4) + "|")
    for arm, v in arms.items():
        c = costs[arm]
        cells = " | ".join(f"{r['mean']:.3f}" for r in v)
        w(f"| `{arm}` | {cells} | **{c.mean():.3f}** | {iqm(c):.3f} | "
          f"{c.std(ddof=1) if len(c) > 1 else float('nan'):.3f} | "
          f"{c.min():.3f} | {c.max():.3f} |")
    anc = rows[0]["anchors"]
    w(f"| myopic | " + " | ".join([""] * len(next(iter(arms.values()))))
      + f" | {anc['myopic']['mean']:.3f} | | | | |")
    w(f"| random | " + " | ".join([""] * len(next(iter(arms.values()))))
      + f" | {anc['random']['mean']:.3f} | | | | |")
    w("")

    # --- diagnostics -------------------------------------------------------
    w("## Training diagnostics")
    w("")
    w("| arm | epochs run | early-stopped | best val loss | val regret | greedy dominance |")
    w("|---|---|---|---|---|---|")
    for arm, v in arms.items():
        def mv(k, fmt="{:.3f}"):
            vals = [r.get(k) for r in v if r.get(k) is not None]
            return fmt.format(float(np.mean(vals))) if vals else "-"
        early = sum(1 for r in v if r.get("stopped_early"))
        w(f"| `{arm}` | {mv('epochs_run', '{:.1f}')} | {early}/{len(v)} | "
          f"{mv('best_val_loss', '{:.4f}')} | {mv('val_regret', '{:.4f}')} | "
          f"{mv('degeneracy_share', '{:.1%}')} |")
    w("")

    # --- cost --------------------------------------------------------------
    w("## Training cost")
    w("")
    w(f"Wall-clock on {b['workers']} workers "
      f"(`torch_threads={min(b['workers'], 32)}`), {b['platform']}. "
      "Collection is timed as its own phase and paid **once per arm**, shared by "
      "that arm's seeds; `train` is therefore pure training in every row. "
      "`s/epoch` is the size-independent figure -- arms early-stop at different "
      "points, so total seconds alone confounds 'slower per epoch' with 'trained "
      "for longer'.")
    w("")
    w("| arm | params | collect (once) | train: mean | sd | min | max | epochs | s/epoch |")
    w("|---|---|---|---|---|---|---|---|---|")
    for arm, v in arms.items():
        ts = np.array([r["train_seconds"] for r in v if r.get("train_seconds")],
                      dtype=float)
        spe = [r.get("seconds_per_epoch") for r in v if r.get("seconds_per_epoch")]
        ep = [r.get("epochs_run") for r in v if r.get("epochs_run")]
        col = next((r.get("collect_seconds") for r in v
                    if r.get("collect_seconds")), None)
        sd = ts.std(ddof=1) if len(ts) > 1 else float("nan")
        w(f"| `{arm}` | {v[0].get('num_parameters', 0):,} | "
          f"{f'{col:,.0f}s' if col else '(reused)'} | "
          f"**{ts.mean():,.0f}s** | {sd:,.0f}s | {ts.min():,.0f}s | "
          f"{ts.max():,.0f}s | {np.mean(ep):.1f} | "
          f"{np.mean(spe):.1f}s |")
    total = sum(r.get("train_seconds") or 0 for r in rows)
    total += sum({(r["arm"]): (r.get("collect_seconds") or 0)
                  for r in rows}.values())
    w("")
    w(f"Total measured compute: **{total / 3600:.1f} h** across "
      f"{len(rows)} training runs and {len(arms)} collections.")
    w("")

    # --- contrasts ---------------------------------------------------------
    w("## Pre-registered contrasts")
    w("")
    if not contrasts:
        w("_No primary contrast is computable from the arms present._")
    else:
        w("| contrast | mean diff | 95% CI | IQM diff | permutation p | Holm p | "
          "P(improvement) | 95% CI | separated | rule |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for key, c in contrasts.items():
            w(f"| {key} | {c['mean_diff']:+.3f} | "
              f"[{c['mean_ci'][0]:+.3f}, {c['mean_ci'][1]:+.3f}] | "
              f"{c['iqm_diff']:+.3f} | {c['p_raw']:.4g}"
              f"{'' if c['p_exact'] else ' (MC)'} | {c['p_holm']:.4g} | "
              f"{c['prob_improvement']:.2f} | "
              f"[{c['poi_ci'][0]:.2f}, {c['poi_ci'][1]:.2f}] | "
              f"{'yes' if c['separated'] else 'no'} | "
              f"{'pass' if c['passes'] else 'fail'} |")
        w("")
        w("`mean diff` is challenger minus baseline, so **negative favours the "
          "challenger**. `separated` means the challenger's worst seed still beat "
          "the baseline's best seed. `P(improvement)` is the chance the challenger "
          "wins if each is trained once.")
    w("")

    # --- verdict -----------------------------------------------------------
    w("## Verdict")
    w("")
    blocking = [g for g in gates if g["ok"] is False and not g.get("advisory")]
    if blocking:
        w("**No verdict.** These gates failed, and each one breaks an assumption "
          "the tests above rest on:")
        w("")
        for g in blocking:
            w(f"- {g['name']} -- {g['detail']}")
        w("")
        w("The numbers are still printed, but they are not evidence until the "
          "gates pass.")
    elif not contrasts:
        w("**No verdict** -- no primary contrast could be computed.")
    else:
        all_pass = all(c["passes"] for c in contrasts.values())
        if all_pass:
            w("**SAT is better than both original attention networks at this "
              "budget.** Every pre-registered condition holds for both contrasts: "
              "Holm-adjusted permutation p < 0.05, the bootstrap CI on the mean "
              "difference lies entirely below zero, and the lower bound on "
              "P(improvement) exceeds 0.5.")
        else:
            w("**Not established.** At least one contrast fails the "
              "pre-registered rule:")
            w("")
            for key, c in contrasts.items():
                if c["passes"]:
                    continue
                why = []
                if c["p_holm"] >= ALPHA:
                    why.append(f"Holm p = {c['p_holm']:.3g} is not below {ALPHA}")
                if c["mean_ci"][1] >= 0:
                    why.append(f"the mean-difference CI reaches "
                               f"{c['mean_ci'][1]:+.3f}, above zero")
                if c["poi_ci"][0] <= 0.5:
                    why.append(f"P(improvement) could be as low as "
                               f"{c['poi_ci'][0]:.2f}")
                w(f"- **{key}**: observed difference {c['mean_diff']:+.3f}; "
                  + "; ".join(why) + ".")
            w("")
            w("The observed direction and magnitude are reported above regardless. "
              "'Not established' is a statement about the evidence, not a claim "
              "that the architectures are equal.")
    w("")
    w("### What this does not show")
    w("")
    w("- **The win, if any, is architecture and features together.** "
      "`SATFeaturizer` writes `nearest_order_dist` and `has_order` per candidate "
      "action (`featurizers_sat.py:205-224`) -- the myopic benchmark's own "
      "decision statistic -- which neither baseline receives. `run_sat_ablation.py` "
      "splits the SAT-internal half of this; the `sat_nofeat` arm here "
      "(`--arms sat,tokens,pointer,sat_nofeat`) closes the rest on the same "
      "sample set.")
    w("- **One MDP configuration, one budget, one training recipe.** No arm was "
      "tuned; all three inherit the shared `lr=1e-4` recipe both original run "
      "scripts already used. An arm that would prefer a different learning rate "
      "is penalised here.")
    w("- **Seeds vary weight init and shuffle order only**, not the collected "
      "data: within an arm every seed fits one shared sample set. The claim "
      "generalises over training runs, not over collections.")
    params = {a: v[0].get("num_parameters") for a, v in arms.items()}
    if all(params.values()) and "sat" in params:
        others = {a: n for a, n in params.items() if a != "sat"}
        ratio = max(params["sat"] / n for n in others.values()) if others else 1.0
        w(f"- **The arms are not capacity-matched.** `sat` carries "
          f"{params['sat']:,} parameters against "
          + ", ".join(f"`{a}` {n:,}" for a, n in sorted(others.items()))
          + f" -- up to {ratio:.1f}x. The transformer size knobs (`d_model=64, "
            "n_layers=2, n_heads=4, ff=128`) are identical; the difference is the "
            "inter-action stack itself, which is the thing under test and cannot "
            "be removed without removing the architecture. So a win here is a win "
            "for a bigger model too, and `run_sat_ablation.py`'s capacity-matched "
            "MLP control is the experiment that separates those.")
    w("- **Cost per period is the only endpoint.** Parameter counts and training "
      "time are reported above but are not part of the decision rule; SAT's "
      "featurizer is also materially more expensive per state, since `_nearest` "
      "scans the whole grid once per action token.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", choices=sorted(BUDGETS), default="default")
    ap.add_argument("--out", default=None,
                    help="also write the markdown report here "
                         "(default: benchmark_results/<tag>/RESULTS.md)")
    args = ap.parse_args()

    budget = BUDGETS[args.budget]
    rows = load(budget.tag)
    if not rows:
        raise SystemExit(
            f"no results under {os.path.join(RESULTS_ROOT, budget.tag)} -- run "
            f"`python run_benchmark.py --budget {args.budget}` first")

    arms = by_arm(rows)
    gates = check_gates(rows, arms, budget.anchors)
    contrasts, costs = analyse(arms)
    report = render(budget.name, rows, arms, gates, contrasts, costs)
    print(report)

    out = args.out or os.path.join(RESULTS_ROOT, budget.tag, "RESULTS.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report)
    summary = {
        "budget": budget.name,
        "gates": gates,
        "contrasts": contrasts,
        "per_seed": {a: [float(x) for x in c] for a, c in costs.items()},
    }
    sp = os.path.join(RESULTS_ROOT, budget.tag, "summary.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {out} and {sp}")


if __name__ == "__main__":
    main()
