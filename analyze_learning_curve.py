"""Does SAT reach a given policy quality in fewer epochs than the alternatives?

Reads `benchmark_results/<tag>/`, builds the per-arm learning curves, and tests
the convergence-speed claim. As with `analyze_benchmark.py`, this is the only
thing allowed to state a conclusion, and it refuses to state one if a gate fails.

Endpoints, fixed before looking:

  * PRIMARY -- epochs-to-threshold: the first ladder rung at which an arm reaches
    cost <= tau, per seed. Seeds that never reach tau are CENSORED (+inf), never
    dropped and never coerced to max(ladder). Compared with a permutation test on
    the Mann-Whitney win rate, which is well-defined under censoring where a
    difference of means is not.
  * SECONDARY -- wall-clock-to-threshold, because epochs are not free and the arms
    differ slightly in seconds per epoch. An architecture that needs half the
    epochs but twice the time per epoch is not faster.
  * SECONDARY -- capped area under the learning curve, which uses every rung
    rather than just the crossing. Costs are capped at the random policy's before
    averaging, so one catastrophic early rung cannot dominate a summary of the
    whole curve.

Thresholds: myopic's cost (a level with independent meaning) and 1.05x the best
cost any arm reaches anywhere on the ladder (a level every arm plausibly targets).

Usage:
    python analyze_learning_curve.py
    python analyze_learning_curve.py --tag learning_curve --out CURVES.md
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np

from benchmark_stats import (
    censored_summary,
    holm,
    permutation_test_ranks,
    rank_win_rate,
)
from learning_curve import epochs_to_threshold

CHALLENGER = "sat"
BASELINES = ("tokens", "pointer")
ALPHA = 0.05


def load(tag: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join("benchmark_results", tag, "*.json"))):
        if os.path.basename(path) in ("summary.json",):
            continue
        with open(path, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    return rows


def organise(rows: list[dict]):
    """-> {arm: {seed: {epochs: row}}}, plus the sorted ladder."""
    out: dict[str, dict[int, dict[int, dict]]] = {}
    ladder = set()
    for r in rows:
        out.setdefault(r["arm"], {}).setdefault(r["seed"], {})[r["rung_epochs"]] = r
        ladder.add(r["rung_epochs"])
    return out, sorted(ladder)


def check_gates(rows, curves, ladder) -> list[dict]:
    gates = []

    groups: dict[str, list[str]] = {}
    for r in rows:
        got = np.asarray(r["anchors"]["myopic"]["returns"], dtype=float)
        key = hashlib.sha256(got.tobytes()).hexdigest()[:12]
        groups.setdefault(key, []).append(f"{r['arm']}_s{r['seed']}_e{r['rung_epochs']}")
    gates.append({
        "name": "CRN splice (myopic returns identical across comparer calls)",
        "ok": len(groups) <= 1,
        "detail": (f"all {len(rows)} records agree element-wise" if len(groups) <= 1
                   else f"{len(groups)} distinct myopic return vectors"),
    })

    bad = [f"{r['arm']}_s{r['seed']}" for r in rows
           if not (3.0 <= float(r["anchors"]["myopic"]["mean"]) <= 3.25)]
    gates.append({
        "name": "anchor reproduction (myopic near the reference 3.096)",
        "ok": not bad,
        "detail": "myopic in range everywhere" if not bad
                  else f"out of range for {sorted(set(bad))}",
    })

    # Every arm must have every rung for every seed, or the curves are not
    # comparable and a missing rung would silently shift a crossing.
    holes = []
    for arm, seeds in curves.items():
        for seed, byE in seeds.items():
            missing = [e for e in ladder if e not in byE]
            if missing:
                holes.append(f"{arm}_s{seed}:{missing}")
    gates.append({
        "name": "complete ladder (every arm-seed has every rung)",
        "ok": not holes,
        "detail": f"all arms cover {ladder}" if not holes else "missing " + ", ".join(holes),
    })

    counts = {a: len(s) for a, s in curves.items()}
    gates.append({
        "name": "seed count (equal across arms, at least 5)",
        "ok": len(set(counts.values())) == 1 and min(counts.values(), default=0) >= 5,
        "detail": str(counts),
    })

    # The top rung must be a plateau, or the ladder stopped before the arms
    # converged and "epochs to threshold" is measuring the ladder, not the arm.
    # Snapshots carry best-val-so-far, so a converged arm's last two rungs are
    # close; a big drop means it was still improving.
    still = {}
    for arm, seeds in curves.items():
        drops = []
        for seed, byE in seeds.items():
            if len(ladder) >= 2 and ladder[-1] in byE and ladder[-2] in byE:
                a, b = byE[ladder[-2]]["mean"], byE[ladder[-1]]["mean"]
                if a > 0:
                    drops.append((a - b) / a)
        if drops and float(np.mean(drops)) > 0.05:
            still[arm] = f"{np.mean(drops):+.1%}"
    gates.append({
        "name": f"ladder reaches a plateau (cost still falling at E={ladder[-1]})",
        "ok": not still,
        "detail": "top two rungs within 5% for every arm" if not still
                  else "still improving at the top rung, so the ladder is too "
                       "short and crossings are censored by the LADDER rather "
                       "than by the architecture: " + str(still),
    })
    return gates


def threshold_table(curves, ladder, tau, key="mean", time_key=None):
    """{arm: array of epochs- (or seconds-) to-threshold, inf where never reached}"""
    out = {}
    for arm, seeds in curves.items():
        vals = []
        for seed in sorted(seeds):
            byE = seeds[seed]
            costs = [byE[e][key] for e in ladder if e in byE]
            es = [e for e in ladder if e in byE]
            hit = epochs_to_threshold(es, costs, tau)
            if hit is None:
                vals.append(float("inf"))
            elif time_key is None:
                vals.append(float(hit))
            else:
                vals.append(float(byE[int(hit)][time_key]))
        out[arm] = np.array(vals, dtype=float)
    return out


def capped_aulc(curves, ladder, cap):
    out = {}
    for arm, seeds in curves.items():
        vals = []
        for seed in sorted(seeds):
            byE = seeds[seed]
            costs = [min(byE[e]["mean"], cap) for e in ladder if e in byE]
            vals.append(float(np.mean(costs)))
        out[arm] = np.array(vals, dtype=float)
    return out


def render(tag, rows, curves, ladder, gates) -> str:
    L: list[str] = []
    w = L.append
    b = rows[0]
    lrs = {arm: next(iter(s.values()))[ladder[0]]["lr"] for arm, s in curves.items()}
    myopic = float(b["anchors"]["myopic"]["mean"])
    random_cost = float(b["anchors"]["random"]["mean"])

    w("# Does SAT learn faster than the original attention networks?")
    w("")
    w(f"Ladder `{ladder}` epochs, {len(next(iter(curves.values())))} seeds per arm, "
      f"samples from budget `{b['source_budget']}`, comparer `{b['comparer']}` "
      f"seed {b['comparer_seed']}.")
    w("")
    w("Each arm runs at **its own learning rate**, chosen by `run_lr_screen.py` "
      "from a common grid on common samples -- "
      + ", ".join(f"`{a}` {lr:g}" for a, lr in sorted(lrs.items()))
      + ". Holding one rate across arms would make this a measurement of how well "
        "that rate suits each architecture rather than of the architectures: the "
        "screen found `tokens` averaging 13.21 at 1e-4 and 3.01 at 1e-3.")
    w("")
    w(f"Anchors: myopic **{myopic:.3f}**, random **{random_cost:.1f}**.")
    w("")

    w("## Validity gates")
    w("")
    w("| gate | status | detail |")
    w("|---|---|---|")
    for g in gates:
        status = "n/a" if g["ok"] is None else ("pass" if g["ok"] else "**FAIL**")
        w(f"| {g['name']} | {status} | {g['detail']} |")
    w("")

    w("## Learning curves")
    w("")
    w("Mean cost across seeds at each rung; per-seed values below.")
    w("")
    w("| arm | " + " | ".join(f"E={e}" for e in ladder) + " |")
    w("|---" * (len(ladder) + 1) + "|")
    for arm in sorted(curves):
        cells = []
        for e in ladder:
            v = [curves[arm][s][e]["mean"] for s in curves[arm] if e in curves[arm][s]]
            cells.append(f"{np.mean(v):.3f}" if v else "-")
        w(f"| `{arm}` | " + " | ".join(cells) + " |")
    w("")
    for arm in sorted(curves):
        w(f"**`{arm}`** per seed:")
        w("")
        w("| seed | " + " | ".join(f"E={e}" for e in ladder) + " | best epoch |")
        w("|---" * (len(ladder) + 2) + "|")
        for s in sorted(curves[arm]):
            byE = curves[arm][s]
            cells = " | ".join(f"{byE[e]['mean']:.3f}" if e in byE else "-"
                               for e in ladder)
            w(f"| {s} | {cells} | {byE[ladder[-1]]['best_epoch']} |")
        w("")

    best_any = min(curves[a][s][e]["mean"]
                   for a in curves for s in curves[a] for e in curves[a][s])
    taus = {"myopic (3.096)": myopic, f"1.05x best observed ({best_any:.3f})":
            1.05 * best_any}

    w("## Epochs to reach a quality threshold")
    w("")
    w("Primary endpoint. `inf` = never reached within the ladder -- **censored, not "
      "dropped**: dropping those seeds would bias the comparison toward whichever "
      "arm fails more often, and coercing them to the top rung would pretend they "
      "just made it.")
    w("")
    contrasts = {}
    raw_p = {}
    for label, tau in taus.items():
        w(f"### tau = {label}")
        w("")
        ep = threshold_table(curves, ladder, tau)
        w("| arm | per seed | reached | median |")
        w("|---|---|---|---|")
        for arm in sorted(ep):
            s = censored_summary(ep[arm])
            per = ", ".join("inf" if not np.isfinite(v) else f"{v:.0f}"
                            for v in ep[arm])
            med = f"{s['median']:.0f}" if s["median"] is not None else "-"
            w(f"| `{arm}` | {per} | {s['reached']}/{s['n']} | {med} |")
        w("")
        for base in BASELINES:
            if CHALLENGER not in ep or base not in ep:
                continue
            obs, p, exact = permutation_test_ranks(ep[CHALLENGER], ep[base])
            key = f"{CHALLENGER} vs {base} @ {label}"
            contrasts[key] = {"win_rate": obs, "p_raw": p, "exact": exact,
                              "tau": tau}
            raw_p[key] = p
        w("")

    adj = holm(raw_p) if raw_p else {}
    for k, v in adj.items():
        contrasts[k]["p_holm"] = v
        contrasts[k]["passes"] = bool(v < ALPHA and contrasts[k]["win_rate"] > 0.5)

    w("### Tests")
    w("")
    w("| contrast | win rate | permutation p | Holm p | verdict |")
    w("|---|---|---|---|---|")
    for k, c in contrasts.items():
        w(f"| {k} | {c['win_rate']:.2f} | {c['p_raw']:.4g}"
          f"{'' if c['exact'] else ' (MC)'} | {c['p_holm']:.4g} | "
          f"{'pass' if c['passes'] else 'fail'} |")
    w("")
    w("`win rate` is P(SAT reaches tau in fewer epochs than the baseline) over "
      "seed pairs, ties counted as half. 0.5 is the null.")
    w("")

    w("## Wall-clock to threshold")
    w("")
    w("Epochs are not free and the arms differ in seconds per epoch, so an arm "
      "needing half the epochs at twice the cost per epoch is not faster.")
    w("")
    spe = {arm: float(np.mean([curves[arm][s][ladder[-1]]["seconds_per_epoch"]
                               for s in curves[arm]])) for arm in curves}
    w("| arm | s/epoch | " + " | ".join(f"s to tau={l}" for l in taus) + " |")
    w("|---" * (2 + len(taus)) + "|")
    for arm in sorted(curves):
        cells = []
        for label, tau in taus.items():
            t = threshold_table(curves, ladder, tau, time_key="seconds_to_rung")
            s = censored_summary(t[arm])
            cells.append(f"{s['median']:.0f}s" if s["median"] is not None else "never")
        w(f"| `{arm}` | {spe[arm]:.1f} | " + " | ".join(cells) + " |")
    w("")

    w("## Area under the learning curve")
    w("")
    w(f"Mean cost across rungs, each capped at the random policy's "
      f"{random_cost:.1f} so one catastrophic early rung cannot dominate. Uses "
      f"every rung rather than only the crossing. Lower is better.")
    w("")
    aulc = capped_aulc(curves, ladder, random_cost)
    w("| arm | AULC (mean across seeds) | per seed |")
    w("|---|---|---|")
    for arm in sorted(aulc):
        w(f"| `{arm}` | **{aulc[arm].mean():.2f}** | "
          + ", ".join(f"{v:.2f}" for v in aulc[arm]) + " |")
    w("")

    w("## Verdict")
    w("")
    blocking = [g for g in gates if g["ok"] is False]
    if blocking:
        w("**No verdict.** These gates failed:")
        w("")
        for g in blocking:
            w(f"- {g['name']} -- {g['detail']}")
    elif contrasts and all(c["passes"] for c in contrasts.values()):
        w("**SAT reaches every tested quality threshold in fewer epochs than both "
          "original attention networks**, at each architecture's own learning "
          "rate. Holm-adjusted p < 0.05 and win rate > 0.5 for every contrast.")
    else:
        w("**Not established for every contrast.** Per-contrast outcomes are in "
          "the table above; the observed win rates and curves stand regardless.")
    w("")
    w("### What this does not show")
    w("")
    w("- **Not a claim about final quality.** `attention_net` reached 2.303 at 200 "
      "epochs in an earlier probe, so its ceiling is at least as good as SAT's. "
      "This measures the RATE, and a faster-converging architecture with an equal "
      "asymptote is exactly what the curves are expected to show.")
    w("- **The learning rate was selected at a 40-epoch budget**, a deliberately "
      "speed-oriented rule applied identically to all arms. Because each rung "
      "carries best-validation-so-far weights, a too-aggressive rate shows up as "
      "an early plateau rather than as damage at high budgets.")
    w("- **One MDP, one collection per arm, one architecture size.** Seeds vary "
      "weight init and shuffle order only.")
    w("- **Dropout is 0.1 in every arm's context encoder**, inherited from torch's "
      "`nn.TransformerEncoderLayer` default rather than chosen. It is a live "
      "hyperparameter nobody tuned, and it affects convergence.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="learning_curve")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load(args.tag)
    if not rows:
        raise SystemExit(f"no results under benchmark_results/{args.tag}")
    curves, ladder = organise(rows)
    gates = check_gates(rows, curves, ladder)
    report = render(args.tag, rows, curves, ladder, gates)
    print(report)
    out = args.out or os.path.join("benchmark_results", args.tag, "CURVES.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
