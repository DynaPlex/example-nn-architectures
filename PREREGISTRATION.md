# Pre-registration: is `sat_agile_net` better than the original attention nets?

Fixed **before** the run. `analyze_benchmark.py` implements exactly what is
written here; if the analysis and this document ever disagree, this document is
the specification and the code is the bug.

The point of writing it down first is narrow and worth stating plainly: with
three arms, several statistics and a handful of seeds, there are enough degrees
of freedom to find a favourable framing after the fact for almost any outcome.
Committing to the arms, the endpoint, the test and the rule in advance is what
makes the result confirmatory rather than a search.

---

## 1. The question

At one matched budget on the DVRP of `mdp.py`, does a policy trained with
`networks_sat.sat_agile_net` reach a lower average cost per period than one
trained with `networks.attention_net` or `networks.attention_pointer_net`?

Explicitly **not** asked here: *why* (features vs relational layer — that is
`run_sat_ablation.py` plus the optional `sat_nofeat` arm), whether it holds at
other budgets or MDP configurations, and whether it is worth its extra compute.

## 2. Why the numbers already on record cannot answer it

| | SAT (`run_sat.py`) | `attention_net` (`train.py --net attention`) |
|---|---|---|
| rollouts per sample `m` | 500 | 2000 |
| training epochs | 40 (patience 8) | 200 (patience 15) |
| comparer trajectories | 128 | 256 |
| training seeds | 1 | 1 |

Four knobs differ. And the reported `±0.021` on SAT's 2.826 is the comparer's
*trajectory* standard error, while the same arm has scored 2.435, 2.663 and 2.826
across three runs — a seed spread of ~0.2, an order of magnitude larger. The unit
of analysis was wrong. `attention_pointer_net` has no published number at all.

## 3. Arms

Defined in `benchmark_config.py`. Only the (featurizer, network, encoding) triple
varies.

| arm | featurizer | network | encoding |
|---|---|---|---|
| `sat` | `SATFeaturizer` | `networks_sat.sat_agile_net` | DIRECTIONAL |
| `tokens` | `TokenFeaturizer` | `networks.attention_net` (`head="moves5"`) | DIRECTIONAL |
| `pointer` | `GridTokenFeaturizer` | `networks.attention_pointer_net` | CELL |

Held identical: the MDP and its defaults; `d_model=64, n_layers=2, n_heads=4,
ff=128, hidden_head=64`; DCL `n=10000, m=500, h=100, slots=256`, collection seed
0, one generation; `epochs=40, batch_size=64, lr=1e-4, patience=8`; comparer
`256` trajectories, `warmup_time=100`, `horizon=500`, seed 42, benchmark
`myopic`.

This is **not** the baselines' published `m=2000` / 200-epoch budget, which is
~2.7 h per arm-seed and does not fit a seed sweep on 16 cores. All three arms are
re-run at the affordable budget, so absolute numbers here will not match the
3.324 on record. That is fine: the point is that the arms match *each other*.

`sat_nofeat` (SAT with `has_order` and `nearest_order_dist` zeroed at the network
input) is available but **not** part of the primary analysis.

## 4. Seeds

**K = 8 training seeds per arm**, seeds `0..7`, the same set for every arm.

Sizing: seed sd ≈ 0.2 and the effect suggested by the existing single-seed
numbers ≈ 0.5–0.7, i.e. Cohen's *d* ≈ 3. At K=8 an exact permutation test
enumerates C(16,8) = 12,870 splits and can reach p ≈ 1.6e-4. **K = 5 is the
floor** — below that the smallest attainable exact p is 0.0079 and at K=4 it is
0.029, so the test loses the ability to clear a Holm correction.

Within an arm all seeds train on **one shared collected sample set**, so the seed
sweep measures weight-init and shuffle variance on a fixed dataset. The claim
therefore generalises over training runs, not over collections.

`torch.manual_seed(seed)` is called **before** the DCL object is constructed:
`train_network` builds the module before calling `torch.manual_seed`
(`dynaplex/nn.py:404` vs `:408`), so weight initialisation is otherwise drawn
from ambient global RNG state and two identical invocations diverge.

## 5. Endpoints

**Primary:** mean cost per period from `PolicyComparer`, lower is better. One
value per (arm, seed).

**Secondary, reported but not part of the decision rule:** best validation loss
and estimated validation cost regret from `train_network` (same rows, since the
validation split is seeded identically and — via gate 3 — the underlying samples
match); per-arm seed sd; greedy-action dominance; parameter count; training
wall-clock.

## 6. Validity gates

Checked before anything is interpreted. A failure in a non-advisory gate blocks
the verdict entirely.

1. **CRN splice.** `myopic`'s per-trajectory returns must be element-wise
   identical across every comparer call, including across the two encodings. This
   is what licenses pooling the CELL and DIRECTIONAL tables, and evaluating seeds
   added later in a separate call.
2. **Anchor reproduction.** `myopic` in [3.0, 3.25] and `random` in [48, 58] in
   every call, matching the reference table. Checked only at budgets whose
   comparer config matches the one those reference values were measured at
   (`warmup_time=100, horizon=500`); the `quick` budget reports this gate as
   not-checkable rather than being given an invented range. Note also that
   `random` is **not** comparable across encodings -- uniform-over-valid draws
   from the policy RNG differently under CELL and DIRECTIONAL (40.04 vs 39.13 in
   one smoke run) -- so only `myopic` is used as the splice probe in gate 1.
3. **Data matching** *(advisory)*. `sat` and `tokens` collect under the same MDP
   with the same DCL seed, so their sample labels and validity masks should hash
   identically. If they do, sampling noise cancels between those two arms; if
   not, they are only budget-matched and the write-up must say so.
4. **Degeneracy.** No arm-seed above 90 % greedy-action dominance. Above that the
   net has collapsed to a constant policy and its cost reflects the budget, not
   the architecture.
5. **Budget truncation** *(advisory)*. Runs should stop on `patience`, not on the
   epoch cap. An arm that systematically hits the cap was still improving, and
   the budget capped it rather than the architecture.
6. **Balance.** Equal seed count across arms, at least 5.

## 7. Statistical analysis

Unit: the training seed. Arms are treated as **unpaired** across seeds — different
architectures consume weight-init RNG differently, so "seed 3" is a label, not a
common random number. (Pairing is real at the *data* level, via gate 3.)

* **Primary test:** two-sided exact permutation test on the difference of arm
  means, all C(16,8) splits enumerated. Chosen over Welch's *t* because K is
  small, the per-seed distribution is skewed by bad inits, and normality is not
  checkable at this K.
* **Effect size:** mean difference with a percentile bootstrap 95 % CI
  (20,000 resamples, each group resampled independently).
* **Robustness:** interquartile-mean difference with the same bootstrap, and
  **probability of improvement** P(challenger < baseline) with a bootstrap CI
  (Agarwal et al. 2021).
* **Dominance envelope:** whether the challenger's worst seed beats the
  baseline's best seed. Reported as a fact, not a test.
* **Multiplicity:** two primary contrasts — `sat` vs `tokens` and `sat` vs
  `pointer` — Holm-Bonferroni adjusted.

## 8. Decision rule

**"SAT is better" is declared only if, for BOTH contrasts:**

1. Holm-adjusted permutation p < 0.05, **and**
2. the 95 % bootstrap CI on the mean difference lies entirely below 0, **and**
3. the lower bound of the P(improvement) CI exceeds 0.5.

Anything else is reported as **"not established"**, together with the observed
direction and magnitude. "Not established" is a statement about the evidence, not
a claim that the architectures are equal.

If SAT loses either contrast, that is the result and it gets written up as the
result.

## 9. Known limitations, agreed in advance

* **The comparison is confounded with features.** SAT's action tokens carry
  `nearest_order_dist` and `has_order` — the myopic policy's own decision
  statistic — which neither baseline receives. A win is a win for the *whole
  design*, not for the relational layer.
* **No arm is tuned.** All three inherit the `lr=1e-4` recipe the existing run
  scripts use. An architecture that would prefer a different rate is penalised.
* **One MDP configuration, one budget.**
* **The arms are not capacity-matched.** The transformer knobs are identical,
  but SAT's inter-action stack gives it ~1.7x the parameters of either baseline
  (143,809 vs ~84,500, measured). That stack *is* the architecture under test, so
  it cannot be removed without removing the thing being measured -- but a win
  here is also a win for a larger model, and `run_sat_ablation.py`'s
  capacity-matched MLP control is what separates the two.
* **Compute is not part of the rule.** SAT's featurizer is materially more
  expensive per state (`_nearest` scans the grid once per action token).

## 10. Deviations

Any departure from this document gets appended here, dated, with the reason —
before the affected numbers are reported.

### 2026-08-19 — the degeneracy probe was measuring itself

**What changed.** `run_benchmark.degeneracy()` now steps the MDP with a uniformly
random *admissible* action instead of a hardcoded action 0.

**Why.** The original (inherited from `run_sat_ablation.py:96-103`) was invalid as
an instrument, and this was established from the probe's own behaviour rather than
from the results it produced:

* Under DIRECTIONAL, action 0 is "stay", so the fleet never moved. Under CELL,
  action 0 is cell (0,0) (`mdp.py:203-205`), which is additionally **inadmissible**
  unless the vehicle is beside the depot — measured taking an invalid action in
  **5 of 60** probe states.
* Either way the 60 probe states covered only **~5 distinct focal positions**, so
  the reported "dominance" described the probe's state distribution, not the
  network.
* The tell: all eight independently initialised `pointer` seeds scored **exactly
  91.7 %** (55/60). Eight different networks cannot agree to three decimals unless
  the measurement has no resolution.

Re-measured with a random-admissible walk (matching `dp.RandomPolicy`, which is
the distribution DCL collects generation 0 from), `pointer` reads **18.3 %** with
its greedy mass spread over ~25 distinct cells. It was never degenerate.

**Integrity notes.** This deviation was made *after* the run and it *unblocks a
gate that was withholding the verdict*, so three things are worth stating plainly:

1. It re-measured **every** arm, not the one it unblocked: `sat` moved from a flat
   40.0 % to 33-52 % and `tokens` from 40-68 % to 35-50 %. The old column was
   wrong everywhere.
2. It changed **no cost, no weights, no samples and no comparer output.** Only the
   two `degeneracy_*` fields were rewritten, from the already-trained agents. The
   superseded values are preserved as `degeneracy_share_old_probe` in each result
   JSON.
3. The decision rule, the arms, the seeds, the endpoint and the tests are
   unchanged.

Guarded going forward by `test_degeneracy_probe_never_takes_an_inadmissible_action`
and `test_degeneracy_probe_visits_a_spread_of_states`, both parametrized over the
two encodings.

### 2026-08-22 — the `default` budget result is withdrawn (epoch cap)

**What happened.** Gate 5 (budget truncation) reported that 7/8 `pointer`, 8/8
`tokens` and 7/8 `sat` runs hit the 40-epoch cap. It is an *advisory* gate, so the
verdict was issued anyway: "SAT is better than both original attention networks at
this budget."

A probe at `--budget long` (identical samples, `epochs=200, patience=15`) shows
that verdict was measuring the cap:

| arm | seed | 40 epochs | 200 epochs | epochs used |
|---|---|---|---|---|
| `tokens` | 0 | 3.028 | 2.380 | 82 |
| `tokens` | 1 | 23.400 | 2.303 | 89 |
| `pointer` | 1 | 50.779 | 2.501 | 110 |
| `pointer` | 0 | 24.623 | 28.150 | 104 |

All four early-stopped between 82 and 110 epochs, so 40 was 2–3× too short, and at
200 epochs `tokens` beats every `sat` seed measured at 40.

**Consequences for this document.**

1. The `default`-budget verdict is **withdrawn**. See
   `benchmark_results/n10000m500h100/SUPERSEDED.md`.
2. **Gate 5 is promoted from advisory to blocking**, and the threshold is
   explicit: if any arm has more than half its seeds hitting the epoch cap, no
   verdict is issued. An advisory gate cannot guard an assumption the whole
   comparison rests on — that was the design error, and it is the same class of
   mistake as the degeneracy probe: a check that was present, fired correctly, and
   was not given the authority to stop the conclusion.
3. **The primary budget becomes `long`** (`n=10000, m=500, h=100`, 200 epochs,
   patience 15) — the smallest budget observed to let every arm converge.

**What is NOT claimed.** `sat` has not been run at 200 epochs. Nothing here says
SAT is better or worse than `attention_net`; it says the earlier answer was
invalid. The observation that `sat`'s seeds were already tight at 40 epochs while
the baselines needed 82–110 is a claim about *convergence speed*, untested
directly.

**Also retracted:** the "regret cliff" reading (cost as a threshold function of
validation regret near 2.8). `pointer` seed 0 sits at regret 2.311 and costs
28.150; `tokens` seed 1 sits at 2.661 and costs 2.303. Validation regret does not
determine closed-loop cost.
