# Module map: the SAT integration against the existing code

Orientation for anyone who knew this repo before the structured-action
transformer (SAT) work landed. It answers three questions: which new file
corresponds to which old one, what genuinely has no counterpart, and what stayed
untouched.

Read `README.md` first for the model and the results. This file is about the
code layout.

---

## 1. The one-line version

The existing code has **two** generations of policy network. SAT is a third, and
the axis it moves along is **where the action space lives**.

| generation | how a state reaches the network | where actions live |
|---|---|---|
| flat MLP | state → 205 floats → MLP | the output layer (`num_actions` logits) |
| token attention | state → typed object tokens (orders, vehicles, depot) → transformer → read the focal vehicle's vector | the output layer, or one token per *grid cell* in the pointer variant |
| **SAT / AGILE** | state → typed **context** tokens **+ one token per action, each with its own features** → context encoder, then attention over `[state ; actions]` | **first-class tokens with their own feature vectors** |

Everything else — the MDP, the DynaML writer mechanism, DCL, the comparer — is
shared and unchanged.

---

## 2. Old → new correspondence

### Featurizers

| existing | new | relationship |
|---|---|---|
| `featurizers.py` — `VehicleFeaturizer`, a flat `2G²+V` = 205-float vector | *(none)* | untouched; still the flat-MLP baseline and the reference run in the engine cross-validation test |
| `featurizers_tokens.py` — `TokenFeaturizer` (orders/vehicles/depot), `GridTokenFeaturizer` (all G² cells/vehicles/depot) | `featurizers_sat.py` — `SATFeaturizer` | **same mechanism, one more token type.** Identical writer API (`FixedObjectWriter` / `VariableObjectWriter` + `_count` companion), identical DynaML restrictions, hand-written `spec()` for the same reason. What is added: a token per **action** (`moves`) carrying per-candidate features, and a `summary` context token |

`SATFeaturizer` is closest to `GridTokenFeaturizer`: both write one token per
action in action order, so token *i* scores action *i* and the validity mask
crops downstream. The difference is that under `DIRECTIONAL` it writes **5**
action tokens rather than 100 cell tokens — which is the improvement the README
had left "for the next person" — and each carries `has_order`,
`nearest_order_dist` and `on_grid` for its target cell rather than just
`(row, col, has_order)`.

### Networks

| existing (`networks.py`) | new (`networks_sat.py`) | relationship |
|---|---|---|
| `attention_net` — per-type embeddings → transformer → **focal vehicle's** output vector → 5 move logits (or a G² map) | `sat_agile_net` | the actor. Old: one readout vector produces all logits. New: each action is its own token, scored after attending over the other actions |
| `attention_pointer_net` — scores each of G² cell tokens with a shared head | `sat_agile_net` | the closer structural analogue — both score per-action tokens with a shared head. New: 5 tokens not 100, richer action features, plus inter-action attention |
| `attention_critic` — same trunk, focal vector → scalar | `sat_agile_critic` | same role (PPO's built-in value head is single-tensor, the bundle is not). New: reads the **state node** after it has attended over the available actions |

### Runners

| existing | new | relationship |
|---|---|---|
| `train.py --net attention` / `--net pointer` (formerly `run_attention.py` / `run_attention_grid.py`) — Snellius DCL, `n=10000, m=2000, h=100` | `run_sat.py --paper` | deliberately the **same budget**, so the numbers compare directly |
| `train.py --quick` (formerly `pipes_check.py`) — laptop DCL smoke | `run_sat.py --quick` | same purpose, folded into a flag |
| `ppo_attention.py` — local PPO trial | `ppo_sat.py` | same `PPOConfig`, so the two are comparable |
| *(none)* | `run_sat_ablation.py` | new: the feature-vs-architecture experiment (§6). Trains `inter_action="attention"` against the capacity-matched `"mlp"` control on **one shared sample set**, then scores both in a single paired comparer call |
| *(none)* | `run_benchmark.py` + `benchmark_config.py` + `benchmark_stats.py` + `analyze_benchmark.py` | new: the **matched head-to-head** (§10). Re-runs all three architectures at one budget over many training seeds, because every number in §6 was a single seed measured at a different budget |
| `train.py` — CLI over the flat featurizer | *(none)* | untouched |
| `visualize.py` — episode → HTML | *(works unchanged)* | it drives an agent through the CPython featurizer path, which `SATFeaturizer` supports |

### Tests

| existing | new | what the new ones add |
|---|---|---|
| `test_mdp.py` (10) | *(none)* | untouched |
| `test_featurizers.py` (3) | *(none)* | untouched |
| `test_featurizers_tokens.py` (5) | `test_featurizers_sat.py` (14) | same idioms — `assert_featurizer_for_mdp`, engine cross-validation against the flat run — plus the **parity oracle** (§4) |
| — | `test_networks_sat.py` (17) | padding independence, permutation invariance, count-dtype invariance, mask independence, AGILE conditioning |
| — | `test_sat_tokenizer.py` (21), `test_sat_net_ref.py` (17), `test_sat_env.py` (16) | the Phase 1 layer (§3) |

103 tests total; the 18 pre-existing ones are unmodified and still pass.

---

## 3. What has no counterpart: the Phase 1 layer

These files implement something the repo did not previously have — **deriving a
token layout from a declared schema instead of hand-writing a featurizer** — and
they are all CPython-only, deliberately outside the engine.

| file | lines | role |
|---|---|---|
| `sat_markers.py` | 66 | the annotation markers: `Action`, `GlobalState`, `OrderedList[T]`, `ActionKey[T]` |
| `sat_schema.py` | 398 | **the source of truth.** The VRP observation dataclasses, the `TOKEN_SCHEMA` table (names, widths, max counts, column order), and `build_sat_observation` |
| `sat_tokenizer.py` | 640 | the automatic tokenizer: walks a dataclass tree by type introspection → token sequence |
| `sat_actions.py` | 122 | `resolve_action`: policy group index → MDP action key, via `ActionKey[T]` |
| `sat_net_ref.py` | 824 | the AGILE policy/value nets, copied **verbatim** from the playground — the reference oracle, *not* the production net |
| `sat_env.py` | 277 | vectorized CPython env over the MDP |
| `sat_trainer.py` | 991 | PPO + GAE(λ) over the above |
| `run_sat_spike.py` | 303 | train, then score against myopic in one calibrated loop |

**Why keep them now that Phase 2a works?** Three reasons.

1. `sat_schema.py` is the naming and ordering authority `featurizers_sat.py` was
   transcribed from, and what the parity test compares against.
2. `sat_tokenizer.py` is an executable specification of the token layout. It is
   how you check the static featurizer is right.
3. `sat_net_ref.py` is the reference semantics for the batched/masked attention
   that `networks_sat.py` simplified away.

They are not on the training path — and the independence is total: **no Phase 2a
module imports any of them**, `sat_schema` included (verified; see §4). Their
influence is exercised only from the test suite. That is what makes them safe to
keep and dangerous to trust: nothing at runtime will notice if they drift out of
agreement with the featurizer.

---

## 4. The structural reason there are two implementations

`sat_tokenizer` cannot ever run inside `write_features`. DynaPlex transpiles a
restricted Python subset (`for`, `while`, `if`, `assert`, assignment) into the C++
engine; the tokenizer needs `get_type_hints`, dynamic `getattr`, dicts of Python
objects. And there is no fallback: DCL collection and the comparer's NN harness
are engine-only (`policy_comparer.py:715-717` refuses `backend="python"` for NN
agents).

The bridge is that reflection is a **compile-time** activity. Once the schema is
fixed, the layout is fixed, and every piece of integer metadata the tokenizer
produced becomes derivable from position:

| tokenizer output | dense equivalent | stored? |
|---|---|---|
| `type_ids` | which *tensor* the token lives in | no |
| `is_group_root` | "lives in `moves`" | no |
| `action_group_ids` | `arange(num_actions)` — token *i* **is** action *i* | no |
| `intra_group_pos` | all zero (VRP actions are single-token) | no |
| `indices_by_type` | the fixed concatenation order in `forward` | no |
| `num_action_groups` | `num_actions`, a constant | no |
| variable counts | `orders_count` | **yes** — the one dynamic int |

That is what makes float32-only writer slabs a non-issue rather than an obstacle:
there are no integers left to store.

**The cost of two implementations is column-order drift** — swap two `append`
lines and every shape still validates, the rehearsal passes, the engine compiles,
and the network trains on permuted features.

And note precisely how weak the coupling is: **`featurizers_sat.py` does not
import `sat_schema.py`.** Its imports are `dataclasses`, `dynaplex.modelling` and
`mdp`, nothing else — deliberately, so the module the transpiler compiles carries
no dependency on the tokenizer layer. Its `spec()` hardcodes the widths
(`(1,4)`, `(1,2)`, `(n_vehicles,4)`, `(max_orders,2)`, `(num_actions,6)`) rather
than reading them from `TOKEN_SCHEMA`. So the table is the authority **by
convention and by test, never by import**:

* `test_featurizers_sat.test_spec_widths_match_the_schema_table` — the declared
  widths still match the table;
* `test_featurizers_sat.test_dense_slabs_match_the_tokenizer_exactly` — every
  column of every token, exact `np.array_equal` on float32, 40 real states, both
  encodings. Verified to fail when a row/col swap is injected;
* `sat_schema.validate_schema()` (at import) — the table still matches its own
  dataclasses.

Those three tests are the only thing holding the two implementations together.
Treat them as load-bearing, not as coverage.

---

## 5. Where `networks_sat` differs from `sat_net_ref`

`networks_sat.py` is not a port of the AGILE paper; it is a dense rewrite of
`sat_net_ref.py`, which is itself the playground's interpretation of AGILE.
Preserved: the defining mechanism — action nodes plus a state node in a complete
graph, so each action's representation depends on which others are available.

| | `sat_net_ref` | `networks_sat` | consequence |
|---|---|---|---|
| type embeddings | separate `nn.Embedding` added after projection | dropped | **exactly neutral** — the reference computes `W_t x + b_t + e_t`, and `b_t + e_t` is one learnable vector per type, which `b_t` already parameterizes |
| within-group action layer | `num_layers` residual blocks over action tokens | dropped | **a real architectural difference.** At group size 1 the reference's own docstring says it "collapses to a per-token MLP", so this removed capacity, not just dead code. Re-add it for an MDP with multi-token actions |
| action-side padding | `-inf` key bias, `max_G+1` dummy pooling bin, `nan_to_num` | dropped | neutral *for VRP only* — the action set is written complete |
| context tokens | all non-action tokens → encoder → mean-pool → 1 state node | **identical** | 37 real context tokens either way on a default state |

There is **no proof of functional equivalence** between the two, deliberately: a
weight-copy parity test would have required keeping the within-group layer and
folding the embeddings, i.e. distorting the shipped net so a test could exist.
What exists instead is behavioural invariants. If fidelity to the paper matters,
that gap is where to start.

---

## 6. Results

| policy | mean cost (lower better) | source |
|---|---|---|
| **SAT/AGILE `agent_gen1`** | **2.826 ± 0.021** | `run_sat.py`, n=10000 / m=500 / h=100, 128 trajectories |
| myopic benchmark | 3.096 | reference table |
| order-token attention (`attention_net`) | 3.324 | `train.py --net attention` (then `run_attention.py`) |
| flat MLP | 4.07 | `train.py dcl`, n=50000 |

Paired against myopic in the same run: **−0.299 ± 0.034**, about nine standard
errors, on a *quarter* of the baselines' labelling budget.

Two caveats, both open:

* **128 comparer trajectories, not 256.** Myopic measures 3.125 in that run
  against 3.096 in the reference table, so the paired delta is the reliable
  statistic. A `--paper` run settles it.
* **The gain is confounded with the features.** SAT's action tokens carry
  `nearest_order_dist` — the myopic benchmark's own decision statistic — which
  the earlier attention nets never received. So this may be better action
  features rather than the relational layer.

`run_sat_ablation.py` resolved the second. It trains `inter_action="attention"`
against `inter_action="mlp"` — the capacity-matched control that keeps depth, FFN
width, residuals and norms but removes the attention sublayer. Three things are
held identical: the samples (collected once, copied into the second arm's workdir
so both fit byte-identical labels), the training seed, and the evaluation (one
paired comparer call).

**Result (`--seeds 2`): the relational layer does most of the work, and the
features do the rest.**

| arm | cost | vs myopic |
|---|---|---|
| attention, seed 0 | **2.435** | −0.689 |
| attention, seed 1 | **2.663** | −0.462 |
| mlp control, seed 1 | 2.855 | −0.270 |
| mlp control, seed 0 | 3.023 | −0.102 |
| myopic | 3.125 | — |

Within-seed the attention is worth 0.587 (≥12σ) and 0.192 (≥4σ), mean 0.389, and
the worst attention arm still beats the best control arm. But the control alone
already beats myopic and the old 3.324 baseline, so the per-action tokens and
their `nearest_order_dist` feature are worth ~0.3–0.5 independently.

### Three traps this experiment exposed

Worth reading before running any variant of it.

1. **`train_network(seed=...)` does not control weight initialisation.** It builds
   the module *before* calling `torch.manual_seed(seed)` (`dynaplex/nn.py`, lines
   ~52 and ~56), so init draws from ambient global RNG state. Two identical
   `--quick` invocations produced 47.10 and 64.60 for the *same* arm.
   `run_sat_ablation.py` therefore calls `torch.manual_seed` itself before each
   arm, and defaults to comparing across seeds. A one-seed architecture
   comparison here measures init luck as much as architecture.
2. **Explicit `workdir=` bypasses DCL's structural hashing.** That is what lets
   two arms share a sample set, but it also means DCL resumes from whatever it
   finds — a `--quick` run followed by a full run at the same path would train on
   the quick run's 1,000 samples while reporting the 10,000 budget. Workdirs are
   budget-tagged (`n10000m500h100/`) to make the collision impossible.
3. **Undertrained arms can be byte-identical and look like a harness bug.** At a
   small budget the logits are near-flat, the greedy argmax collapses onto one
   arbitrary action, and two different networks can collapse onto the *same*
   one — producing identical comparer output. `degeneracy()` runs before the
   comparer and labels any arm above 90 % dominance as uninformative.
4. **The MDP mutates `State` in place.** Building a batch with
   `states.append(state)` inside a stepping loop yields N aliases of the FINAL
   state, not N distinct states — snapshot with `copy.deepcopy` on append. This
   bit every `_states()` / `_real_batch()` test fixture here and the first
   version of `degeneracy()`, which consequently reported every agent as 100 %
   degenerate (one state gives one action, trivially dominant). Fixing it also
   exposed a genuine test defect: `test_logits_are_finite_on_inadmissible_actions`
   asserted every state has an inadmissible action, which is false under
   DIRECTIONAL for a vehicle in the grid interior — it only passed because the
   aliased state happened to sit on a boundary.

---

## 7. Throughput, and why Phase 1 exists at all

| path | rate |
|---|---|
| Phase 1 CPython spike (`run_sat_spike.py`) | ~19 timesteps/s |
| Phase 2a DCL collection, 16 laptop cores | ~2.9M rollout periods/s |
| Phase 2a DCL collection, 190-worker Genoa node | ~25M rollout periods/s |

The Phase 1 PPO spike never trained a usable policy — sampled, it scored about
the same as random; read greedily, its near-flat logits became an arbitrary
constant policy and cost climbed to the `max_orders` cap. **That result says
nothing about the architecture**, only about the budget, and it is exactly why
Phase 2a was built. Do not cite it as evidence either way.

The bottleneck has since inverted: collection is nearly free and CPU transformer
*training* dominates (~48 s/epoch at n=10000). On a cluster, a `--paper` run's
cost is almost entirely the 200 training epochs, not the simulation — the
opposite of how the attention runs' budget was originally sized.

---

## 8. Modified existing files

| file | change |
|---|---|
| `README.md` | SAT sections, the results table, the calibration note |
| `.gitignore` | `sat_spike_*.pt` (spike checkpoints) |
| `snellius/train.job` | a first argument ending in `.py` now runs that script instead of `train.py`, so `sbatch snellius/train.job run_sat.py --paper` works. The documented `train.py` interface is unchanged |

Nothing in `mdp.py`, `featurizers.py`, `featurizers_tokens.py`, `networks.py`,
`train.py`, `visualize.py` or the three original test files was touched.

---

## 9. Known open items

* ~~**ablation B** — needs its own sample set (the featurizer changes), so it
  cannot share one with ablation A~~ **Wrong, and now implemented.** The
  featurizer never had to change: `sat_agile_net(action_features="geometry")`
  zeroes `moves[..., 4:6]` inside `forward`, which is *exactly* equivalent to a
  featurizer that never wrote those columns — a zero input contributes nothing to
  the output and its weights receive no gradient (both halves asserted in
  `test_run_benchmark.py`). So ablation B runs on the **same** sample set as the
  full net, with byte-identical labels. Available as the `sat_nofeat` arm:
  `python run_benchmark.py --arms sat,tokens,pointer,sat_nofeat`
* a `--paper` run for a 256-trajectory number directly comparable to 3.324
* `--generations 2`, letting DCL roll out the trained net
* `max_orders` is 100 but typical occupancy is ~30, so the context sequence is
  107 padded tokens against 37 real — the encoder does ~8.4x more attention than
  needed and the `orders` slab is 70% padding (~54 MB of 78 MB at DCL's 97,280
  slots). Lowering it is a modelling call; the writer's overflow assert would
  catch a bad choice loudly
* whether the playground's AGILE port — and therefore this one — is faithful to
  the paper is unverified (`reference_papers/agile_vs_sat_comparison.md` in the
  playground)
* Phase 2b codegen (emitting `write_features` from `TOKEN_SCHEMA`) is deliberately
  **not** done: at one MDP the generator would be ~400 lines to replace ~40

---

## 10. The matched head-to-head

§6's table has a hole that is worth being blunt about: **every number in it is one
training seed, and the arms were measured at different budgets.** SAT's 2.826 is
`m=500` / 40 epochs / 128 comparer trajectories; `attention_net`'s 3.324 is
`m=2000` / 200 epochs / 256. Four knobs differ. And the `±0.021` is the comparer's
*trajectory* error, while the SAT arm alone has scored 2.435, 2.663 and 2.826 —
a seed spread of ~0.2, an order of magnitude larger than the error bar being
quoted. The statistical unit was wrong.

| file | role |
|---|---|
| `benchmark_config.py` | the arm table and the frozen recipe, as data, so the harness and the analysis cannot drift apart |
| `run_benchmark.py` | trains every (arm, seed); persists raw per-trajectory returns as JSON. Idempotent, so `--seeds` can be raised later without recomputing |
| `benchmark_stats.py` | exact permutation test, bootstrap CIs, IQM, P(improvement), Holm. Numpy only — no new dependency |
| `analyze_benchmark.py` | the six validity gates, the pre-registered tests, and the only thing allowed to state a verdict |
| `PREREGISTRATION.md` | arms, endpoints, tests, decision rule — fixed before the run |

Three things it does that nothing here did before:

1. **The seed is the unit.** K=8 per arm, exact permutation test over all
   C(16,8)=12,870 splits, bootstrap CI on the mean difference, P(improvement),
   Holm across the two contrasts. K=5 is the floor — below that the smallest
   attainable exact p is 0.0079 and a Holm correction cannot be cleared.
2. **The gates can withhold the verdict.** A collapsed policy, a drifted
   comparer, unbalanced seeds, or a broken CRN splice each block the conclusion
   rather than quietly degrading it. The load-bearing one is the splice: the
   CELL-encoded pointer arm needs its own comparer call, so `myopic`'s
   per-trajectory returns must come back element-wise identical from every call
   before the tables may be pooled. Verified on real data — they do.
3. **The numbers stop being prose.** Every result in this file and the README was
   transcribed by hand from stdout. `benchmark_results/<tag>/*.json` holds the raw
   returns, and `RESULTS.md` is generated.

Two facts it establishes without training anything:

* **`sat` and `tokens` are data-matched.** Same MDP, same DCL seed, and their
  sample labels and validity masks hash identically — so sampling noise cancels
  exactly between those two arms.
* **The arms are NOT capacity-matched.** Identical `d_model`/`n_layers`/`n_heads`/
  `ff`, but the inter-action stack gives `sat` 143,809 parameters against 84,613
  (`tokens`) and 84,417 (`pointer`) — 1.7x. That stack is the architecture under
  test, so it cannot be removed without removing what is being measured; §6's
  `"mlp"` control is the experiment that separates size from structure.
