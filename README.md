# example-dynaplex-dot-attention

A [DynaPlex](https://dynaplex.github.io/DynaPlex/) example: a stochastic
vehicle-routing model on a grid, solved with neural-network policies whose
inputs are **tokens** (orders, vehicles, grid cells) rather than one flat
feature vector, and trained with dot-attention networks. It shows how a
DynaPlex featurizer emits several named tensors, how a custom PyTorch network
consumes that bundle, and how the same pipeline trains with Deep Controlled
Learning (DCL) or PPO — on a laptop or on a cluster node, unchanged.

The repository is a template: fork it, replace the model, keep the wiring.

## Quickstart

Requires **Python 3.11–3.14** (`requirements.txt` pins `dynaplex==1.14.0`,
which ships wheels for all four on Linux, macOS and Windows).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest          # a few seconds
.venv/bin/python visualize.py       # writes episode.html — open it in a browser
.venv/bin/python train.py dcl --net attention   # a small DCL run, then a comparison table
```

The visualization is a self-contained HTML player (play/pause, step, scrubber):
vehicles are triangles labeled `index:capacity` (orange = full, walking back to
the depot), waiting orders are red dots. Try `--policy random`, `--seed`,
`--periods`, or `--agent dynaplex_runs/.../agent_gen1` to watch a trained
policy — see `python visualize.py --help`.

## The model: dynamic vehicle routing on a grid

The "1-step decision" variant of the stylized dynamic vehicle routing problem
with stochastic customer requests from Akkerman, Mes & van Jaarsveld (2025),
*A comparison of reinforcement learning policies for dynamic vehicle routing
problems with stochastic customer requests*, Computers & Industrial Engineering
200, 110747 (`akkermans_model.pdf` in this repository).

A fleet of vehicles drives on a G×G grid; orders arrive at random cells and
cost `holding_cost` for every period they wait. Each period, every vehicle with
remaining capacity chooses its **next location**. Picking up an order takes one
unit of capacity, and a vehicle that runs out is skipped in the decision cycle
while it automatically walks back to the depot at (0,0), where its capacity
resets. At most `max_orders` orders wait in the system at once. The last two
rules are deliberate simplifications of the paper's model; the details and the
full period order are in the `mdp.py` docstring.

The action space comes in two **encodings** of that same decision
(`ActionEncoding`, a constructor choice — the dynamics are identical, only the
action's name differs, and trajectories are bit-identical across encodings
given the same seed):

- `CELL` (default): an action is a grid cell (G² actions), with validity
  masking admitting only the current cell and its in-grid neighbors, so at
  most 5 are ever available;
- `DIRECTIONAL`: an action is a move — 5 actions, stay/up/down/left/right,
  off-grid moves masked — the encoding the paper's NNPFA actually used.

Benchmarks: the paper's **myopic policy** (step toward the nearest waiting
order, ties random) implemented in `mdp.py`, and DynaPlex's generic
`RandomPolicy` (uniform over valid actions).

## Files

```
mdp.py                    the model: VehicleRoutingMDP (+ActionEncoding), State/Vehicle, MyopicPolicy
featurizers.py            count-map features (one flat vector) — the MLP baseline's input
featurizers_tokens.py     token bundles (orders/vehicles/depot, and all cells/vehicles/depot)
networks.py               the dot-attention networks (attention_net, attention_pointer_net, a PPO critic)
train.py                  train a policy: DCL or PPO, MLP or either attention network
ppo_attention.py          a tuned local PPO trial of the attention policy
visualize.py              episode simulation -> self-contained interactive HTML player
test_mdp.py               dynamics invariants, encoding equivalence, engine integration
test_featurizers.py       feature layout semantics + engine collection
test_featurizers_tokens.py token-bundle semantics, token/action order contract
snellius/                 venv setup + SLURM job script for the Dutch national cluster
```

## From state to tokens: the featurizers

A DynaPlex featurizer is a small dataclass whose `write_features(state)` is
compiled into the simulation engine, so observations are produced inside the
rollouts without a Python round trip. `featurizers_tokens.py` is the heart of
this example: it declares one **writer field per object type**, and each field
becomes a named tensor of the observation.

- `TokenFeaturizer` — `orders` [max_orders, 2] with an `orders_count` per
  sample (a *variable* number of tokens: the padding mask is derived from the
  count), `vehicles` [n_vehicles, 4] (position, remaining capacity, and a flag
  marking the deciding "focal" vehicle), `depot` [1, 2].
- `GridTokenFeaturizer` — every grid cell as a token, `cells` [G², 3]
  (position, has-order), written in row-major order, which is exactly the
  `CELL` action order; plus the same vehicle and depot tokens.

The module's long comment walks through what the `@featurizer` decorator
generates around `write_features`: the sizing spec, the holder that owns the
tensor slabs, the per-row reset/finish. That is the part to read when
adapting the example to a model of your own.

## The networks

Both nets in `networks.py` share one skeleton: each object type is embedded
into `d_model` by its own small MLP, the tokens run through a standard
`TransformerEncoder`, and only the readout differs.

- **`attention_net`** (with `TokenFeaturizer`, `DIRECTIONAL` encoding):
  tokens are the orders (masked past `orders_count`), the vehicles and the
  depot. The focal vehicle's output vector is read out into the 5 move logits.
- **`attention_pointer_net`** (with `GridTokenFeaturizer`, `CELL` encoding):
  a shared head scores each cell token, so token *i*'s score IS the logit of
  action *i*. The network never masks anything; DynaPlex's ordinary validity
  mask crops the logits downstream.

A network is handed to DynaPlex as a `dynaplex.nn.Net("networks.attention_net",
d_model=64, ...)`: an importable factory path plus keyword arguments, so a saved
agent rebuilds itself on any machine that has this repository on its path.

Open improvement, deliberately left for the next person: the pointer net embeds
all G² cells although at most 5 are ever valid actions — embedding only the
*action* cells should keep the pointer structure at a fraction of the compute.

## Training

```bash
.venv/bin/python train.py dcl                    # MLP on the flat features (the baseline)
.venv/bin/python train.py dcl --net attention    # order tokens + attention_net
.venv/bin/python train.py dcl --net pointer      # grid tokens + attention_pointer_net
.venv/bin/python train.py ppo --net attention    # PPO from scratch (actor + attention critic)
```

DCL is close to the paper's winning NNPFA on this problem variant: collect
states, label each with the best action found by rollouts, train a classifier
on the labels (`--generations 2` continues by rolling out the trained net
instead). Every run ends with a table of the trained agents against the myopic
and random benchmarks on common random numbers (lower cost is better).
Artifacts land under `dynaplex_runs/` and runs are idempotent: rerunning the
same configuration resumes instead of recomputing. See `python train.py --help`
for the budget and network knobs.

How a collection is spread over the machine (processes, engines, workers,
rollout slots, the device the network runs on) is not a knob of this repository:
DynaPlex's `machine="auto"` recognizes the machine it runs on — a laptop, or a
Snellius Genoa/H100/A100 allocation — and applies the plan engineered for it
(`python -m dynaplex.machine` prints what it sees). On a laptop with an Apple
GPU the trained policy's rollouts go to Metal from generation 1 on.

### At paper scale, on Snellius (or any SLURM cluster)

The paper-scale budget (10 000 labeled states, 2 000 rollouts of 100 periods
per candidate action, random generation-0 rollouts) is heavy for a laptop;
`snellius/` has the two files needed to run on the Dutch national cluster
(adapt the module line for other SLURM clusters):

```bash
git clone <this repo> && cd <repo>   # on a login node
snellius/setup.sh                    # venv + pinned deps (torch from the CPU index)
sbatch --time=04:00:00 snellius/train.job dcl --net pointer --rollout random \
    --n 10000 --m 2000 --h 100 --patience 15
squeue -u $USER                      # wait; output lands in results/
```

A full 192-core Genoa node sustains about 25M rollout periods per second on
this model at generation 0, so size jobs as `n * ~4.5 * m * h / 25e6` seconds
and set the walltime accordingly. With that budget and one DCL generation the
pointer net reached an average cost per period of about 2.15 against 3.10 for
the myopic benchmark, the attention net on order tokens 3.32 (measured with
DynaPlex 1.12; 1.14 warms every trajectory up before labeling, so expect
somewhat different numbers).

## Adapting the example

There is deliberately no `pyproject.toml` — this is a fork-and-adapt example,
not a package. Everything lives flat at the repository root and is run from
there; `requirements.txt` pins a known-good DynaPlex release. Model classes
compile into DynaPlex's engine (see the `PolicyComparer` test, `backend ==
"engine"`), so the MDP and featurizer code sticks to the DynaML subset — the
[language reference](https://dynaplex.github.io/DynaPlex/reference/language-reference/)
describes what is allowed.

To bring your own model: replace `mdp.py`, write a featurizer that emits the
tokens your problem has, keep or adjust the network heads, and point `train.py`
at the new classes.

## The structured-action transformer (SAT) integration

A second, in-progress line of work ports a **structured-action transformer** —
an AGILE-style relational net plus an *automatic tokenizer* — from
`dynaplex2_playground`. The idea: instead of hand-writing a featurizer, declare
the observation as dataclasses and derive the token layout from their fields and
type annotations.

**Phase 1** is a CPython spike, deliberately outside the engine; **Phase 2a**
puts the same architecture on the compiled engine under DCL.

> **[`MODULE_MAP.md`](MODULE_MAP.md)** maps every new module onto its existing
> counterpart (`featurizers_tokens.py` → `featurizers_sat.py`, `networks.py` →
> `networks_sat.py`, …), says what has no counterpart and why, and lists the open
> items. Start there if you knew this repo before the SAT work.

Phase 1 files:

| file | role |
|---|---|
| `sat_markers.py` | the annotation markers: `Action`, `GlobalState`, `OrderedList[T]`, `ActionKey[T]` |
| `sat_schema.py` | **the source of truth** — the VRP observation dataclasses, the `TOKEN_SCHEMA` table, and `build_sat_observation` |
| `sat_tokenizer.py` | the automatic tokenizer (dataclass reflection → token sequence) |
| `sat_net_ref.py` | the AGILE policy/value nets — the *reference oracle*, not the production net |
| `sat_actions.py` | `resolve_action`: group index → MDP action key |
| `sat_env.py` | vectorized CPython env over the MDP |
| `sat_trainer.py` | PPO + GAE(λ) over the above |
| `run_sat_spike.py` | train, then score against myopic in one calibrated loop |

```bash
.venv/bin/python run_sat_spike.py --quick   # ~1 min, checks the wiring
.venv/bin/python run_sat_spike.py           # ~35 min on a laptop
```

> **Proposal (2026-09-10): generate `featurizers_sat.py` from the schema.**
> The reflection the tokenizer does per state is a compile-time activity once
> the schema is fixed, and `@featurizer` already synthesizes methods as written
> source at import. [`FEATURIZER_GENERATION.md`](FEATURIZER_GENERATION.md)
> sketches generating `write_features`/`spec()` from `sat_schema` +
> `build_sat_observation`, the one annotation a list needs (its capacity), and
> the three builder idioms that do not compile on 1.14 as written.

### Why this can't just be dropped into the engine

The two batching models are structurally incompatible, and it is worth being
precise about why, because it shapes everything above:

| | this repo | the playground |
|---|---|---|
| batch object | preallocated fixed-shape numpy slabs `[B, *sample_shape]`, **shared with the C++ workers** | Python list of per-sample ragged "seq dicts" |
| shapes | static, declared once by `spec()` | discovered at runtime; the net grows via `extend_for_new_type` |
| padding | engine-side, at write time | inside `forward` |
| feature code runs | **transpiled to C++** (for/while/if/assert/assign only) | CPython reflection (`get_type_hints`, dynamic `getattr`, dicts) |

So `sat_tokenizer` can never run inside `write_features`. But its reflection is
really a *compile-time* activity — once the schema is known the layout is fixed
— and that is the seam Phase 2 exploits: make the layout static, and every piece
of integer metadata (`type_ids`, `action_group_ids`, `intra_group_pos`) becomes
derivable from the slab index and needs no storage at all. The only genuinely
dynamic quantity left is a variable token count, which `VariableObjectWriter`
already carries as its `_count` companion.

Two consequences are baked into the Phase 1 code as hard errors rather than
left as landmines: the token type universe is **closed** (a late-discovered type
raises instead of growing the registry), and a type whose feature width varies
per instance raises too. Both are exactly what a frozen `spec()` cannot express.

### Phase 1 result: inconclusive on the architecture, conclusive on the bottleneck

At 30,720 timesteps (~27 min of CPython), DIRECTIONAL encoding:

| policy | cost/period | ÷ myopic |
|---|---|---|
| myopic | 3.03 | 1.00 |
| random (valid actions) | 49.2 | 16.2 |
| **SAT/AGILE, sampled** | **≈ random** | — |
| **SAT/AGILE, greedy** | **90.6** | 29.9 |
| cap (`holding_cost × max_orders`) | 100.0 | 33.0 |

Read this carefully, because the greedy number is misleading. Policy entropy
after training is **1.50 against a uniform 1.61** — the logits are still nearly
flat, and sampling the policy scores essentially the same as random. But its
`argmax` picked only "left" and "right" on every state visited, so the greedy
readout turns a near-uniform policy into an arbitrary *constant* one: the vehicle
oscillates between two cells, never reaches an order, and cost climbs to the
`max_orders` cap. Hence 90.6.

So Phase 1 says nothing about whether AGILE helps on this problem — the policy
never got far enough to ask. What it does establish is the ceiling: **~19
timesteps/s**, against DCL refilling 97,280 slots per *round* across 190
compiled workers. That gap, not the cost figure, is the argument for Phase 2.
`run_sat_spike.py` now reports sampled and greedy side by side plus entropy and
the action histogram, since greedy-minus-sampled is what separates
"undertrained" from "architecture failed".

### Calibration, and what Phase 1 numbers mean

`PolicyComparer` refuses `backend="python"` for NN policies, so the spike scores
policies in its own CPython loop. That loop is calibrated, not just
self-consistent: at `episodes=32, warmup=100, horizon=500` it puts myopic at
**3.0815** against the engine comparer's **3.096**. Absolutes are therefore
meaningful at an adequate budget, but with tens of trajectories rather than 256 —
read the ratio as the headline. At a small budget the transient dominates (the
same myopic policy scores 4.39 at `warmup=20, horizon=60`).

### Phase 2a: the engine bridge

| file | role |
|---|---|
| `featurizers_sat.py` | `SATFeaturizer` — the same tokens, written into dense slabs by a **transpilable** `write_features` |
| `networks_sat.py` | `sat_agile_net` / `sat_agile_critic` — AGILE over the dense bundle, `[B, num_actions]` out |
| `run_sat.py` | DCL run, `--quick` / default / `--paper` budgets |

```bash
.venv/bin/python run_sat.py --quick   # ~1 min pipes check
.venv/bin/python run_sat.py          # laptop-scale learning run
.venv/bin/python run_sat.py --paper  # the paper budget of train.py (cluster)
```

The bridge works by making the token layout **static**, at which point all the
seq dict's integer metadata becomes derivable from position and needs no storage:

| seq dict field | in the dense layout | stored? |
|---|---|---|
| `type_ids` | which *tensor* the token lives in | no |
| `is_group_root` | "lives in `moves`" | no |
| `action_group_ids` | `arange(num_actions)` — `moves` token *i* **is** action *i* | no |
| `intra_group_pos` | all zero (VRP actions are single-token) | no |
| `indices_by_type` | the fixed concat order in `forward` | no |
| `num_action_groups` | `num_actions`, a constant | no |
| variable counts | `orders_count` | **yes** — the one dynamic int |

That is what makes float32-only writer slabs a non-issue rather than an obstacle.
The net drops three things `sat_net_ref` needs, each because the dense layout
makes them redundant rather than wrong: type embeddings (each type already has
its own projection, whose bias absorbs exactly that constant), the within-group
action layer (at group size 1 its attention is a softmax over one element), and
all action-side padding (the action set is written complete, so the `-inf` key
bias, the dummy pooling bin and the NaN guards all go).

**Throughput — the headline.** Gen-0 collection sustains **~2.7M rollout
periods/s on 16 laptop cores** (~25M on a Snellius Genoa node), against the
CPython spike's ~19 timesteps/s. The bottleneck has completely moved: collection
is now nearly free and CPU transformer *training* dominates (~48 s/epoch at
n=10000), which is why the laptop budget spends on label quality (`m`) and caps
epochs.

**Why DCL rather than PPO.** DCL labels each state with the best action found by
sequential-halving rollouts and trains a classifier, so there is no exploration
to collapse. The Phase 1 PPO failure said nothing about the architecture; this is
the run that can.

### Result: SAT/AGILE beats myopic

`python run_sat.py`, DIRECTIONAL, n=10000 / m=500 / h=100, 16 laptop cores,
comparer on 128 trajectories paired on common random numbers:

| policy | mean cost | delta vs myopic |
|---|---|---|
| **SAT/AGILE `agent_gen1`** | **2.826 ± 0.021** | **−0.299 ± 0.034** |
| myopic (this run) | 3.125 ± 0.034 | — |
| random | 52.8 ± 0.36 | +49.7 |

Against the published table (myopic 3.096 / flat MLP 4.07 / order-token attention
3.324) this is the **first architecture here to beat the myopic benchmark** — the
previous best attention net was still 7% worse than it. The paired delta is
−0.299 against a delta error of 0.034, about nine standard errors, so the sign is
not in doubt. And it got there on a *quarter* of the baselines' labelling budget
(m=500 against m=2000), with noisier labels.

Two honest caveats:

* **128 comparer trajectories, not 256.** Myopic measures 3.125 here against
  3.096 in the reference table, so read the paired delta rather than the
  absolute. A `--paper` run settles it.
* **The gain was confounded with the features.** SAT's action tokens carry
  `nearest_order_dist` and `has_order` per candidate cell — the myopic
  benchmark's own decision statistic — which the earlier attention nets did not
  get. **This is now resolved: see the ablation below.**

This also refutes the prediction in the planning document, which put AGILE at
3.2–3.4 and argued its premise was degenerate on this MDP because the action set
is a deterministic function of the focal vehicle's position. That reasoning was
wrong.

### Is SAT actually better? The matched benchmark

Everything above compares numbers produced at **different budgets, with one
training seed each**. SAT's 2.826 is `m=500` / 40 epochs / 128 comparer
trajectories; `attention_net`'s 3.324 is `m=2000` / 200 epochs / 256. And the
`+/-0.021` on 2.826 is the comparer's *trajectory* error, while the same arm has
scored 2.435, 2.663 and 2.826 across three runs -- seed spread ~0.2, roughly ten
times larger. So the table above shows a direction, not a result.

`run_benchmark.py` is the experiment that can settle it: all three architectures
re-run at ONE budget with ONE recipe over many training seeds, the statistical
unit being the seed rather than the trajectory, against a decision rule fixed in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before the run.

```bash
python run_benchmark.py --budget quick --seeds 2       # ~5 min, checks the wiring
python run_benchmark.py --budget long --seeds 8        # the real run, ~16 h
python analyze_benchmark.py --budget long              # gates, tests, RESULTS.md
```

> [!WARNING]
> **The first run of this benchmark produced a withdrawn result.** At
> `--budget default` (40 epochs) it concluded "SAT is better than both original
> attention networks". Nearly every run had hit the epoch cap, and re-training
> the baselines at 200 epochs on the same samples took `attention_net` from
> 3.028 / 23.400 down to **2.380 / 2.303** — better than every SAT seed measured
> at 40 epochs. Use `--budget long`; see
> [`benchmark_results/n10000m500h100/SUPERSEDED.md`](benchmark_results/n10000m500h100/SUPERSEDED.md)
> and §10 of [`PREREGISTRATION.md`](PREREGISTRATION.md). **The relative standing
> of the three architectures is currently an open question**, not something this
> repo has measured.

| file | role |
|---|---|
| `benchmark_config.py` | the arm table and the frozen recipe, as data |
| `run_benchmark.py` | trains every (arm, seed), persists raw per-trajectory returns |
| `benchmark_stats.py` | exact permutation test, bootstrap CIs, IQM, P(improvement), Holm |
| `analyze_benchmark.py` | validity gates, the pre-registered tests, the report |
| `PREREGISTRATION.md` | arms, endpoints, tests and decision rule, fixed in advance |

Six **validity gates** run before anything is interpreted, and a failure in a
non-advisory one blocks the verdict entirely -- the analysis prints the numbers
and refuses to conclude from them. The load-bearing one is the *CRN splice*: the
CELL-encoded pointer arm needs its own comparer call, so `myopic`'s
per-trajectory returns must come back element-wise identical from every call
before the two tables may be pooled. (They do; verified on real data.) The others
catch a collapsed policy, a drifted comparer, arms that hit the epoch cap instead
of early-stopping, and too few or unbalanced seeds.

Two facts the harness already establishes, independent of any training run:

* **`sat` and `tokens` are data-matched.** They collect under the same MDP with
  the same DCL seed, and their sample labels and validity masks hash identically,
  so sampling noise cancels exactly between those two arms.
* **The arms are not capacity-matched.** Same `d_model`/`n_layers`/`n_heads`/`ff`,
  but SAT's inter-action stack gives it 143,809 parameters against ~84,500 for
  either baseline. That stack is the architecture under test, so it cannot be
  removed without removing what is being measured -- which is exactly what the
  ablation below controls for.

### Ablation: the relational layer does most of the work

`python run_sat_ablation.py --seeds 2`. Both arms share ONE collected sample set
and the same training seed; the only difference is what sits between the action
nodes and the head — AGILE's complete-graph attention, or a **capacity-matched
MLP control** with the attention sublayer removed (same depth, FFN width,
residuals and norms). All six policies scored in a single comparer call, paired
on common random numbers.

| arm | cost | vs myopic |
|---|---|---|
| **attention, seed 0** | **2.435** | −0.689 |
| **attention, seed 1** | **2.663** | −0.462 |
| mlp control, seed 1 | 2.855 | −0.270 |
| mlp control, seed 0 | 3.023 | −0.102 |
| myopic | 3.125 | — |
| random | 52.8 | +49.7 |

Reading it:

* **The relational layer earns its keep.** Within-seed it is worth **0.587**
  (≥12σ) and **0.192** (≥4σ); mean effect **0.389**. The ordering holds 2/2, and
  the *worst* attention arm (2.663) still beats the *best* control arm (2.855),
  so the two sets do not overlap.
* **The features matter too, but less.** The control — same tokens, same
  features, no cross-action conditioning — still reaches 2.855–3.023, beating
  both myopic (3.125) and the old order-token baseline (3.324). So per-action
  tokens carrying `nearest_order_dist` are worth roughly 0.3–0.5 on their own,
  and the attention adds another ~0.4 on top.
* **Seed variance is real and large.** The attention arm spans 2.435–2.663 here
  and scored 2.826 in the earlier `run_sat.py` run. Weight init is a bigger lever
  than it looks — see the trap below — so treat the *effect* (paired, within
  seed) as the finding and any single absolute as one draw.

### Tests

`test_featurizers_sat.py` centres on the **parity oracle**: two independent
implementations of one layout now exist (reflection vs. dense writes), and the
failure mode that matters is column-order drift — invisible to every shape check,
rehearsal and compile. The comparison is asserted with exact `np.array_equal` on
float32 across 40 real states and both encodings, and it has been verified to
fail when a row/col swap is injected. `test_networks_sat.py` pins what the dense
rewrite changed: padding independence (scribble past `orders_count`, the output
must not move), order-permutation invariance, count-dtype invariance, mask
independence, and that an action's logit really does depend on the *other*
available actions (AGILE's defining property).

## Notes for adapting

There is deliberately no `pyproject.toml` — this is a fork-and-adapt example,
not a package. Everything lives flat at the repo root and is run from there;
`requirements.txt` pins a known-good dynaplex release. Model classes compile
into DynaPlex's engine (see the `PolicyComparer` test, `backend == "engine"`),
so the MDP code sticks to the DynaML subset — the
[language reference](https://dynaplex.github.io/DynaPlex/reference/language-reference/)
describes what is allowed.
