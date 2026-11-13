# example-nn-architectures

A [DynaPlex](https://dynaplex.github.io/DynaPlex/) example about **neural
network architectures for sequential decision problems**: one stochastic
vehicle-routing model on a grid, one training recipe, and three ways to present
the state to a network — a flat feature vector, a graph, a set of tokens for
the candidate moves. The point is to see
what changes when the architecture changes, with everything else held fixed.

| architecture | the network sees | featurizer | network |
|---|---|---|---|
| flat MLP | two G×G count maps and the capacities, as one vector | `featurizer_flat.py` | DynaPlex's default MLP |
| GIN | one node per cell on the grid graph, message passing along the lattice | `featurizer_graph.py` | `networks.gin_net` |
| move tokens | one token per admissible move, attending to order, vehicle and depot tokens | `featurizer_moves.py` | `networks.move_token_net` |

The GIN starts from the "Node Actions" architecture of Begnardi, Baier, van
Jaarsveld & Zhang (2023), *Deep Reinforcement Learning for Two-sided Online
Bipartite Matching in Collaborative Order Picking*, ACML, and adds what a
one-step move action needs on top of a destination action (see Results). The
move-token net
is the structured-action idea of AGILE (Jain et al.): score each available
action as a token; its `interact` switch turns candidate-to-candidate
attention off, which makes the relational part of that design measurable.

## Quickstart

Requires **Python 3.11–3.14** (`requirements.txt` pins `dynaplex==2.0.0`).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest          # ~20 seconds
.venv/bin/python visualize.py       # writes episode.html — open it in a browser
.venv/bin/python train_dcl.py       # train one architecture, then a comparison table
```

The visualization is a self-contained HTML player (play/pause, step, scrubber):
vehicles are triangles labeled `index:capacity` (orange = full, walking back to
the depot), waiting orders are red dots. The constants at the top of
`visualize.py` pick the policy — `"myopic"`, `"random"`, or the directory of a
trained agent — the seed and the number of periods.

## The model: dynamic vehicle routing on a grid

The "1-step decision" variant of the stylized dynamic vehicle routing problem
with stochastic customer requests from Akkerman, Mes & van Jaarsveld (2025),
*A comparison of reinforcement learning policies for dynamic vehicle routing
problems with stochastic customer requests*, Computers & Industrial Engineering
200, 110747 (`akkermans_model.pdf` in this repository).

A fleet of vehicles drives on a G×G grid; orders arrive at random cells and
cost `holding_cost` for every period they wait. Each period, every vehicle with
remaining capacity chooses its **next location**: its own cell or one of the
in-grid neighbours. Picking up an order takes one unit of capacity, and a
vehicle that runs out is skipped in the decision cycle while it automatically
walks back to the depot at (0,0), where its capacity resets. At most
`max_orders` orders wait in the system at once. The last two rules are
deliberate simplifications of the paper's model; the details and the full
period order are in the `mdp.py` docstring.

An action is a **grid cell** (the `CELL` encoding, G² actions, validity
masking admits at most 5). Every network in this repository therefore emits
one logit per cell, in row-major order, and never looks at the mask: DynaPlex
crops the logits to the valid cells when training (masked cross-entropy) and
when acting (masked argmax). The model also carries the paper's `DIRECTIONAL`
encoding (5 move actions), unused here.

Benchmarks: the paper's **myopic policy** (step toward the nearest waiting
order, ties random) in `mdp.py`, and DynaPlex's `RandomPolicy`.

## From state to network input: the featurizers

A DynaPlex featurizer is a small dataclass whose `write_features(state)` is
compiled into the simulation engine, so observations are produced inside the
rollouts without a Python round trip. It declares one **writer field per
tensor**; a `FixedObjectWriter` field becomes a `[count, width]` tensor, a
`VariableObjectWriter` field a padded tensor plus a `_count` per sample. The
network's `forward(batch)` receives a dict keyed by those field names.

- `featurizer_flat.py` — a single `GlobalStateWriter`: the vehicle-count grid,
  the order grid, the capacities (deciding vehicle first). 2G² + V numbers.
- `featurizer_graph.py` — `nodes` [G², 5]: per cell, the deciding vehicle is
  here, how many other vehicles are here, an order waits here, the depot is
  here, Manhattan distance to the deciding vehicle. The lattice edges are not
  features; the network builds them once from the node count.
- `featurizer_moves.py` — `candidates` [5, 4]: for stay/up/down/left/right the
  target cell's coordinates, whether an order waits there, and an in-grid
  flag; plus `orders` (variable, with `orders_count`), `vehicles` [V, 4] with a
  flag marking the deciding vehicle, and `depot` [1, 2].

Each module's docstring explains its layout; the tests pin the conventions
that the networks rely on (token order equals action order, one focal vehicle
per state, candidates are exactly the valid cells).

## The networks

All factories in `networks.py` have the signature
`factory(spec, num_actions, **kwargs) -> nn.Module` and are handed to DynaPlex
as `Net("networks.<factory>", **kwargs)`: an importable path plus keyword
arguments, so a saved agent rebuilds itself on any machine that has this
repository on its path.

- **`gin_net`** — input projection, then six residual rounds of
  `x_i ← x_i + MLP(x_i + Σ_{j adjacent} M(x_j))`, with `M` a per-layer MLP on
  every neighbour and the sum one matmul with the fixed lattice adjacency;
  then a shared MLP scores each node from three vectors: its own embedding,
  the mean of all node embeddings, and the embedding of the deciding
  vehicle's node (origin). The paper's layer is `M = identity`, a `(1 + ε)`
  self weight, no residual, and a per-node readout only; the Results section
  shows what each change is worth.
- **`move_token_net`** — each token type embedded by its own small MLP, one
  `TransformerEncoder` over `[candidates | depot | vehicles | orders]` with the
  padded orders masked, a shared head scoring each candidate token, and the
  five scores scattered into the G² action vector at the candidates' own
  coordinates. `interact=False` forbids candidate tokens from attending to one
  another.

## Training

```bash
.venv/bin/python train_dcl.py              # Deep Controlled Learning, one architecture
.venv/bin/python train_ppo.py              # PPO from scratch, flat features
.venv/bin/python compare_architectures.py  # every architecture at one budget, one table
```

There are no command-line flags: the architecture and the budget are a few
constants at the top of each script, meant to be edited.

`train_dcl.py` is the route that produces strong policies on this model:
collect states under the myopic policy, label each with the best move found by
rollouts, train the network to imitate the labels, and score the result
against the myopic and random benchmarks on common random numbers (lower cost
is better). `compare_architectures.py` does that for every architecture on
one collection: the states are collected and labelled once, without any
featurizer, and every arm is trained on exactly those states, so only the
featurizer/network pair differs. Artifacts land under `dynaplex_runs/`, and
runs are idempotent: rerunning loads the stored collection and agents instead
of recomputing, and adding an arm costs that arm's training only.

How a collection is spread over the machine is not a knob of this repository:
DynaPlex's `machine="auto"` recognizes a laptop or a cluster allocation and
applies the plan engineered for it (`python -m dynaplex.machine` prints what
it sees). `snellius/` has the two files needed to run on a SLURM cluster:

```bash
snellius/setup.sh                                          # once, on a login node
sbatch --time=08:00:00 snellius/train.job compare_architectures.py
```

### Results

The budget in `compare_architectures.py` — 10 000 labelled states, 500
rollouts of 100 periods per candidate move, one generation — is the smallest
at which the attention networks learn on this model. A first trial at 4 000
states and 200 rollouts left every architecture except the GIN at chance.

Average cost per period over the comparer's episodes, paired on common random
numbers (M4 Pro laptop, DynaPlex 1.14.0, 2026-11-07; one generation, 200 epochs
with patience 60):

| policy            | mean cost | delta vs myopic | notes                               |
|-------------------|----------:|----------------:|-------------------------------------|
| myopic (benchmark)|     3.096 |               0 |                                     |
| random            |     53.19 |          +50.09 |                                     |
| flat_mlp          |     87.75 |          +84.65 | overfits, stops at epoch 85         |
| gin (paper layer) |     4.019 |           +0.92 | 3 layers, node readout, lr 1e-4     |
| gin (as shipped)  |     2.161 |           -0.94 | 6 layers, lr 1e-3                   |
| moves             |     2.254 |           -0.84 | validation 28%                      |
| moves_independent |     2.861 |           -0.24 | no move-to-move attention           |

Paired standard errors on the deltas are 0.02 to 0.08 for the trained
policies; two seeds of the same GIN differ by 0.1, so the shipped GIN and the
move-token net are tied. Letting the five move tokens attend to each other is
worth 0.6 per period. All attention arms sit at about 27% validation accuracy
because 60% of the labelled states are myopic ties and the rollout labels on
those are mostly noise — validation loss is a poor guide here (the 3-layer
GIN with readouts matches the 6-layer one on validation loss and is 1.1 worse
in the comparer).

**Why the paper's GIN needs changes here.** In the paper the action is a
destination, so the node that gets scored is the order's own node and three
hops of context around it suffice. Here the action is a one-step move: the
five candidate nodes are neighbours, their few-hop views nearly coincide, and
an order six cells away is invisible to all of them, so they tie exactly where
the labels are already noisy. The switches were added one at a time, every
variant retrained on the SAME stored sample set (`dynaplex.train_network` on
`samples_gen0`, two to five minutes each on the laptop) and scored on the same
comparer episodes:

| GIN variant (lr 1e-3 unless noted)                    | mean cost |
|-------------------------------------------------------|----------:|
| paper layer, 3 layers, lr 1e-4                        |     4.019 |
| paper layer, 3 layers                                 |     3.200 |
| 6 layers, residual                                    |     3.557 |
| 3 layers, graph + origin readout                      |     2.847 |
| 3 layers, residual, graph + origin                    |     3.296 |
| 6 layers, residual, graph + origin                    |     2.231 (seed 1: 2.359) |
| 6 layers, residual, graph + origin, lr 1e-4           |     2.444 |
| 9 layers, residual, graph + origin                    |     2.280 |
| 6 layers, residual, graph + origin, learned ε         |     2.287 |
| 6 layers, residual, graph + origin, message MLP       |     2.161 |
| 6 layers, residual, graph + origin, learned ε + message |   2.292 |
| **shipped** = 6 layers, residual, graph + origin, message MLP | **2.161** |
| shipped without residual                              |     2.702 (error 0.19) |
| shipped without the graph-mean readout                |     2.342 |
| shipped without the origin readout                    |     2.297 |
| shipped without both readouts                         |     2.404 (error 0.14) |
| shipped with 3 layers                                 |     3.336 |

The readouts carry the result: scoring a candidate as an origin–destination
pair with a global summary attached is the paper's Edge Actions idea for a
single origin, and each readout is worth 0.15–0.2 on its own. Depth helps
once the readouts are in, and the residual connections are what make six
layers trainable (without them the policy is unstable across comparer
episodes). The learning rate is worth 0.2 (DynaPlex's default 1e-4 is ten
times below the paper's). A learned transform of the neighbours before the
sum is the best single point estimate; a learned self weight adds nothing on
top of it, so the shipped network has no ε at all. Everything that did not
earn its place was removed: `gin_net` is one architecture with no switches.

For reference, the predecessor of this repository measured, with the paper's
budget (2 000 random-policy rollouts per move, DynaPlex 1.12, Snellius): cell
tokens 2.15 and flat MLP 4.07 against myopic 3.10 average cost per period.

## Adapting the example

There is deliberately no `pyproject.toml` — this is a fork-and-adapt example,
not a package. Everything lives flat at the repository root and is run from
there; `requirements.txt` pins a known-good DynaPlex release. To try another
architecture, write a featurizer (its `write_features` must stay inside the
engine's Python subset: for-range loops, if/else, index arithmetic, `float()`
casts, single comparisons) and a factory in `networks.py`, add a test that
pins whatever convention the two share, and add the pair to the `ARMS` table.
