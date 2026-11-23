# example-nn-architectures

A [DynaPlex](https://dynaplex.github.io/DynaPlex/) example about **neural
network architectures for sequential decision problems**: one stochastic
vehicle-routing model on a grid, one training recipe, and three ways to present
the state to a network — a flat feature vector, a graph, a set of tokens for
the candidate moves. 

An MDP State contains all information to represent future transitions, and this is reflected in the model state. When doing DRL, we need something that maps the state to a potential input for a neural network. This repo shows that a single model can be input to various neural networks, depending on how the state is represented. In particular, the mapping from a state to neural network requires a featurizer, and this repo has three featurizers. It also shows how to train neural networks of three flavors:

| architecture | the network sees | featurizer | network |
|---|---|---|---|
| flat MLP | two G×G count maps and the capacities, as one vector | `featurizer_flat.py` | DynaPlex's default MLP |
| GIN | one node per cell on the grid graph, message passing along the lattice | `featurizer_graph.py` | `networks.gin_factory` |
| move tokens | one token per admissible move, attending to order, vehicle and depot tokens | `featurizer_moves.py` | `networks.move_token_factory` |

<paragraph explaining the two action representations>

The GIN starts from the "Node Actions" architecture of Begnardi, Baier, van
Jaarsveld & Zhang (2023), *Deep Reinforcement Learning for Two-sided Online
Bipartite Matching in Collaborative Order Picking*, ACML, and scores a
destination node in the same way. Here the destination is always within one
step of the vehicle, and the network needed more layers and two extra readouts
to tell those nearby cells apart (see Results). The
move-token net scores each available action as a token; its `interact`
switch turns candidate-to-candidate attention off, which makes the
relational part of that design measurable.

## Quickstart

Requires **Python 3.11–3.14** (`requirements.txt` pins `dynaplex==2.0.0`).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python visualize.py                # writes episode.html — open it in a browser
.venv/bin/python train_dcl_architectures.py  # train every architecture, then a comparison table
```

The visualization is a self-contained HTML player (play/pause, step, scrubber):
vehicles are triangles labeled `index:capacity` (orange = full, walking back to
the depot), orders are red dots with their weight written in them when it is
more than one. The constants at the top of
`visualize.py` pick the policy — `"myopic"`, `"random"`, or the directory of a
trained agent — the seed and the number of periods.

## The model: dynamic vehicle routing on a grid

The model is adapted from the "1-step decision" variant of the stylized
dynamic vehicle routing problem with stochastic customer requests in Akkerman,
Mes & van Jaarsveld (2025), *A comparison of reinforcement learning policies
for dynamic vehicle routing problems with stochastic customer requests*,
Computers & Industrial Engineering 200, 110747 (`akkermans_model.pdf` in this
repository).

A fleet of vehicles drives on a G×G grid. Each period, with probability
`order_probability`, one unit of demand arrives on a cell. The demand waiting
on one cell is an **order**, and its **weight** is the number of units; every
unit costs `holding_cost` for every period it waits. Each period, every
vehicle with remaining capacity chooses its **next location**: its own cell or
one of the in-grid neighbours. A vehicle that stands on an order takes as much
of its weight as it can carry. A vehicle that runs out of capacity walks back
to the depot at (0,0), where its capacity resets.

**Differences from the paper.** The model is not the paper's model, and the
costs below cannot be compared with the paper's:

- A cell can hold more than one unit of demand. In the paper a cell holds one
  order or none.
- At most `max_orders` cells (20) hold an order at the same time. Demand is
  never refused: at the cap, a unit is added to one of the existing orders. So
  a policy never has to look at more than 20 orders, which keeps the networks
  small. Under the myopic policy about three cells hold an order on average,
  and the cap is not reached.
- A vehicle without capacity takes no decisions: it walks to the depot by
  itself (first up, then left).

The full order of events within a period is in the `mdp.py` docstring.

An action is a **grid cell** (the `CELL` encoding, G² actions, validity
masking admits at most 5). Every network in this repository therefore emits
one logit per cell, in row-major order, and never looks at the mask: DynaPlex
crops the logits to the valid cells when training and when acting. The model
also carries the paper's `DIRECTIONAL` encoding (5 move actions), unused here.

Benchmarks: the paper's **myopic policy** (step toward the nearest order; on
equal distance toward the heavier one, what is still tied at random) in
`mdp.py`, and DynaPlex's `RandomPolicy`.

## From state to network input: the featurizers

A DynaPlex featurizer is a small dataclass whose `write_features(state)` is
compiled into the simulation engine, so observations are produced inside the
rollouts without a Python round trip. It declares one **writer field per
tensor**; a `FixedObjectWriter` field becomes a `[count, width]` tensor, a
`VariableObjectWriter` field a padded tensor plus a `_count` per sample. The
network's `forward(batch)` receives a dict keyed by those field names.

- `featurizer_flat.py` — a single `GlobalStateWriter`: the vehicle-count grid,
  the weight waiting on each cell, the capacities (deciding vehicle first).
  2G² + V numbers.
- `featurizer_graph.py` — `nodes` [G², 5]: per cell, the deciding vehicle is
  here, how many other vehicles are here, the weight of the order here, the
  depot is here, Manhattan distance to the deciding vehicle. The lattice edges are not
  features; the network builds them once from the node count.
- `featurizer_moves.py` — `candidates` [5, 4]: for stay/up/down/left/right the
  target cell's coordinates, the weight of the order there, and an in-grid
  flag; plus `orders` [`max_orders`, 3] (position and weight of each order,
  padded, with `orders_count`), `vehicles` [V, 4] with a flag marking the
  deciding vehicle, and `depot` [1, 2]. With the default model that is 31
  tokens.

Each module's docstring explains its layout and the conventions that its
network relies on (token order equals action order, one focal vehicle per
state, candidates are exactly the valid cells).

## The networks

All factories in `networks.py` have the signature
`factory(spec, num_actions, **kwargs) -> nn.Module` and are handed to DynaPlex
as `Net("networks.<factory>", **kwargs)`: an importable path plus keyword
arguments, so a saved agent rebuilds itself on any machine that has this
repository on its path.

- **`gin_factory`** — input projection, then six residual rounds of
  `x_i ← x_i + MLP(x_i + Σ_{j adjacent} M(x_j))`, with `M` a per-layer MLP on
  every neighbour and the sum one matmul with the fixed lattice adjacency;
  then a shared MLP scores each node from three vectors: its own embedding,
  the mean of all node embeddings, and the embedding of the deciding
  vehicle's node (origin). The paper's layer is `M = identity`, a `(1 + ε)`
  self weight, no residual, and a per-node readout only; the Results section
  shows what each change is worth.
- **`move_token_factory`** — each token type embedded by its own small MLP, one
  `TransformerEncoder` over `[candidates | depot | vehicles | orders]` with the
  padded orders masked, a shared head scoring each candidate token, and the
  five scores scattered into the G² action vector at the candidates' own
  coordinates. `interact=False` forbids candidate tokens from attending to one
  another.

## Training

```bash
.venv/bin/python train_dcl_architectures.py  # Deep Controlled Learning, every architecture, one table
.venv/bin/python train_ppo.py                # PPO from scratch
```

There are no command-line flags: each script is a short sequence of calls
with the budget written in them, meant to be edited.

`train_dcl_architectures.py` is the route that produces strong policies on
this model: collect states under the myopic policy, label each with the best
move found by rollouts, train a network to imitate the labels, and score the
result against the myopic and random benchmarks on common random numbers
(lower cost is better). It does that for every architecture on one
collection: the states are collected and labelled once, without any
featurizer, and every arm is trained on exactly those states, so only the
featurizer/network pair differs. Artifacts land under `dynaplex_runs/`, and
runs are idempotent: rerunning loads the stored collection and agents instead
of recomputing, and adding an arm costs that arm's training only. A comment in
the script shows how to train a second generation.

How a collection is spread over the machine is not a knob of this repository:
DynaPlex's `machine="auto"` recognizes a laptop or a cluster allocation and
applies the plan engineered for it (`python -m dynaplex.machine` prints what
it sees). `snellius/` has the two files needed to run on a SLURM cluster:

```bash
snellius/setup.sh                                          # once, on a login node
sbatch --time=08:00:00 snellius/train.job train_dcl_architectures.py
```

### Results

The budget in `train_dcl_architectures.py` is 10 000 labelled states, 500
rollouts of 100 periods per candidate move, one generation, and 200 epochs
with patience 60. The whole script takes about 11 minutes on an M4 Pro laptop:

| step | time |
|---|---:|
| collect and label the states (once) | 1 min 7 s |
| train the flat MLP | 4 s |
| train the GIN | 4 min 21 s |
| train the move-token network | 4 min 31 s |
| compare the five policies | 1 min 14 s |

Average cost per period over the comparer's 256 episodes, paired on common
random numbers (DynaPlex 2.0.0, 2026-11-23):

| policy            | mean cost | delta vs myopic | error of delta | notes                    |
|-------------------|----------:|----------------:|---------------:|--------------------------|
| myopic (benchmark)|     3.031 |               0 |                |                          |
| random            |     99.89 |          +96.86 |           0.84 |                          |
| flat_mlp          |    183.35 |         +180.32 |           1.05 | stops at epoch 90        |
| gin               |     2.250 |           -0.78 |           0.02 | stops at epoch 175       |
| moves             |     2.279 |           -0.75 |           0.02 | stops at epoch 195       |

The GIN and the move-token network both beat the myopic policy by a quarter of
its cost, and they are tied with each other. The flat MLP does not learn a
usable policy from this budget: it is worse than random.

The validation accuracy of both networks is about 24%, and that is expected:
in many labelled states several moves are equally good, and the rollouts
cannot separate them (31% of the labels are more than one standard error
ahead of the second-best move). The cost in the comparer is the measure to
look at, not the accuracy.

### Earlier measurements: how the GIN got its shape

The measurements in this subsection were made on the previous version of the
model (one order per cell, at most 100 orders, DynaPlex 1.14.0, 2026-11-07 and
2026-11-08). They have not been repeated on the present model. On that model
the myopic policy cost 3.096, the GIN as it is now 2.161 and the move-token
network 2.254; with `interact=False` (move tokens do not attend to each other)
the move-token network ended at 2.861.

**Why the paper's GIN needed changes here.** In the paper a destination can be
far away and is itself a node that matters, so three rounds of message passing
around it are enough to tell its value. Here the five admissible destinations
are the vehicle's own cell and its neighbours. They usually hold no order
themselves, their few-hop views nearly coincide, and an order six cells away
is invisible to all of them, so the paper's network scores them almost the
same. The changes were added one at a time, every variant retrained on the
same stored collection and scored on the same comparer episodes:

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
| **as it is now** = 6 layers, residual, graph + origin, message MLP | **2.161** |
| as it is now, without residual                              |     2.702 (error 0.19) |
| as it is now, without the graph-mean readout                |     2.342 |
| as it is now, without the origin readout                    |     2.297 |
| as it is now, without both readouts                         |     2.404 (error 0.14) |
| as it is now, with 3 layers                                 |     3.336 |

The readouts carry the result: scoring a candidate as an origin–destination
pair with a global summary attached is the paper's Edge Actions idea for a
single origin, and each readout is worth 0.15–0.2 on its own. Depth helps
once the readouts are in, and the residual connections are what make six
layers trainable (without them the policy is unstable across comparer
episodes). The learning rate is worth 0.2 (DynaPlex's default 1e-4 is ten
times below the paper's). A learned transform of the neighbours before the
sum is the best single point estimate; a learned self weight adds nothing on
top of it, so the network has no ε at all. Everything that did not
earn its place was removed: `gin_factory` is one architecture with no switches.

## Adapting the example

There is deliberately no `pyproject.toml` — this is a fork-and-adapt example,
not a package. Everything lives flat at the repository root and is run from
there; `requirements.txt` pins a known-good DynaPlex release. To try another
architecture, write a featurizer (its `write_features` must stay inside the
engine's Python subset: for-range loops, if/else, index arithmetic, `float()`
casts, single comparisons) and a factory in `networks.py`, and add one more
`train_policy` call and one more row for the comparer to
`train_dcl_architectures.py`.
