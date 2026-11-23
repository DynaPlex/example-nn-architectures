# example-nn-architectures

A [DynaPlex](https://dynaplex.github.io/DynaPlex/) example about **neural
network architectures for sequential decision problems**: one stochastic
vehicle-routing model on a grid, one training recipe, and three ways to present
the state to a network — a flat feature vector, a graph, a set of tokens for
the candidate moves.

## The model

A fleet of vehicles drives on a 10×10 grid. Demand arrives on random cells,
and every unit of demand costs money for every period it waits. Each period,
every vehicle decides where to go next: it stays where it is or moves to a
neighbouring cell. A vehicle picks up the demand on the cell it stands on, as
much as it can carry, and when it is full it drives back to the depot in the
corner to unload. The goal is a policy that keeps the waiting demand low.

The best way to get a feeling for the model is to watch it. Run the commands
below from the root of the repository, with Python 3.11 to 3.14.

First install. The first line creates an empty virtual environment in the
folder `.venv`, and the second installs DynaPlex and the other packages of
[`requirements.txt`](requirements.txt) into it:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then simulate one episode, which writes the file `episode.html`:

```bash
.venv/bin/python visualize.py
```

Then open that file in the browser, for example on macOS:

```bash
open episode.html
```

On Windows the command is `start episode.html` and on Linux
`xdg-open episode.html`. You can also just double-click the file.

The page is a player with play/pause, step, and a slider to jump to any
period. Vehicles are triangles labeled `index:capacity` (orange = full,
driving back to the depot), and orders are red dots with their weight written
in them when it is more than one. What you see is the myopic policy, in which
every vehicle steps toward the nearest order. The constants at the top of
[`visualize.py`](visualize.py) pick the policy (`"myopic"`, `"random"`, or the directory of a
trained agent), the seed and the number of periods.

## The action representation

When developing DRL, it is important to think about how actions are
represented in your model. For example, in this model a vehicle decides where
to go next, and there are two logical ways to represent that decision
(`ActionEncoding` in [`mdp.py`](mdp.py)). The dynamics are identical under both.

In the **cell** representation an action is a grid cell: the vehicle's next
location. On the 10×10 grid there are 100 actions, of which at most five are
valid for a given vehicle: its own cell and the neighbours.

In the **directional** representation an action is a move: stay, up, down,
left or right. There are five actions, and a move that would leave the grid is
not valid.

## The neural network architectures

An MDP State contains all information to represent future transitions, and this is reflected in the model state. When doing DRL, we need something that maps the state to a potential input for a neural network. This repo shows that a single model can be input to various neural networks, depending on how the state is represented. In particular, the mapping from a state to neural network requires a featurizer, and this repo has three featurizers. It also shows how to train neural networks of three flavors:

| architecture | the network sees | the network scores | featurizer | network |
|---|---|---|---|---|
| flat MLP | two G×G count maps and the capacities, as one vector | 5 directions | [`featurizer_flat.py`](featurizer_flat.py) | DynaPlex's default MLP |
| GIN | one node per cell on the grid graph, message passing along the lattice | 100 grid cells | [`featurizer_graph.py`](featurizer_graph.py) | `networks.gin_factory` |
| move tokens | one token per move, attending to order, vehicle and depot tokens | 5 directions | [`featurizer_moves.py`](featurizer_moves.py) | `networks.move_token_factory` |

The GIN uses the cell representation. It has one node per cell, so its task is
to score each of the 100 cells for how attractive it is for the vehicle to go
there. This follows the "Node Actions" architecture of Begnardi, Baier, van
Jaarsveld & Zhang (2023), *Deep Reinforcement Learning for Two-sided Online
Bipartite Matching in Collaborative Order Picking*, ACML, where the network
also scores nodes.

The flat MLP and the move-token network use the directional representation,
which is the more straightforward one: the network outputs five scores, one
per move.

A network never looks at which actions are valid. It scores all of them for a state, and
DynaPlex only looks at the scores for the actions that are valid for that state (as indicated by `write_action_validity` in [`mdp.py`](mdp.py)), and selects the one with highest score. 

## The model in detail

The model is adapted from the "1-step decision" variant of the stylized
dynamic vehicle routing problem with stochastic customer requests in Akkerman,
Mes & van Jaarsveld (2025), *A comparison of reinforcement learning policies
for dynamic vehicle routing problems with stochastic customer requests*,
Computers & Industrial Engineering 200, 110747 ([`akkermans_model.pdf`](akkermans_model.pdf) in this
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

The full order of events within a period is in the [`mdp.py`](mdp.py) docstring.

Benchmarks: the paper's **myopic policy** (step toward the nearest order; on
equal distance toward the heavier one, what is still tied at random) in
[`mdp.py`](mdp.py), and DynaPlex's `RandomPolicy`.

## From state to network input: the featurizers

A DynaPlex featurizer is a small dataclass whose `write_features(state)` is
compiled into the simulation engine, so observations are produced inside the
rollouts without a Python round trip. It declares one **writer field per
tensor**; a `FixedObjectWriter` field becomes a `[count, width]` tensor, a
`VariableObjectWriter` field a padded tensor plus a `_count` per sample. The
network's `forward(batch)` receives a dict keyed by those field names.

- [`featurizer_flat.py`](featurizer_flat.py) — a single `GlobalStateWriter`: the vehicle-count grid,
  the weight waiting on each cell, the capacities (deciding vehicle first).
  2G² + V numbers.
- [`featurizer_graph.py`](featurizer_graph.py) — `nodes` [G², 5]: per cell, the deciding vehicle is
  here, how many other vehicles are here, the weight of the order here, the
  depot is here, Manhattan distance to the deciding vehicle. The lattice edges are not
  features; the network builds them once from the node count.
- [`featurizer_moves.py`](featurizer_moves.py) — `candidates` [5, 4]: for stay/up/down/left/right the
  target cell's coordinates, the weight of the order there, and an in-grid
  flag; plus `orders` [`max_orders`, 3] (position and weight of each order,
  padded, with `orders_count`), `vehicles` [V, 4] with a flag marking the
  deciding vehicle, and `depot` [1, 2]. With the default model that is 31
  tokens.

Each module's docstring explains its layout and the conventions that its
network relies on (node or token order equals action order, one focal vehicle
per state).

## The networks

All factories in [`networks.py`](networks.py) have the signature
`factory(spec, num_actions, **kwargs) -> nn.Module` and are handed to DynaPlex
as `Net("networks.<factory>", **kwargs)`: an importable path plus keyword
arguments, so a saved agent rebuilds itself on any machine that has this
repository on its path.

- **The MLP** — not in `networks.py`: it is the standard network that is built
  into DynaPlex (`dynaplex.nn.mlp_factory`) and is used when no network is
  given. Two hidden layers of 128 units, from the flat feature vector (205
  numbers) to five scores, one per move.
- **`gin_factory`** — one logit per node, and node i is cell i, which is action
  i in the cell representation. Input projection, then six residual rounds of
  `x_i ← x_i + MLP(x_i + Σ_{j adjacent} M(x_j))`, with `M` a per-layer MLP on
  every neighbour and the sum one matmul with the fixed lattice adjacency;
  then a shared MLP scores each node from three vectors: its own embedding,
  the mean of all node embeddings, and the embedding of the deciding
  vehicle's node (origin). The paper's layer is `M = identity`, a `(1 + ε)`
  self weight, no residual, and a per-node readout only.
- **`move_token_factory`** — each token type embedded by its own small MLP, one
  `TransformerEncoder` over `[candidates | depot | vehicles | orders]` with the
  padded orders masked, and a shared head scoring each of the five move
  tokens. Those five scores are the output: token k is move k, which is action
  k in the directional representation. `interact=False` forbids move tokens
  from attending to one another.

PPO also needs a critic, a second network that estimates the cost still to
come from a state. It only helps during training and is not part of the
trained agent. For the flat feature vector DynaPlex builds one itself; the
critics for the GIN and the move tokens are in
[`critic_networks.py`](critic_networks.py). Deep Controlled Learning has no
critic.

## Training

There are two training scripts. The first trains every architecture with Deep
Controlled Learning and prints a comparison table per action representation:

```bash
.venv/bin/python train_dcl_architectures.py
```

The second trains the GIN with PPO, from scratch. The same calls for the other
two architectures are in the script, commented out:

```bash
.venv/bin/python train_ppo.py
```

There are no command-line flags: each script is a short sequence of calls
with the budget written in them, meant to be edited.

[`train_dcl_architectures.py`](train_dcl_architectures.py) is the route that produces strong policies on
this model: collect states under the myopic policy, label each with the best
move found by rollouts, train a network to imitate the labels, and score the
result against the myopic and random benchmarks on common random numbers
(lower cost is better). A collection belongs to one model, and the action
representation is part of the model, so the script collects twice: once with
cells for the GIN, once with directions for the flat MLP and the move-token
network. Collecting does not involve a featurizer, so those two are trained on
exactly the same labelled states. A comparer also works on one model, so there
are two tables; both use the same seed, so they are about the same episodes
and the myopic row is identical. Artifacts land under `dynaplex_runs/`, and
runs are idempotent: rerunning loads the stored collection and agents instead
of recomputing, and adding an arm costs that arm's training only. A comment in
the script shows how to train a second generation.

How a collection is spread over the machine is not a knob of this repository:
DynaPlex's `machine="auto"` recognizes a laptop or a cluster allocation and
applies the plan engineered for it (`python -m dynaplex.machine` prints what
it sees). [`snellius/`](snellius/) has the two files needed to run on a SLURM cluster:

```bash
snellius/setup.sh                                          # once, on a login node
sbatch --time=08:00:00 snellius/train.job train_dcl_architectures.py
```

### Results

The budget in [`train_dcl_architectures.py`](train_dcl_architectures.py) is, for each of the two
collections, 10 000 labelled states and 500 rollouts of 100 periods per
candidate move; one generation, and 200 epochs with patience 60. The whole
script takes about 12 minutes on an M4 Pro laptop: about one minute for each
of the two collections, four to five minutes each to train the GIN and the
move-token network, and a minute for the comparisons. The flat MLP trains in
seconds.

Average cost per period over the comparer's 256 episodes, paired on common
random numbers (DynaPlex 2.0.0, 2026-11-23). The script prints one table per
action representation; they are about the same episodes, and the myopic row
is identical in both.

Cell representation:

| policy            | mean cost | delta vs myopic | error of delta |
|-------------------|----------:|----------------:|---------------:|
| myopic (benchmark)|     3.031 |               0 |                |
| random            |     99.89 |          +96.86 |           0.84 |
| gin               |     2.250 |           -0.78 |           0.02 |

Directional representation:

| policy            | mean cost | delta vs myopic | error of delta |
|-------------------|----------:|----------------:|---------------:|
| myopic (benchmark)|     3.031 |               0 |                |
| random            |     99.88 |          +96.85 |           0.82 |
| flat_mlp          |    167.50 |         +164.47 |           1.88 |
| moves             |     2.037 |           -0.99 |           0.02 |

The GIN and the move-token network both beat the myopic policy clearly, so
they learn something useful that the myopic rule does not have. A likely part
of it is coordination: under the myopic policy every vehicle drives to its
own nearest order, so several vehicles can chase the same one, while a
network sees where the other vehicles are. We have not measured what the
networks do differently; the way to find out is to watch a trained agent with
[`visualize.py`](visualize.py).

The difference between the two is within the variation between training runs: the
same move-token network has ended between 2.0 and 2.3 in runs that differed
only in floating-point detail, because a different epoch was kept. The flat
MLP does not learn a usable policy from this budget: it is worse than random.

The validation accuracy of both networks is about 24%, and that is expected:
in many labelled states several moves are equally good, and the rollouts
cannot separate them (31% of the labels are more than one standard error
ahead of the second-best move). The cost in the comparer is the measure to
look at, not the accuracy.

PPO is not in these tables. [`train_ppo.py`](train_ppo.py) ran its 500 000
steps with the GIN (about 27 minutes on the same laptop) and did not learn a
useful policy: its cost was 205 per period, worse than random. PPO starts from
nothing and first has to discover that driving toward orders pays off, and
this budget is not enough for that. The script is there to show how PPO is
called, not to produce a policy.

## Adapting the example

There is deliberately no `pyproject.toml` — this is a fork-and-adapt example,
not a package. Everything lives flat at the repository root and is run from
there; [`requirements.txt`](requirements.txt) pins a known-good DynaPlex release. To try another
architecture, write a featurizer (its `write_features` must stay inside the
engine's Python subset: for-range loops, if/else, index arithmetic, `float()`
casts, single comparisons) and a factory in [`networks.py`](networks.py), and add one more
`train_policy` call and one more row for the comparer to
[`train_dcl_architectures.py`](train_dcl_architectures.py).
