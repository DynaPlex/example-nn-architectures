# Generating the engine featurizer from the SAT schema

*Proposal, 2026-09-10. Not built. Companion to `MODULE_MAP.md` §9 ("Phase 2b
codegen is deliberately not done").*

## The idea in one paragraph

`featurizers_sat.py` is a hand transcription of `sat_schema.py`: the dataclasses
say which tokens exist and which columns they carry, `build_sat_observation`
says how each column is computed from the state, and `write_features` repeats
both in writer calls. The parity test exists because two copies drift. The
proposal is to keep ONE copy — the schema dataclasses and the builder — and
generate `write_features` (and the writer fields and `spec()`) from them by
reflection **at import time, in CPython**, the way `@featurizer` already
synthesizes `bind`/`reset`/`finish`/`spec`. The generated method is ordinary
written source (rendered from an AST, registered in `linecache`, exec'd into a
`dynaplex._synth.*` module), so it can be printed and read, and the DynaML
compiler sees plain code. Nothing changes in the language or the engine.

What this automates is the **layout**: which writer, fixed or variable, how
wide, in what column order, which tensor holds the actions. The feature
**values** still come from the builder someone wrote. "Auto-featurization" here
means "declare the tokens, write the values once, get the engine featurizer for
free" — not "features from the MDP with no code".

## Inputs

1. **Token dataclasses** (`sat_schema.py` today). One per token type; field
   order is column order; every field is a float (or an `int`/`bool` the
   generator casts). Markers, all existing in `sat_markers.py`:
   * `class MoveAction(Action)` — instances are actions; the tensor built from
     this list is the action tensor the network scores.
   * `action_index: ActionKey[int]` — the MDP action id (also a column).
2. **The observation root** (`SATObservation`): a dataclass whose fields are a
   token (`depot: DepotTok`) or a list of tokens (`orders: list[OrderTok]`).
   A single token becomes a `FixedObjectWriter` of count 1. A list needs a
   capacity the type cannot express — the one new annotation:
   * `vehicles: Annotated[list[VehicleTok], dp.len("mdp.n_vehicles")]` — fixed
     count, `FixedObjectWriter`, no count tensor;
   * `orders: Annotated[list[OrderTok], dp.max_len("mdp.max_orders")]` —
     variable, `VariableObjectWriter` plus the `orders_count` companion.
   The string is an expression over `mdp`, exactly as `TokenSpec.max_count` is
   today. (Whether `dp.len`/`dp.max_len` live in `dynaplex.modelling` or stay in
   this repo is open; nothing depends on it yet.)
3. **The builder**: `build_sat_observation(state, mdp) -> SATObservation` and
   its helpers. It must be in the DynaML subset except for two things the
   generator rewrites: constructing a token dataclass, and appending one to a
   list.

## Output

A `@featurizer` class equivalent to today's `SATFeaturizer`:

* writer fields in root-field order, kinds from the annotations;
* `spec()` from the capacities and widths;
* `write_features(self, state)` = the builder body with each
  `xs.append(Tok(a=e1, b=e2))` rewritten to `self.xs.begin_object()`,
  `self.xs.append(e1)`, `self.xs.append(e2)` (field order, not keyword order),
  a single-token field rewritten to the same without `begin_object`, and the
  final root construction dropped. Helper functions that *return a token*
  (`_move_action`) are inlined at the call site; helpers that return a scalar
  (`nearest_order_distance`) become methods and are called as today.

The action tensor is whichever root field holds `Action` instances; the
network reads it by role, as `sat_agile_net` reads `moves` now.

## Contracts the generator cannot check

* **Action tokens complete and in action-index order** — logit position ==
  action id. The builder guarantees it; `test_move_token_i_is_action_i` proves
  it. Keep that test.
* A variable list never exceeds its `max_len` — the writer asserts at runtime.
* The builder is called at `AWAIT_ACTION` only (it reads `current_vehicle`).

## What the builder may use — measured on dynaplex 1.14.0

Probed by compiling small featurizers through `collect_samples`:

| idiom in the current builder | compiles? | what to do |
|---|---|---|
| `and`/`or`, `not` | yes | — |
| conditional expression `1.0 if c else 0.0` | yes | — |
| helper method returning `int` (`_nearest`) | yes | — |
| chained comparison `0 <= row < g` | **no** (`Only single comparison supported`) | `0 <= row and row < g` — the generator can rewrite this mechanically |
| `for k, (dr, dc) in enumerate(_MOVES)` | **no** (tuple-unpacking target) | `for k in range(5)` with the deltas from a class field or an `if` ladder |
| module-level constant `_MOVES` | **no** (`used before assignment`) | make it a `Final` field of the featurizer, or inline |

So the builder needs three small edits before generation is possible; none
changes what it computes. A generator that rewrites chained comparisons is
cheap; the other two are better fixed in the builder.

## Acceptance

`featurizers_sat.py` IS the expected output. The slice is done when the
generated class, pointed at `SATObservation` + `build_sat_observation`, passes
`test_featurizers_sat.py` unchanged — in particular the parity oracle
(`test_dense_slabs_match_the_tokenizer_exactly`) and the engine collection test.
At that point the hand-written featurizer is deleted and the schema table's
`columns`/`max_count` become redundant with the dataclasses and annotations.

## Open questions

* Nested actions (an `Action` with list-of-token children, `OrderedList`): the
  tokenizer supports them, the dense net does not need them for the VRP. Out of
  scope for the first slice; the generator should refuse them loudly.
* `int`/`bool`/`Enum` token fields: cast to float as the tokenizer's
  `_to_float` does, or require `float` in the dataclass. Requiring `float`
  is simpler and matches the writers.
* Where the markers and the generator live once a second schema exists:
  `dynaplex.modelling` (they are 40 lines), with this repo as the worked
  example.
