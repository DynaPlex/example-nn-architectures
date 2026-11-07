"""The automatic tokenizer: SATObservation dataclass trees -> flat token sequences.

Ported from dynaplex2_playground `data/sequence_model.py` (Phase 1 of the SAT
integration). This walks a dataclass tree by TYPE INTROSPECTION and produces a
flat sequence of feature vectors, each tagged with a type ID. "Automatic" means
the token layout is derived from the observation's dataclass fields and type
annotations — you declare a schema, not a feature-writing procedure.

Where this port diverges from upstream, and why
-----------------------------------------------
This file is CPython-only and always will be: it needs dynamic `getattr`,
`get_type_hints`, dicts of Python objects and `isinstance`, none of which the
DynaPlex interpreter can transpile (it compiles a subset of for/while/if/assert/
assignment into the engine). So it cannot run inside `write_features`.

That is not a defeat, because the reflection here is really a COMPILE-TIME
activity: once the observation schema is known, the token layout is fixed. This
port therefore hardens the tokenizer into a setup-time schema deriver and an
executable reference (see `sat_schema`, and the parity tests) whose output the
static engine-side featurizer must reproduce. Four upstream behaviours are
removed as part of that hardening, each because it is unrepresentable under a
frozen observation spec:

  * the Cython `_fast_walk` path — a second implementation to keep in parity,
    for a setup-time cost that no longer matters;
  * dynamic registry growth + `extend_for_new_type` callbacks — see
    `_assert_registry`: the type universe is now CLOSED and a late-discovered
    type raises;
  * the ragged-feature fallback in `_build_tensors_py`, which silently
    zero-padded a type whose width varied per instance — exactly the bug a
    static spec cannot survive, so it now raises;
  * `__getstate__`/`__setstate__` — this object is never pickled here; DynaPlex
    agents are weights + a JSON sidecar, no pickled code ever.

Action grouping
---------------
When an Action dataclass contains nested child objects (other dataclasses or
lists of dataclasses), those children become separate tokens.  All tokens that
belong to the same action sub-tree share an ``action_group_id``.  The root
Action token is additionally marked via ``is_group_root``.

For flat actions (no nesting) each action is its own group of size 1, which
preserves full backward compatibility with existing code.

Intra-group positional ordering
-------------------------------
Each token within an action group is assigned an ``intra_group_position``:
root = 0, then children are numbered sequentially (1, 2, 3, …) in the order
they appear in the dataclass fields (single nested objects first, then list
items preserving their list order).  This lets the policy net add positional
embeddings so that, e.g., ``[Stop_A, Stop_B]`` is distinguishable from
``[Stop_B, Stop_A]``.

Performance
-----------------------
All type introspection (``get_type_hints``, ``fields``, ``isinstance``,
``_is_primitive``, ``_is_list_of_dataclass``, ``_is_ordered_list``) is
performed **once** per dataclass type and cached in a ``_TypePlan``.
Subsequent ``build()`` calls follow the pre-computed plan with zero
reflection overhead.  Tensor construction uses ``torch.tensor`` on flat
lists instead of element-wise Python loops.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Callable, Dict, List, Type, get_args, get_origin, get_type_hints
from enum import Enum

import torch

from sat_markers import Action, ActionKey, OrderedList


# ============================================================================
# CACHED WALK PLAN
# ============================================================================

# Field categories — avoid string comparisons in the hot path
_CAT_PRIMITIVE = 0
_CAT_LIST_DC = 1
_CAT_SINGLE_DC = 2
_CAT_SKIP = 3


class _FieldPlan:
    """Cached per-field introspection result."""
    __slots__ = ('name', 'category', 'is_ordered', 'to_float_fn', 'is_action_key')

    def __init__(
        self,
        name: str,
        category: int,
        is_ordered: bool = False,
        to_float_fn: Callable[[Any], float] | None = None,
        is_action_key: bool = False,
    ):
        self.name = name
        self.category = category
        self.is_ordered = is_ordered
        self.to_float_fn = to_float_fn
        self.is_action_key = is_action_key


class _TypePlan:
    """Cached per-type introspection result.

    Replaces ``get_type_hints``, ``fields``, ``_is_primitive``,
    ``_is_list_of_dataclass``, ``_is_ordered_list``, ``_unwrap_optional``
    — all resolved once and stored here.
    """
    __slots__ = ('type_name', 'is_action_base', 'field_plans', 'has_primitives',
                 'is_dataclass')

    def __init__(
        self,
        type_name: str,
        is_action_base: bool,
        field_plans: List[_FieldPlan],
        has_primitives: bool,
        is_dc: bool = True,
    ):
        self.type_name = type_name
        self.is_action_base = is_action_base
        self.field_plans = field_plans
        self.has_primitives = has_primitives
        self.is_dataclass = is_dc


def _make_to_float_fn(ftype: Type) -> Callable[[Any], float]:
    """Create a fast value→float converter for a field, resolved at plan time.

    For ``Enum`` fields the member→index mapping is pre-built so that
    ``build()`` does an O(1) dict lookup instead of O(n) ``list().index()``.
    """
    try:
        if issubclass(ftype, Enum):
            mapping = {member: float(i) for i, member in enumerate(ftype)}
            return mapping.__getitem__
    except TypeError:
        pass
    return float  # works for int, float, bool


# ============================================================================
# TOKEN TYPES
# ============================================================================

class TokenInfo:
    """Metadata about a single token in the sequence."""
    __slots__ = ("type_name", "features", "is_action", "source_obj",
                 "action_group_id", "is_group_root", "intra_group_position")

    def __init__(
        self,
        type_name: str,
        features: List[float],
        is_action: bool,
        source_obj: Any = None,
        action_group_id: int = -1,
        is_group_root: bool = False,
        intra_group_position: int = 0,
    ):
        self.type_name = type_name
        self.features = features
        self.is_action = is_action
        self.source_obj = source_obj
        self.action_group_id = action_group_id
        self.is_group_root = is_group_root
        self.intra_group_position = intra_group_position


# ============================================================================
# SEQUENCE BUILDER
# ============================================================================

class SequenceBuilder:
    """Walks an SATObservation dataclass tree and produces a flat token sequence.

    Each dataclass instance (GlobalState, Item, MoveAction, etc.) becomes one
    token.  The root SATObservation container is NOT included as a token (it has no
    features of its own).

    Output dict keys
    ~~~~~~~~~~~~~~~~
    - ``type_features``     : dict[str, Tensor]  — per-type raw features
    - ``type_ids``          : Tensor[seq_len]     — integer type id per position
    - ``action_mask``       : BoolTensor[seq_len] — True for every token that
                              belongs to *any* action group (root or child)
    - ``action_group_ids``  : LongTensor[seq_len] — group id (≥ 0) for action
                              sub-tree tokens, -1 for context tokens
    - ``is_group_root``     : BoolTensor[seq_len] — True only for the root
                              Action token of each group
    - ``intra_group_pos``   : LongTensor[seq_len] — position of this token
                              within its action group (0 for root, 1+ for
                              children in order).  0 for context tokens.
    - ``num_action_groups`` : int — number of distinct action groups
    - ``seq_len``           : int
    - ``type_names``        : list of unique type names (index = type_id)
    - ``token_type_names``  : list[str] per-position type name
    - ``indices_by_type``   : dict[str, list[int]] — global indices per type
    """

    def __init__(self, action_base: Type = Action, skip_root: bool = True):
        self.action_base = action_base
        self.skip_root = skip_root
        # Discovered during first call; extended dynamically when new types appear
        self._type_name_to_id: Dict[str, int] | None = None
        self._type_names: List[str] | None = None
        self._raw_dims: Dict[str, int] | None = None
        # cached type introspection plans
        self._type_plans: Dict[type, _TypePlan] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, root: Any) -> Dict[str, Any]:
        """Convert a root dataclass (e.g. SATObservation) into a flat token dict."""
        tokens: list = []
        counter = [0]  # mutable counter for action group IDs

        self._walk(root, tokens, is_root=True, action_group_id=-1, counter=counter)

        # Discover the type registry on the first call; every later call must
        # agree with it exactly (see _assert_registry: the type universe is
        # CLOSED). A growing registry is what the dense/engine port cannot
        # represent — the observation spec is frozen before the engine build.
        if self._type_name_to_id is None:
            self._discover_types(tokens)
        else:
            self._assert_registry(tokens)

        result = self._build_tensors_py(tokens, counter[0])
        # collect from group-root token source_obj (O(seq_len) pass)
        result["_actions"] = [tok.source_obj for tok in tokens if tok.is_group_root]

        return result

    def _build_tensors_py(self, tokens: List[TokenInfo], num_action_groups: int) -> Dict[str, Any]:
        """Pure-Python tensor assembly (fallback when Cython is not available)."""
        # Build tensors
        seq_len = len(tokens)
        type_ids = torch.zeros(seq_len, dtype=torch.long)
        action_group_ids = torch.full((seq_len,), -1, dtype=torch.long)
        is_group_root = torch.zeros(seq_len, dtype=torch.bool)
        intra_group_pos = torch.zeros(seq_len, dtype=torch.long)
        token_type_names: List[str] = []

        # Group features by type so we can build per-type tensors
        features_by_type: Dict[str, List[List[float]]] = {}
        indices_by_type: Dict[str, List[int]] = {}
        # Count per-type tokens seen so far (for local index tracking)
        _type_count: Dict[str, int] = {}
        # Precompute action_root_map: action_idx → (type_name, local_idx)
        action_root_map: List[tuple] = []

        for i, tok in enumerate(tokens):
            # Registry is already up-to-date (extended in build() before we get here)
            tid = self._type_name_to_id[tok.type_name]
            type_ids[i] = tid
            action_group_ids[i] = tok.action_group_id
            is_group_root[i] = tok.is_group_root
            intra_group_pos[i] = tok.intra_group_position
            token_type_names.append(tok.type_name)
            features_by_type.setdefault(tok.type_name, []).append(tok.features)
            indices_by_type.setdefault(tok.type_name, []).append(i)
            # Track local index within this type
            local_idx = _type_count.get(tok.type_name, 0)
            _type_count[tok.type_name] = local_idx + 1
            # If this token is a group root, record the mapping
            if tok.is_group_root:
                action_root_map.append((tok.type_name, local_idx))

        # Derive action_mask from action_group_ids (backward compatible)
        action_mask = action_group_ids >= 0


        # Create per-type feature tensors — use torch.tensor on
        # the list-of-lists directly (single C-level conversion) instead of
        # an element-wise double Python loop.
        type_features: Dict[str, torch.Tensor] = {}
        for tname, feat_lists in features_by_type.items():
            raw_dim = self._raw_dims[tname]
            # Fast path: all feature lists have the expected dimension
            # (guaranteed by plan-based walk).  torch.tensor handles the
            # list-of-lists → 2D tensor conversion in C.
            # NO ragged fallback: the original silently zero-padded/truncated a
            # ragged list, which masks the exact bug class the dense engine port
            # cannot survive (a type whose feature width varies per instance).
            # A width disagreement is a schema bug — let it raise.
            tensor = torch.tensor(feat_lists, dtype=torch.float32)
            assert tensor.shape[1] == raw_dim, (
                f"token type {tname!r} produced width {tensor.shape[1]}, "
                f"registry declares {raw_dim} — a dataclass field set that "
                f"varies per instance cannot be given a static spec")
            type_features[tname] = tensor

        # Pre-convert indices_by_type to LongTensors so that downstream
        # _build_embeddings can use index_copy_ without repeated
        # torch.tensor() creation (~825K calls avoided per training run).
        indices_by_type_t: Dict[str, torch.Tensor] = {
            tname: torch.tensor(idxs, dtype=torch.long)
            for tname, idxs in indices_by_type.items()
        }

        return {
            "type_features": type_features,
            "type_ids": type_ids,
            "action_mask": action_mask,
            "action_group_ids": action_group_ids,
            "is_group_root": is_group_root,
            "intra_group_pos": intra_group_pos,
            "num_action_groups": num_action_groups,
            "seq_len": seq_len,
            "type_names": list(self._type_names),
            "token_type_names": token_type_names,
            "indices_by_type": indices_by_type_t,
            "_action_root_map": action_root_map,
        }

    @property
    def type_name_to_id(self) -> Dict[str, int]:
        assert self._type_name_to_id is not None, "Call build() first"
        return dict(self._type_name_to_id)

    @property
    def type_names(self) -> List[str]:
        assert self._type_names is not None, "Call build() first"
        return list(self._type_names)

    @property
    def raw_dims(self) -> Dict[str, int]:
        assert self._raw_dims is not None, "Call build() first"
        return dict(self._raw_dims)

    # ------------------------------------------------------------------
    # Type plan cache
    # ------------------------------------------------------------------

    def _get_plan(self, tp: type) -> _TypePlan:
        """Return cached type plan, creating it on first encounter."""
        plan = self._type_plans.get(tp)
        if plan is not None:
            return plan
        plan = self._build_type_plan(tp)
        self._type_plans[tp] = plan
        return plan

    def _build_type_plan(self, tp: type) -> _TypePlan:
        """Introspect a dataclass type *once* and cache the results."""
        if not is_dataclass(tp):
            return _TypePlan(
                type_name=tp.__name__,
                is_action_base=False,
                field_plans=[],
                has_primitives=False,
                is_dc=False,
            )

        type_hints = get_type_hints(tp)
        is_action = isinstance(tp, type) and issubclass(tp, self.action_base)

        field_plans: List[_FieldPlan] = []
        has_prims = False

        for f in fields(tp):
            ftype = type_hints.get(f.name, f.type)
            ftype = _unwrap_optional(ftype)

            # Detect ActionKey[T] wrapper — unwrap to inner type, mark as key field
            is_ak = get_origin(ftype) is ActionKey
            if is_ak:
                inner = get_args(ftype)
                ftype = inner[0] if inner else int

            if _is_primitive(ftype):
                has_prims = True
                field_plans.append(_FieldPlan(
                    name=f.name,
                    category=_CAT_PRIMITIVE,
                    is_action_key=is_ak,
                    to_float_fn=_make_to_float_fn(ftype),
                ))
            elif _is_list_of_dataclass(ftype):
                field_plans.append(_FieldPlan(
                    name=f.name,
                    category=_CAT_LIST_DC,
                    is_ordered=_is_ordered_list(ftype),
                ))
            elif is_dataclass(ftype):
                field_plans.append(_FieldPlan(
                    name=f.name,
                    category=_CAT_SINGLE_DC,
                ))
            else:
                field_plans.append(_FieldPlan(
                    name=f.name,
                    category=_CAT_SKIP,
                ))

        return _TypePlan(
            type_name=tp.__name__,
            is_action_base=is_action,
            field_plans=field_plans,
            has_primitives=has_prims,
        )

    # ------------------------------------------------------------------
    # Walk (uses cached plans)
    # ------------------------------------------------------------------

    def _walk(
        self,
        obj: Any,
        tokens: List[TokenInfo],
        is_root: bool = False,
        action_group_id: int = -1,
        counter: List[int] | None = None,
        ordered_pos_counter: List[int] | None = None,
        position_override: int = -1,
    ) -> None:
        """Recursively walk the dataclass tree, creating one token per object.

        Uses ``_TypePlan`` cache — zero reflection after the first encounter
        of each type.

        Parameters
        ----------
        action_group_id:
            -1 means "not inside an action sub-tree".  ≥ 0 means this token
            belongs to the action group with that id (propagated to children).
        counter:
            A mutable ``[int]`` used to allocate fresh group ids.
        ordered_pos_counter:
            A mutable ``[int]`` tracking the next intra-group position for
            items originating from ``OrderedList`` fields.  Shared across
            an action group so that positions are globally sequential.
            ``None`` when not inside an action group.
        position_override:
            If ≥ 0, use this as the token's ``intra_group_position`` instead
            of the default 0.  Used for ordered-list items.
        """
        plan = self._get_plan(type(obj))
        if not plan.is_dataclass:
            return

        # Detect whether *this* object is an Action root
        is_action_root = plan.is_action_base and action_group_id == -1

        # If we just entered an action sub-tree, allocate a new group id
        if is_action_root:
            action_group_id = counter[0]
            counter[0] += 1
            ordered_pos_counter = [1]

        # Collect primitive features and nested objects using the cached plan
        primitives: List[float] = []
        nested_objs: List[Any] = []
        nested_lists: List[tuple[List[Any], bool]] = []

        for fp in plan.field_plans:
            cat = fp.category
            if cat == _CAT_PRIMITIVE:
                primitives.append(fp.to_float_fn(getattr(obj, fp.name)))
            elif cat == _CAT_LIST_DC:
                value = getattr(obj, fp.name)
                if value is not None:
                    nested_lists.append((value, fp.is_ordered))
            elif cat == _CAT_SINGLE_DC:
                value = getattr(obj, fp.name)
                if value is not None:
                    nested_objs.append(value)
            # _CAT_SKIP: nothing to do

        # Add this object as a token (skip root container if requested)
        if not (is_root and self.skip_root):
            if primitives:
                intra_pos = position_override if position_override >= 0 else 0
                tokens.append(TokenInfo(
                    type_name=plan.type_name,
                    features=primitives,
                    is_action=(action_group_id >= 0),
                    source_obj=obj,
                    action_group_id=action_group_id,
                    is_group_root=is_action_root,
                    intra_group_position=intra_pos,
                ))

        # Recurse into nested single objects (position 0, no ordering)
        for nested in nested_objs:
            self._walk(nested, tokens,
                       action_group_id=action_group_id, counter=counter,
                       ordered_pos_counter=ordered_pos_counter)

        # Recurse into nested lists
        for lst, is_ordered in nested_lists:
            for item in lst:
                if item is None:
                    continue
                if is_ordered and action_group_id >= 0 and ordered_pos_counter is not None:
                    pos = ordered_pos_counter[0]
                    ordered_pos_counter[0] += 1
                    self._walk(item, tokens,
                               action_group_id=action_group_id, counter=counter,
                               ordered_pos_counter=ordered_pos_counter,
                               position_override=pos)
                else:
                    self._walk(item, tokens,
                               action_group_id=action_group_id, counter=counter,
                               ordered_pos_counter=ordered_pos_counter)

    def _discover_types(self, tokens: List[TokenInfo]) -> None:
        """Build the type registry from the first sample."""
        seen: Dict[str, int] = {}
        dims: Dict[str, int] = {}
        for tok in tokens:
            if tok.type_name not in seen:
                seen[tok.type_name] = len(seen)
                dims[tok.type_name] = len(tok.features)
            else:
                # Validate consistent feature dim
                assert dims[tok.type_name] == len(tok.features), (
                    f"Inconsistent feature dim for {tok.type_name}: "
                    f"expected {dims[tok.type_name]}, got {len(tok.features)}"
                )
        self._type_name_to_id = seen
        self._type_names = list(seen.keys())
        self._raw_dims = dims

    def _assert_registry(self, tokens: List[TokenInfo]) -> None:
        """Require every token type to be one already in the registry.

        THE CLOSED-UNIVERSE RULE. Upstream this method GREW the registry and
        fired callbacks so the net could add projections mid-run
        (`extend_for_new_type`). That is unrepresentable on the DynaPlex engine:
        `spec()` is computed before the engine build, lands in the SampleSet
        manifest and the agent sidecar, and the slabs are already allocated —
        190 worker worlds each hold a featurizer clone and there is no channel
        to grow a slab across the seam. A type universe that can grow at
        runtime therefore cannot be given a static spec at all, so discovering
        one late is a MODELLING ERROR and must be loud here rather than a
        silent shape change 200 lines downstream.

        The universe is enumerable up front: it is the set of dataclasses
        reachable from the observation root's type annotations. Declare them
        all in the schema (see sat_schema.TOKEN_SCHEMA); if an action type is
        genuinely state-dependent, it still needs its own always-present
        writer field with its own max count.
        """
        for tok in tokens:
            if tok.type_name not in self._type_name_to_id:
                raise TypeError(
                    f"token type {tok.type_name!r} (dim={len(tok.features)}) appeared "
                    f"after the registry was fixed as {self._type_names!r}. The type "
                    f"universe must be closed — every token type has to be reachable "
                    f"from the observation root on EVERY state, so that spec() can be "
                    f"declared once. Add it to the schema, or emit it unconditionally "
                    f"with a zero-count when absent.")
            declared = self._raw_dims[tok.type_name]
            if declared != len(tok.features):
                raise TypeError(
                    f"token type {tok.type_name!r} produced {len(tok.features)} features "
                    f"but the registry declares {declared} — a per-instance-varying "
                    f"field set cannot be given a static spec.")


# ============================================================================
# HELPERS
# ============================================================================

def _unwrap_optional(tp: Type) -> Type:
    from typing import Union
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _is_primitive(tp: Type) -> bool:
    """int / float / bool / Enum are feature columns. Anything else is not.

    `str` is REJECTED here, not accepted. Upstream listed it as primitive and
    then handed it to `float()`, so a str field died with an opaque ValueError
    deep inside the walk; and there is no representation for it downstream
    anyway (writer slabs are float32, and DynaML has no strings). Failing at
    plan time names the offending field instead.
    """
    if tp is str:
        raise TypeError(
            "str is not a valid token feature: token columns are numeric "
            "(the writer slabs are float32 and DynaML has no string type). "
            "Use an Enum for a categorical, or drop the field.")
    if tp in (int, float, bool):
        return True
    try:
        if issubclass(tp, Enum):
            return True
    except TypeError:
        # not a class (e.g. a bare `list`, `dict`, np.ndarray annotation):
        # not primitive, and _build_type_plan will categorise it as SKIP.
        pass
    return False


def _is_list_of_dataclass(tp: Type) -> bool:
    origin = get_origin(tp)
    if origin in (list, List):
        args = get_args(tp)
        if len(args) == 1 and is_dataclass(args[0]):
            return True
    # Also match OrderedList[T] — it's a subclass of list
    if isinstance(tp, type) and issubclass(tp, OrderedList):
        # bare OrderedList without subscript — shouldn't happen in practice
        return False
    # Check if it's OrderedList[SomeDataclass]
    if hasattr(tp, "__origin__") and isinstance(tp.__origin__, type) and issubclass(tp.__origin__, OrderedList):
        args = get_args(tp)
        if len(args) == 1 and is_dataclass(args[0]):
            return True
    return False


def _is_ordered_list(tp: Type) -> bool:
    """Return True if ``tp`` is ``OrderedList[T]`` (order-sensitive)."""
    if hasattr(tp, "__origin__") and isinstance(tp.__origin__, type) and issubclass(tp.__origin__, OrderedList):
        return True
    if isinstance(tp, type) and issubclass(tp, OrderedList):
        return True
    return False


def _to_float(value: Any, tp: Type) -> float:
    if isinstance(value, Enum):
        return float(list(type(value)).index(value))
    if tp == bool:
        return float(value)
    return float(value)

