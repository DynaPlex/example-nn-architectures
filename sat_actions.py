"""Action resolution: convert a policy group-index to the MDP action key.

Eliminates ``action_to_key`` by reading the action key directly from the
Action dataclass object that was produced by ``make_action_set``/
``build_sat_observation``.

Resolution order (cached per Action type):

1. Fields annotated ``ActionKey[T]`` — returned as a scalar (one field) or
   tuple (multiple fields, in declaration order).
2. First declared primitive field (int / float / bool / Enum) — raw value off
   the instance.  Works when the first field stores the raw integer id, which
   is the de-facto convention in all flat-action MDPs.
3. Fallback — the group index ``g`` itself (e.g. ``order_picking`` where
   ``modify_state_with_action`` takes a candidate-list index).

Usage::

    from sat_actions import resolve_action

    action_key = resolve_action(seq["_actions"], chosen_group_index)
    mdp.modify_state_with_action(state, ctx, action_key)

``seq["_actions"]`` is automatically populated by ``SequenceBuilder.build()``
and holds the original Action dataclass instances in group-ID order.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Union, get_args, get_origin, get_type_hints

from sat_markers import ActionKey

# ---------------------------------------------------------------------------
# Per-type extractor cache (module-level — shared across all call sites)
# ---------------------------------------------------------------------------
_extractor_cache: Dict[type, Callable[[Any], Any] | None] = {}


def _build_extractor(tp: type) -> Callable[[Any], Any] | None:
    """Build a key-extractor function for Action type *tp* (called once, then cached)."""
    if not is_dataclass(tp):
        return None

    try:
        hints = get_type_hints(tp)
    except Exception:
        hints = {}

    dc_fields = fields(tp)

    # --- Pass 1: look for ActionKey[T]-annotated fields ---
    ak_names: List[str] = []
    for f in dc_fields:
        ftype = hints.get(f.name, type(None))
        if get_origin(ftype) is ActionKey:
            ak_names.append(f.name)

    if ak_names:
        if len(ak_names) == 1:
            name = ak_names[0]
            return lambda obj, n=name: getattr(obj, n)
        else:
            names = tuple(ak_names)
            return lambda obj, ns=names: tuple(getattr(obj, n) for n in ns)

    # --- Pass 2: first declared primitive field ---
    for f in dc_fields:
        ftype = hints.get(f.name, type(None))
        # Unwrap Optional[T]
        if get_origin(ftype) is Union:
            args = [a for a in get_args(ftype) if a is not type(None)]
            if len(args) == 1:
                ftype = args[0]
        # Primitive check: int, float, bool, or an Enum subclass
        if ftype in (int, float, bool):
            name = f.name
            return lambda obj, n=name: getattr(obj, n)
        try:
            if isinstance(ftype, type) and issubclass(ftype, Enum):
                name = f.name
                return lambda obj, n=name: getattr(obj, n)
        except TypeError:
            pass

    # --- Pass 3: no usable id field — fallback to group index ---
    return None


def resolve_action(actions: List[Any], group_index: int) -> Any:
    """Return the MDP action key for the chosen action group.

    Parameters
    ----------
    actions : list
        The flat list of Action objects from ``seq["_actions"]``.
        Populated automatically by :meth:`SequenceBuilder.build`.
    group_index : int
        The group index chosen by the policy network.

    Returns
    -------
    The action key suitable for ``mdp.modify_state_with_action(state, ctx, key)``.
    When no id field is found the function returns ``group_index`` itself,
    which is correct for MDPs whose ``modify_state_with_action`` takes the
    candidate-list index (e.g. ``order_picking``).
    """
    if not actions:
        return group_index

    action = actions[group_index]
    tp = type(action)

    if tp not in _extractor_cache:
        _extractor_cache[tp] = _build_extractor(tp)

    extractor = _extractor_cache[tp]
    if extractor is not None:
        return extractor(action)
    return group_index

