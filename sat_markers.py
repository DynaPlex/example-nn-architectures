"""Shared base classes and type markers for dataclass-based MDPs."""
from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar, List

T = TypeVar("T")


class Action(ABC):
    """Marker base class for action dataclasses."""

    pass


class GlobalState(ABC):
    """Marker base class for global state dataclasses."""

    pass


class OrderedList(List[T]):
    """Marker type for lists where item order matters.

    Use as a type annotation on dataclass fields::

        @dataclass
        class MyAction(Action):
            steps: OrderedList[Step]    # order-sensitive → positional embeddings
            items: list[Item]           # order-invariant → no positional signal

    The ``SequenceBuilder`` detects this annotation and assigns incrementing
    ``intra_group_position`` values (1, 2, 3, …) to the list items.  Plain
    ``list`` fields get position 0 for all items (same as root), making the
    positional embedding a no-op.
    """
    pass


class ActionKey(Generic[T]):
    """Marker type for action-key fields on Action dataclasses.

    Use as a type annotation on one or more fields of an Action dataclass
    to declare which field(s) form the action key passed to
    ``modify_state_with_action``::

        @dataclass(slots=True)
        class ScheduleAction(Action):
            job_id: ActionKey[int]      # key component 1
            machine_id: ActionKey[int]  # key component 2
            estimated_time: float       # feature-only field

    Resolution order in ``resolve_action()``:

    1. Fields annotated ``ActionKey[T]`` → scalar (one field) or tuple (multiple,
       in declaration order).
    2. First declared primitive field (int/float/bool/Enum) → raw value off the
       instance.
    3. Fallback → the group index ``g`` itself.

    The annotated field still contributes to the Transformer feature vector;
    the annotation is purely a key-extraction hint.
    """
    pass


