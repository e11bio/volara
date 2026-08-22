"""Ordered lazy operations for a :class:`~volara.datasets.Dataset`.

An op is either a **named model** or a **plain callable**. Both cross the worker boundary: volara's
``PydanticCallable`` cloudpickles a callable to base64, which is how ``LambdaTask`` already ships
one. (A bare ``Callable`` annotation does not — it raises ``PydanticSerializationError`` at dump
time — so use ``PydanticCallable``.)

Prefer a named model where one fits. It is reviewable in a config file, it is stable to hash, and a
consumer can reason about it: slabreg fences raw-derived artifacts by hashing the ops that define a
slab's pixel frame, which a base64 cloudpickle blob cannot support — the bytes move with the Python
and cloudpickle versions, so every worker upgrade would look like a re-framing. Reach for a callable
when no named op fits.

Order is explicit because it changes the result. ``OmeNormalize`` indexes the channel axis and so
must run BEFORE ``SelectChannels`` collapses it; ``ReverseAxes`` names axes of the collapsed array
and so must run AFTER. Expressing that as a list makes it reviewable instead of implied by the order
of ``if`` statements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import numpy as np
from funlib.persistence import Array
from pydantic import Field

from .utils import PydanticCallable, StrictBaseModel


class DatasetOp(StrictBaseModel, ABC):
    """One lazy transform, applied to a funlib ``Array`` in place."""

    @abstractmethod
    def apply(self, arr: Array) -> None: ...


class SelectChannels(DatasetOp):
    """Collapse leading non-spatial axes, e.g. ``[T,C,Z,Y,X] -> [Z,Y,X]``.

    Each element is applied in sequence to axis 0 of the array as it stands: an ``int`` drops that
    axis, a ``list[int]`` subsamples it. Passed to ``np.s_[]`` as-is.
    """

    op: Literal["select_channels"] = "select_channels"
    channels: list[list[int] | int] | int

    def apply(self, arr: Array) -> None:
        chans = self.channels
        if isinstance(chans, list):
            for c in chans:
                if isinstance(c, list):
                    arr.lazy_op(_take(c))
                else:
                    arr.lazy_op(np.s_[c])
        else:
            arr.lazy_op(np.s_[chans])


def _take(indices: list[int]):
    """Bind ``indices`` per op. A closure over a loop variable would late-bind every op to the last."""
    return lambda d: d[indices]


class ReverseAxes(DatasetOp):
    """Reverse the named axes of the array AS IT STANDS at this point in the list.

    Read-only: a reversed array reports ``is_writeable`` True while ``__setitem__`` raises, so a
    write mode is refused when this op is present.
    """

    op: Literal["reverse_axes"] = "reverse_axes"
    axes: list[int]

    def apply(self, arr: Array) -> None:
        ndim = len(arr.shape)
        bad = [a for a in self.axes if not 0 <= a < ndim]
        if bad:
            raise ValueError(
                f"reverse_axes {bad} out of range for the {ndim}-D array at this point in the op "
                f"list. Axes name the array AS IT STANDS here, not the store."
            )
        rev = set(self.axes)
        arr.lazy_op(tuple(slice(None, None, -1) if i in rev else slice(None) for i in range(ndim)))


class OmeNormalize(DatasetOp):
    """Scale each channel into [0, 1] using OMERO window bounds. Needs the channel axis PRESENT."""

    op: Literal["ome_normalize"] = "ome_normalize"
    metadata: Path | str

    @property
    def bounds(self) -> list[tuple[float, float]]:
        import zarr

        omero = zarr.open_group(str(self.metadata)).attrs["omero"]
        return [(c["window"]["min"], c["window"]["max"]) for c in omero["channels"]]

    def apply(self, arr: Array) -> None:
        bounds = self.bounds

        def _norm(data):
            data = data.astype(np.float32)
            c, *shape = data.shape
            shift = np.array([lo for lo, _ in bounds], np.float32).reshape(c, *((1,) * len(shape)))
            scale = np.array([hi - lo for lo, hi in bounds], np.float32).reshape(
                c, *((1,) * len(shape)))
            return (data - shift) / scale

        arr.lazy_op(_norm)


class ScaleShift(DatasetOp):
    """``data * scale + shift`` in float32."""

    op: Literal["scale_shift"] = "scale_shift"
    scale: float
    shift: float

    def apply(self, arr: Array) -> None:
        scale, shift = self.scale, self.shift
        arr.lazy_op(lambda d: d.astype(np.float32) * scale + shift)


class StackWith(DatasetOp):
    """Concatenate another dataset along axis 0."""

    op: Literal["stack_with"] = "stack_with"
    other: Any

    def apply(self, arr: Array) -> None:
        other = self.other.array("r").data
        arr.lazy_op(lambda d: np.concatenate([d, other], axis=0))


NamedOp = Annotated[
    Union[SelectChannels, ReverseAxes, OmeNormalize, ScaleShift, StackWith],
    Field(discriminator="op"),
]

#: A named op, or any callable ``data -> data`` (cloudpickled to reach a worker).
#:
#: Left-to-right: a mapping carrying an ``op`` discriminator is a named op. Smart-mode would try the
#: callable branch first and die decoding a dict as base64.
AnyDatasetOp = Annotated[
    Union[NamedOp, PydanticCallable], Field(union_mode="left_to_right")
]


def apply_op(op, arr: Array) -> None:
    """Apply a named op or a bare callable to ``arr``."""
    if isinstance(op, DatasetOp):
        op.apply(arr)
    else:
        arr.lazy_op(op)
