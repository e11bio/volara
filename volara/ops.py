"""Ordered lazy operations for a :class:`~volara.datasets.Dataset`.

Each keyword field (``ome_norm``, ``scale_shift``, ``stack``, ``channels``, ``flip``) becomes one of
the named ops here; ``Dataset.ops`` is a list of plain callables applied after all of them.
``Dataset.resolved_ops`` assembles the two.

The order is a constraint, not a preference: ``OmeNormalize`` indexes the channel axis so it must
run BEFORE ``SelectChannels`` collapses it, and ``ReverseAxes`` names axes of the collapsed array so
it must run AFTER. A named op stays introspectable -- a consumer can see that a dataset reverses z,
which a cloudpickled callable does not expose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
from funlib.persistence import Array

from .utils import StrictBaseModel


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
    """Bind ``indices`` at construction, so each op in the loop holds its own."""
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
        arr.lazy_op(
            tuple(
                slice(None, None, -1) if i in rev else slice(None) for i in range(ndim)
            )
        )


class OmeNormalize(DatasetOp):
    """Scale each channel into [0, 1] using OMERO window bounds. Needs the channel axis PRESENT."""

    op: Literal["ome_normalize"] = "ome_normalize"
    metadata: Path | str

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Every ``(min, max)`` window the metadata lists, in its own channel order."""
        import zarr

        metadata_group = zarr.open_group(str(self.metadata))
        omero: dict = metadata_group.attrs["omero"]  # type: ignore[assignment]
        channels_meta: list[dict] = omero["channels"]
        return [(c["window"]["min"], c["window"]["max"]) for c in channels_meta]

    def windows(self, channels: int) -> list[tuple[float, float]]:
        """The first ``channels`` bounds; OMERO metadata often lists more than a store carries."""
        bounds = self.bounds
        if len(bounds) < channels:
            raise ValueError(
                f"{self.metadata} lists {len(bounds)} OMERO channels, too few for the "
                f"{channels}-channel array it normalizes."
            )
        return bounds[:channels]

    def apply(self, arr: Array) -> None:
        def _norm(data):
            data = data.astype(np.float32)
            c, *shape = data.shape
            windows = self.windows(c)
            shape1 = (c, *((1,) * len(shape)))
            shift = np.array([lo for lo, _ in windows], np.float32).reshape(shape1)
            scale = np.array([hi - lo for lo, hi in windows], np.float32).reshape(
                shape1
            )
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
    """Concatenate another dataset along axis 0.

    ``other`` is ``Any`` because typing it as a ``Dataset`` would import ``volara.datasets``, which
    imports this module; the check `apply` makes is what the annotation cannot.
    """

    op: Literal["stack_with"] = "stack_with"
    other: Any

    def apply(self, arr: Array) -> None:
        if not hasattr(self.other, "array"):
            raise TypeError(
                f"stack_with needs a Dataset to read from, got {type(self.other).__name__}."
            )
        other = self.other.array("r").data
        arr.lazy_op(lambda d: np.concatenate([d, other], axis=0))


def apply_op(op, arr: Array) -> None:
    """Apply a named op or a bare callable to ``arr``."""
    if isinstance(op, DatasetOp):
        op.apply(arr)
    else:
        arr.lazy_op(op)
