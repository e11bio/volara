import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Literal, Sequence, Union

import numpy as np
from cloudvolume import CloudVolume
from funlib.geometry import Coordinate
from funlib.persistence import Array, open_ds, prepare_ds
from pydantic import AliasChoices, Field, model_validator

from .ops import (
    OmeNormalize,
    ReverseAxes,
    ScaleShift,
    SelectChannels,
    StackWith,
    apply_op,
)
from .paths import PydanticUPath, is_local, rm, to_upath, zarr_store
from .utils import OpenMode, PydanticCallable, PydanticCoordinate, StrictBaseModel

ArraySchedulerName = Literal["single-threaded", "synchronous", "threads", "processes"]
"""A dask scheduler name accepted by ``dask.compute(scheduler=...)``."""

_array_scheduler: ContextVar[ArraySchedulerName | None] = ContextVar(
    "volara_array_scheduler", default=None
)


@contextmanager
def array_scheduler(scheduler: ArraySchedulerName | None):
    """Open every array within this context with the given dask scheduler.

    :meth:`Dataset.array` hands the current value to ``funlib.persistence.open_ds``
    as ``scheduler``, which is what the returned ``Array`` computes its reads with.
    ``None`` (the default outside any context) leaves the choice to dask. Workers
    run under :attr:`~volara.blockwise.BlockwiseTask.array_scheduler`, which is
    ``"single-threaded"`` by default: daisy already runs one block per worker, so
    dask's thread pool would only add contention inside each worker.
    """
    token = _array_scheduler.set(scheduler)
    try:
        yield
    finally:
        _array_scheduler.reset(token)


def current_array_scheduler() -> ArraySchedulerName | None:
    """The scheduler :meth:`Dataset.array` will open arrays with right now."""
    return _array_scheduler.get()


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Dataset(StrictBaseModel, ABC):
    """
    A Dataset base class that defines the common attributes and methods
    for all dataset types.
    """

    store: PydanticUPath
    """
    Where the array lives: a local path or a URL on any fsspec filesystem
    (`s3://bucket/data.zarr/raw`). Serialised as that string.
    """

    voxel_size: PydanticCoordinate | None = None
    offset: PydanticCoordinate | None = None
    axis_names: list[str] | None = None
    units: list[str] | None = None
    writable: bool = True

    ops: list[PydanticCallable] | None = None
    """
    Extra lazy operations, applied IN ORDER and AFTER every keyword-driven op.

    Each is a plain callable ``data -> data``, cloudpickled to base64 so it reaches a worker the
    same way `LambdaTask.lambda_func` does. The escape hatch for a transform that has no keyword.

    Two limits of the callable path, both inherited from funlib:

    - A lambda or a `__main__` function is pickled BY VALUE; one imported from a module is pickled
      by reference, so the worker must be able to import that module and gets whatever version it
      has installed.
    - funlib does not update an array's roi or voxel_size for a callable op, so one that changes
      shape leaves the metadata describing the array before it. Use `channels` for that; it goes
      through funlib's index path, which does update them.
    - Validation produces a fresh function object, so a Dataset carrying `ops` is NOT equal to its
      own round trip and two rebuilds of the same JSON are not equal to each other. A consumer that
      dedupes, caches or fences on a dataset config needs the keyword fields, which do compare
      equal, not this one.

    Read-only, for the same reason: a callable makes the array unwriteable, so a write mode is
    refused up front rather than at the first block write.

    See `resolved_ops` for the order the keyword fields resolve to, and `volara.ops` for why that
    order is fixed.
    """

    flip: list[int] | None = None
    """
    Axis indices to reverse, applied AFTER `channels` has collapsed any leading axes -- so
    they index the FINAL array, not the store. For a `[T,C,Z,Y,X]` store read with
    `channels=[0, 0]` the result is `[Z,Y,X]` and `flip=[0]` reverses Z.

    Read-only: a reversed array reports `is_writeable` True but `__setitem__` raises, so a
    write mode is refused up front rather than at the first write.

    Corrects a volume acquired in a flipped orientation, so that nothing downstream needs to
    know. It is a lazy slice, not a copy, and it travels in this config -- a worker that
    rebuilds the model applies the same reversal, which an in-memory wrapper would not.
    """

    channels: list[list[int] | int] | int | None = None
    """
    We want to be able to subsample channels from a dataset. Specifically
    we often want to slice away a channel e.g. make a [C,Z,Y,X] dataset
    into a [Z,Y,X] dataset by selecting only one channel, slice specific
    channels form a dataset e.g. make a [C,Z,Y,X] dataset into a [C',Z,Y,X],
    or a combination of the two e.g. make a [T,C,Z,Y,X] dataset into a [C',Z,Y,X]
    dataset.

    Anything passed in will be passed directly to numpy indexing with `np.s_[]`
    with the exception of lists which will have each element passed to `np.s_[]`
    in sequence.

    Valid options are:
    - 0: `[C,Z,Y,X] -> [Z,Y,X]`
    - [0,0]: `[T,C,Z,Y,X] -> [Z,Y,X]`
    - [[0,1,2]]: `[C,Z,Y,X] -> [3,Z,Y,X]`
    """

    zarr_kwargs: dict = Field(default_factory=dict)

    storage_options: dict | None = Field(
        default=None, validation_alias=AliasChoices("storage_options", "s3_kwargs")
    )
    """
    fsspec filesystem options for a remote `store` (credentials, a custom
    endpoint such as Lyve Cloud, ...). `None` uses the environment.
    `s3_kwargs` is accepted as the former name.
    """

    @model_validator(mode="after")
    def _apply_storage_options(self):
        if self.storage_options:
            self.store = to_upath(self.store, **self.storage_options)
        return self

    @property
    def is_local(self) -> bool:
        return is_local(self.store)

    @property
    def prepared_store(self):
        """The object handed to prepare_ds/open_ds: the path itself when local,
        an fsspec-backed zarr store (carrying `storage_options`) otherwise."""
        return zarr_store(self.store)

    @property
    def name(self) -> str:
        """
        A name for this dataset. Often it is simply the name of the
        path provided as the store. We use it to differentiate between
        multiple runs of the same blockwise task on different data.
        """
        return self.store.name

    def drop(self) -> None:
        """Delete this dataset's store wherever it lives; a missing store is a no-op."""
        rm(self.store, recursive=True)

    def spoof(self, spoof_dir: Path):
        if not self.is_local:
            raise ValueError(
                f"Not spoofing dataset: store {self.store} is remote and spoofing symlinks"
            )

        spoof_path = spoof_dir / f"spoof_{self.name}"

        if not spoof_path.parent.exists():
            spoof_path.parent.mkdir(parents=True, exist_ok=True)

        if self.store.exists() and not self.writable:
            """
            If the store is not writable, it is an input to some task and we can
            safely read from it.
            """
            print("Symlinking", self.store)
            if not spoof_path.exists():
                spoof_path.symlink_to(
                    Path(str(self.store)).absolute(), target_is_directory=True
                )
        else:
            print("Spoofing", self.store)

        return self.__class__(
            store=spoof_dir / f"spoof_{self.name}",
            **self.model_dump(exclude={"store"}),
        )

    def prepare(
        self,
        shape: Sequence[int],
        chunk_shape: Sequence[int],
        offset: Sequence[int],
        voxel_size: Sequence[int],
        units: Sequence[str],
        axis_names: Sequence[str],
        types: Sequence[str],
        dtype,
    ) -> None:
        # prepare ds
        array = prepare_ds(
            self.prepared_store,
            shape=shape,
            offset=Coordinate(offset),
            voxel_size=Coordinate(voxel_size),
            units=units,
            axis_names=axis_names,
            types=types,
            chunk_shape=chunk_shape,
            dtype=dtype,
            mode="a",
            **self.zarr_kwargs,
        )
        array._source_data.attrs.update(self.attrs)

    def lazy_ops(self, arr: Array) -> None:
        """
        Apply any lazy operations to the array.
        By default, does nothing.
        Subclasses can override this method to apply
        specific lazy operations.

        Runs before every op in `resolved_ops`. `Raw` no longer overrides it -- its `ome_norm`,
        `scale_shift` and `stack` are named ops now, so for a subclass that calls
        `super().lazy_ops(arr)` those three move from before its own ops to after them, and for
        one that does not call super they are applied rather than suppressed. Relative to
        `channels` and `flip` this hook's position is unchanged.
        """
        pass

    def array(self, mode: OpenMode = "r") -> Array:
        if not self.writable and mode != "r":
            raise ValueError(
                f"Dataset {self.store} is not writable, cannot open in mode other than 'r'."
            )

        metadata = {
            "voxel_size": self.voxel_size if self.voxel_size is not None else None,
            "offset": self.offset if self.offset is not None else None,
            "axis_names": self.axis_names if self.axis_names is not None else None,
            "units": self.units if self.units is not None else None,
        }
        arr = open_ds(
            self.prepared_store,
            mode=mode,
            scheduler=_array_scheduler.get(),
            **{k: v for k, v in metadata.items() if v is not None},  # type: ignore[invalid-argument-type]
            **self.zarr_kwargs,
        )

        self.lazy_ops(arr)

        ops = self.resolved_ops()
        if mode != "r":
            if any(isinstance(o, ReverseAxes) for o in ops):
                raise ValueError(
                    f"Dataset {self.store} reverses axes, which is read-only; "
                    f"cannot open in mode {mode!r}."
                )
            if self.ops:
                raise ValueError(
                    f"Dataset {self.store} has ops={len(self.ops)} callables, which make the "
                    f"array unwriteable; cannot open in mode {mode!r}."
                )
        for op in ops:
            apply_op(op, arr)
        return arr

    def resolved_ops(self) -> list:
        """Every op to apply, in order: the keyword-driven named ops, then ``ops``.

        Subclasses that add keywords of their own override this and prepend, as `Raw` does. See
        `volara.ops` for why the order is fixed.
        """
        named: list = []
        if self.channels is not None:
            named.append(SelectChannels(channels=self.channels))
        if self.flip:
            named.append(ReverseAxes(axes=self.flip))
        return named + list(self.ops or [])

    @property
    @abstractmethod
    def attrs(self):
        pass


class Raw(Dataset):
    """
    Represents a dataset containing raw intensities.
    Has support for sampling specific channels, normalizing
    with provided scale and shifting, or reading in normalization
    bounds from OMERO metadata.
    """

    dataset_type: Literal["raw"] = "raw"
    ome_norm: PydanticUPath | None = None
    scale_shift: tuple[float, float] | None = None
    stack: "PydanticDataset | None" = None

    @property
    def bounds(self) -> list[tuple[float, float]] | None:
        """The windows `ome_norm` applies: one per channel the store carries."""
        if self.ome_norm is None:
            return None
        array = open_ds(self.prepared_store, mode="r", **self.zarr_kwargs)
        return OmeNormalize(metadata=self.ome_norm).windows(array.data.shape[0])

    @property
    def attrs(self):
        attrs = {}
        if self.ome_norm:
            attrs["bounds"] = self.bounds
        return attrs

    def resolved_ops(self) -> list:
        """`Raw`'s three keywords all precede the base class's, `ome_norm` first of all."""
        named: list = []
        if self.ome_norm:
            named.append(OmeNormalize(metadata=self.ome_norm))
        if self.scale_shift is not None:
            named.append(
                ScaleShift(scale=self.scale_shift[0], shift=self.scale_shift[1])
            )
        if self.stack is not None:
            named.append(StackWith(other=self.stack))
        return named + super().resolved_ops()


class Affs(Dataset):
    """
    Represents a dataset containing affinities.
    Requires the inclusion of the neighborhood for these
    affinities.
    """

    dataset_type: Literal["affs"] = "affs"
    neighborhood: list[PydanticCoordinate] = Field(default_factory=list)

    @property
    def attrs(self):
        return {"neighborhood": self.neighborhood}

    def model_post_init(self, context):
        provided = len(self.neighborhood) > 0
        try:
            in_array = self.array("r")
        except FileNotFoundError as e:
            in_array = None
            if not provided:
                raise ValueError(
                    "Affs(..., neighborhood=?)\n"
                    "neighborhood must be provided when referencing an array that does not yet exist\n"
                ) from e
        if in_array is not None and "neighborhood" in in_array.attrs:
            neighborhood = in_array.attrs["neighborhood"]
            if not provided:
                self.neighborhood = list(Coordinate(offset) for offset in neighborhood)
            else:
                assert np.isclose(neighborhood, self.neighborhood).all(), (
                    f"(Neighborhood metadata) {neighborhood} != {self.neighborhood} (given Neighborhood)"
                )
        else:
            if not provided:
                raise ValueError(
                    "Affs(..., neighborhood=?)\n"
                    "neighborhood must be provided when referencing an affs array that does not have "
                    "a neighborhood key in the `.zattrs`"
                )
        return super().model_post_init(context)


class LSD(Dataset):
    """
    Represents a dataset containing local shape descriptors.
    """

    dataset_type: Literal["lsd"] = "lsd"

    @property
    def attrs(self):
        return {"lsds": True}


class Labels(Dataset):
    """
    Represents an integer label dataset.
    """

    dataset_type: Literal["labels"] = "labels"

    @property
    def attrs(self):
        return {}


class MaskDataset(Dataset):
    """A per-block skip mask for a blockwise task.

    Points to a zarr array in the task's *block-grid* space: one element per
    block, not per voxel. Any nonzero value means "skip this block", zero means
    "run it". The array must have one cell per block of the task
    (:attr:`BlockwiseTask.block_grid_shape`), and cell ``(0, 0, ...)`` is the
    first block of the task's ``write_roi`` whatever the context.

    Handed to :attr:`BlockwiseTask.skip_mask`, it is passed straight to daisy as
    ``Task(skip_mask=...)``: masked blocks are reported as skipped and release
    their downstream dependencies, but are never recorded as done, so the mask
    applies to the run it is passed to. A later run with a narrower mask (or
    none) runs the blocks it no longer covers, while blocks that completed stay
    done. This is the generic form of the surface-band skip: a thin predicted
    surface only touches a few blocks, so the rest can be skipped up front.

    Grid-space, not voxel-space: the inherited voxel-metadata fields are unused
    and the array may have any integer or bool dtype; daisy casts it. Read it
    like any other dataset, with :meth:`array`. How the mask is built is left to
    the caller.
    """

    dataset_type: Literal["mask"] = "mask"

    @property
    def attrs(self):
        return {}


class _CloudVolumeView:
    """
    Zero-based, numpy-like view of a CloudVolume.

    CloudVolume indexes in absolute voxel coordinates; dask indexes from 0.
    Doing the shift here (instead of via ``getitem=`` on ``da.from_array``)
    lets dask use its default getter, which fuses slices into the read. That
    keeps the task graph small and each read fetches exactly the requested ROI.
    """

    def __init__(self, vol: CloudVolume):
        self.vol = vol
        self.shape = tuple(int(s) for s in vol.shape)  # type: ignore[unresolved-attribute]
        self.dtype = np.dtype(vol.dtype)  # type: ignore[unresolved-attribute]
        self.ndim = len(self.shape)
        # last dimension in CV is always channel and has no offset
        self._offset = tuple(int(o) for o in vol.voxel_offset) + (0,)  # type: ignore[unresolved-attribute]

    def __getitem__(self, index) -> np.ndarray:
        if not isinstance(index, tuple):
            index = (index,)
        shifted = tuple(
            slice(
                (s.start or 0) + o,
                (self.shape[i] if s.stop is None else s.stop) + o,
                s.step,
            )
            if isinstance(s, slice)
            else slice(s + o, s + o + 1)
            for i, (s, o) in enumerate(zip(index, self._offset))
        )
        data = np.asarray(self.vol[shifted])
        # CloudVolume always returns all dims; drop the ones indexed by an int
        # so we behave like a numpy array.
        squeeze = tuple(slice(None) if isinstance(s, slice) else 0 for s in index)
        return data[squeeze]


class CloudVolumeWrapper(Dataset):
    """
    Represents a volumetric dataset through Cloud Volume.
    """

    dataset_type: Literal["cloudvolume"] = "cloudvolume"
    mip: int = 0
    timestamp: int = int(time.time())  # default to current time
    agglomerate: bool = True
    data_name: str | None = None
    fill_missing: bool = False

    def array(self, mode: OpenMode = "r") -> Array:
        import dask.array as da

        # This override never reaches the lazy-op path, so nothing in `resolved_ops` can take
        # effect here. Refusing is loud; ignoring would be a silently different array from the one
        # the same config produces for every other Dataset.
        unsupported = [
            n
            for n in ("flip", "channels", "ops")
            if getattr(self, n, None) not in (None, [])
        ]
        if unsupported:
            raise NotImplementedError(
                f"CloudVolumeWrapper does not apply {', '.join(unsupported)}; it overrides "
                f"array() without the lazy-op path. Drop the field or use a zarr-backed Dataset."
            )

        vol = CloudVolume(
            str(self.store),
            mip=self.mip,
            use_https=True,
            agglomerate=self.agglomerate,
            timestamp=self.timestamp,
            fill_missing=self.fill_missing,
        )

        metadata = {
            "axis_names": self.axis_names if self.axis_names is not None else None,
            "units": self.units if self.units is not None else None,
            "offset": self.offset if self.offset is not None else vol.voxel_offset,  # type: ignore[unresolved-attribute]
            "types": ["space" for _ in range(len(vol.shape) - 1)]  # type: ignore[unresolved-attribute]
            + ["channel"],  # last dimension in CV is always channel
        }

        view = _CloudVolumeView(vol)
        # Coarse dask grid (8x the storage chunks). Because the default getter
        # fuses slices into the read, this only shrinks the task graph (e.g.
        # ~20k keys instead of ~10M for FlyWire-sized volumes); it never causes
        # over-reading. With one dask chunk per storage chunk, dask spent ~4s
        # culling the graph on every read regardless of ROI size.
        chunks = tuple(int(c) * 8 for c in vol.chunk_size) + (int(vol.num_channels),)  # type: ignore[unresolved-attribute]
        dask_arr = da.from_array(
            view,
            chunks=chunks,
            meta=np.empty((0,) * view.ndim, dtype=view.dtype),
        )

        return Array(
            dask_arr,  # type: ignore
            **{k: v for k, v in metadata.items() if v is not None},  # type: ignore
        )

    @property
    def name(self) -> str:
        return (
            self.data_name
            if self.data_name
            else str(self.store).rstrip("/").split("/")[-1]
        )

    @property
    def attrs(self):
        return {}


PydanticDataset = Annotated[
    Union[Raw, Affs, LSD, Labels, MaskDataset, CloudVolumeWrapper],
    Field(discriminator="dataset_type"),
]

# `Raw.stack` is annotated with this union, which is defined below `Raw` and so
# reaches it as a string. Pydantic would resolve the forward reference on first
# use; resolving it here keeps `Raw` complete from import.
Raw.model_rebuild()
