import inspect
import logging
import time
from abc import ABC, abstractmethod
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

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def stored_layout(array: Array) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """``(chunks, shards)`` of the zarr array behind a funlib ``Array``, over every axis;
    ``shards`` is ``None`` for an unsharded array. Read off zarr itself, so it is right on a
    funlib without sharding support too. A non-zarr source has chunks only."""
    source = array._source_data
    chunks = tuple(int(c) for c in getattr(source, "chunks", array.chunk_shape))
    shards = getattr(source, "shards", None)
    return chunks, None if shards is None else tuple(int(s) for s in shards)


def stored_unit_of(array: Array) -> tuple[int, ...]:
    """The smallest region a block may write on its own: the shard when the array is sharded,
    else the chunk. Over every axis of the array."""
    chunks, shards = stored_layout(array)
    return shards if shards is not None else chunks


def _voxels(offset, voxel_size):
    """A world offset in voxels, as an int where it is one."""
    if int(voxel_size) == 0:
        return offset
    return int(offset) // int(voxel_size) if int(offset) % int(voxel_size) == 0 else offset / voxel_size


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

    chunk_shape: PydanticCoordinate | None = None
    """
    The unit this dataset is WRITTEN in, in voxels over EVERY axis of the array, channels
    included: the zarr chunk, or the shard under `sharding`. By default it is the block the task
    writes at once (its `write_size` in voxels), which is what every task stored before this
    field existed.

    A chunk (or shard) is one stored object (one file, one S3 key). A block that writes part of
    an object reads it back, merges and rewrites it, so two blocks sharing an object race and
    one loses its data silently. A chunk smaller than the write block is therefore allowed only
    when the block is a whole number of chunks per axis; `prepare` refuses anything else before
    an array exists, and refuses an existing array written in another unit.
    """

    sharding: bool = False
    """
    Store this dataset as a zarr v3 sharded array: `chunk_shape` (or the write block) is the
    SHARD, one stored object holding `inner_chunk_shape` chunks that a reader fetches on their
    own. Writing is constrained exactly as by a plain chunk, so nothing about block sizing
    changes; reads of a window with context fetch only the inner chunks they touch. Pick
    `chunk_shape` as today, the largest block a worker writes comfortably.

    A creation option: an existing array is judged by its write unit alone, so a sharded array
    is reused by a dataset that does not say `sharding` and vice versa. Needs a
    funlib.persistence whose `prepare_ds` takes `sharding` (funkelab/funlib.persistence#40);
    an older install is refused by name.
    """

    inner_chunk_shape: PydanticCoordinate | None = None
    """
    With `sharding`, the chunk inside each shard, over every axis; must divide `chunk_shape`.
    Defaults to funlib's (64 per axis, the whole axis where the shard is smaller). Declared
    against an existing sharded array it must match its inner chunk.
    """

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

    @model_validator(mode="after")
    def _check_storage_layout(self):
        for name in ("chunk_shape", "inner_chunk_shape"):
            v = getattr(self, name)
            if v is not None and any(int(c) <= 0 for c in v):
                raise ValueError(
                    f"Dataset {self.store}: {name} must be positive per axis, got {tuple(v)}."
                )
        if self.inner_chunk_shape is None:
            return self
        if not self.sharding:
            raise ValueError(
                f"Dataset {self.store}: inner_chunk_shape={tuple(self.inner_chunk_shape)} only "
                "applies with sharding=True."
            )
        if self.chunk_shape is not None:
            if self.chunk_shape.dims != self.inner_chunk_shape.dims:
                raise ValueError(
                    f"Dataset {self.store}: chunk_shape {tuple(self.chunk_shape)} and "
                    f"inner_chunk_shape {tuple(self.inner_chunk_shape)} differ in dimensions."
                )
            if any(c % i != 0 for c, i in zip(self.chunk_shape, self.inner_chunk_shape)):
                raise ValueError(
                    f"Dataset {self.store}: chunk_shape {tuple(self.chunk_shape)} (the shard) "
                    f"must be a whole number of inner_chunk_shape "
                    f"{tuple(self.inner_chunk_shape)} per axis."
                )
        return self

    @property
    def stored_unit(self) -> Coordinate | None:
        """The unit this dataset declares it is written in (`chunk_shape`: the chunk, or the
        shard under `sharding`), or ``None`` when it declares none and the unit is whatever
        block the task writes. A write covering whole units is a plain put; one covering part
        of a unit reads it back, merges and rewrites it, and races any other writer of it."""
        return self.chunk_shape

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
        """Create this dataset's array, or reuse a compatible one, for a task to write into.

        ``chunk_shape`` here is the block the calling task writes at once, in voxels over
        every axis of ``shape``; :attr:`chunk_shape` on the Dataset decides the unit it is
        STORED in: one chunk per block (the default, and what every task did before the
        field existed) or several whole chunks -- and under :attr:`sharding` that unit is a
        shard of :attr:`inner_chunk_shape` chunks. Whichever it is, the block must be a
        whole number of stored units per axis. Anything else is refused here, before an
        array exists, because a block writing part of a stored object races every other
        block that shares it and one of them loses its data silently.

        An existing array is checked FIRST, against what is on disk: written in another
        unit than this dataset would write in, a declared inner chunk that differs, a block
        that is not whole units of it, or an origin that moved, are each refused here by
        name (``ValueError``) before funlib compares shape and dtype.
        """
        write_shape = Coordinate(chunk_shape)
        if write_shape.dims != len(shape):
            raise ValueError(
                f"Dataset {self.store}: the write block {tuple(write_shape)} must have one "
                f"element per axis of the array shape {tuple(shape)}."
            )
        stored_chunk = self.chunk_shape if self.chunk_shape is not None else write_shape
        unit_name = "shard" if self.sharding else "chunk"
        if stored_chunk.dims != write_shape.dims:
            raise ValueError(
                f"Dataset {self.store}: chunk_shape {tuple(stored_chunk)} must have one "
                f"element per axis of the array shape {tuple(shape)}, like the write block "
                f"{tuple(write_shape)}."
            )
        if any(w % u != 0 for w, u in zip(write_shape, stored_chunk)):
            raise ValueError(
                f"Dataset {self.store}: the block a task writes at once, "
                f"{tuple(write_shape)} voxels, is not a whole number of {unit_name}s of "
                f"{tuple(stored_chunk)} per axis. A {unit_name} is one stored object, so blocks "
                f"sharing one would overwrite each other's data silently; make the "
                f"task's write size a multiple of the {unit_name}, or the {unit_name} a "
                f"divisor of the write size."
            )

        kwargs = dict(self.zarr_kwargs)
        if self.sharding:
            if "sharding" not in inspect.signature(prepare_ds).parameters:
                raise RuntimeError(
                    f"Dataset {self.store} declares sharding, but the installed "
                    f"funlib.persistence cannot create a sharded array: its prepare_ds has "
                    f"no sharding parameter (needs funkelab/funlib.persistence#40)."
                )
            kwargs["sharding"] = True
            if self.inner_chunk_shape is not None:
                kwargs["inner_chunk_shape"] = self.inner_chunk_shape

        # ⛔ Against the array that is actually there, BEFORE funlib sees it: a funlib without
        # sharding support compares the inner chunk, not the write unit, and where it refuses it
        # names neither the layout nor the knob.
        try:
            existing = open_ds(self.prepared_store, mode="r", **self.zarr_kwargs)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            self._check_stored_layout(existing, write_shape, stored_chunk)
            # ⛔ funlib relabels an existing array to the offset it is given, so a task whose
            # write roi moved would have its blocks anchored off the array's own grid with no
            # error. Compared in VOXELS: a finished array may carry another voxel size than the
            # one a task prepares with (slabreg relabels its canvas to physical nm).
            if existing.roi.dims != len(tuple(offset)):
                raise ValueError(
                    f"Dataset {self.store}: the array on disk has {existing.roi.dims} spatial "
                    f"axes ({list(existing.axis_names)}), this task prepares {len(tuple(offset))}; "
                    "the axes differ, not the layout. Recreate the array."
                )
            # exact, in voxels: o_have / v_have == o_want / v_want, cross-multiplied so a
            # sub-voxel move is not rounded away
            moved = [
                d
                for d, (oh, vh, ow, vw) in enumerate(
                    zip(existing.roi.offset, existing.voxel_size, offset, voxel_size)
                )
                if int(oh) * int(vw) != int(ow) * int(vh)
            ]
            if moved:
                have = tuple(_voxels(o, v) for o, v in zip(existing.roi.offset, existing.voxel_size))
                want = tuple(_voxels(o, v) for o, v in zip(offset, voxel_size))
                raise ValueError(
                    f"Dataset {self.store}: the array on disk starts at voxel {have}, this "
                    f"task would prepare it at {want} (axes {moved} differ). Its blocks would "
                    "then sit off the array's own stored grid; recreate the array or keep the "
                    "roi it was created with."
                )

        array = prepare_ds(
            self.prepared_store,
            shape=shape,
            offset=Coordinate(offset),
            voxel_size=Coordinate(voxel_size),
            units=units,
            axis_names=axis_names,
            types=types,
            chunk_shape=stored_chunk,
            dtype=dtype,
            mode="a",
            **kwargs,
        )
        # and the array funlib handed back, created or reused, is stored as declared
        self._check_stored_layout(array, write_shape, stored_chunk)
        array._source_data.attrs.update(self.attrs)

    def _check_stored_layout(self, array: Array, write_shape, stored_chunk) -> None:
        """Refuse unless ``array`` is written in the unit this dataset would write in (its shard
        if sharded, else its chunk) and its chunk matches a declared inner chunk; ``write_shape``
        being a whole number of that unit then follows from :meth:`prepare`'s own check. A sharded
        array met by a funlib without sharding support is refused by name. See :meth:`prepare`."""
        real_chunks, real_shards = stored_layout(array)
        real_unit = real_shards if real_shards is not None else real_chunks
        if real_unit != tuple(stored_chunk):
            raise ValueError(
                f"Dataset {self.store}: the array on disk is written in "
                f"{'shards' if real_shards is not None else 'chunks'} of {real_unit}, but this "
                f"dataset would write in {'shards' if self.sharding else 'chunks'} of "
                f"{tuple(stored_chunk)}"
                + ("" if self.chunk_shape is not None else " (it declares no chunk_shape, so "
                   "the task's write block is the unit)")
                + ". Declare chunk_shape to match the array, or recreate it."
            )
        if self.inner_chunk_shape is not None and real_chunks != tuple(self.inner_chunk_shape):
            # funlib compares a declared inner chunk to the array's chunks, sharded or not
            raise ValueError(
                f"Dataset {self.store}: the array on disk has "
                f"{'inner chunks' if real_shards is not None else 'chunks (it is not sharded)'} "
                f"{real_chunks}, not the inner chunks {tuple(self.inner_chunk_shape)} this dataset "
                "declares; recreate it or declare inner_chunk_shape to match."
            )
        if real_shards is not None and "sharding" not in inspect.signature(prepare_ds).parameters:
            # this funlib compares the INNER chunk to the write block and would refuse the
            # array with a bare PermissionError; say what is going on instead
            raise RuntimeError(
                f"Dataset {self.store}: the array on disk is sharded, and the installed "
                "funlib.persistence cannot write into a sharded array through prepare_ds (it "
                "compares the inner chunk, not the shard; needs funkelab/funlib.persistence#40)."
            )

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
