import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
from typing import Annotated, Literal, Sequence, Union

import numpy as np
import zarr
from cloudvolume import CloudVolume
from funlib.geometry import Coordinate
from funlib.persistence import Array, open_ds, prepare_ds
from pydantic import Field

from .utils import OpenMode, PydanticCoordinate, StrictBaseModel

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Dataset(StrictBaseModel, ABC):
    """
    A Dataset base class that defines the common attributes and methods
    for all dataset types.
    """

    store: Path | str

    voxel_size: PydanticCoordinate | None = None
    offset: PydanticCoordinate | None = None
    axis_names: list[str] | None = None
    units: list[str] | None = None
    writable: bool = True

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

    s3_kwargs: dict | None = None
    """
    Optional config for S3-compatible stores like Lyve Cloud
    Leave as None for existing local / standard S3 behavior
    """

    def _s3fs(self):
        import s3fs  # type: ignore[unresolved-import]

        if self.s3_kwargs is None:
            return s3fs.S3FileSystem()

        return s3fs.S3FileSystem(**self.s3_kwargs)

    @property
    def prepared_store(self):
        """
        Return the actual object passed to prepare_ds/open_ds.

        Existing behavior:
        - Path -> Path
        - normal s3:// string -> string

        Custom S3 behavior:
        - s3:// string + s3_kwargs -> fsspec mapper
        """
        if (
            self.s3_kwargs is not None
            and isinstance(self.store, str)
            and self.store.startswith("s3://")
        ):
            return self._s3fs().get_mapper(self.store, check=False)

        return self.store

    @property
    def name(self) -> str:
        """
        A name for this dataset. Often it is simply the name of the
        path provided as the store. We use it to differentiate between
        multiple runs of the same blockwise task on different data.
        """
        if isinstance(self.store, Path):
            return self.store.name
        else:
            return self.store.rstrip("/").split("/")[-1]

    def drop(self) -> None:
        """
        Delete this dataset.
        """
        if not isinstance(self.store, Path):
            if isinstance(self.store, str) and self.store.startswith("s3://"):
                # drop an s3 zarr
                fs = self._s3fs()
                try:
                    fs.rm(self.store, recursive=True)
                except FileNotFoundError:
                    pass
            else:
                raise ValueError(
                    f"Not dropping dataset: store {self.store} is not a Path or s3 path"
                )
        elif self.store.exists():
            rmtree(self.store)

    def spoof(self, spoof_dir: Path):
        if not isinstance(self.store, Path):
            raise ValueError(f"Not spoofing dataset: store {self.store} is not a Path")

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
                spoof_path.symlink_to(self.store.absolute(), target_is_directory=True)
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

        if self.channels is not None:
            if isinstance(self.channels, list):
                for channels in self.channels:
                    if isinstance(channels, list):
                        arr.lazy_op(lambda d: d[channels])
                    else:
                        arr.lazy_op(np.s_[channels])
            else:
                arr.lazy_op(np.s_[self.channels])

        if self.flip:
            if mode != "r":
                raise ValueError(
                    f"Dataset {self.store} has flip={self.flip}, which is read-only; "
                    f"cannot open in mode {mode!r}."
                )
            ndim = len(arr.shape)
            bad = [a for a in self.flip if not 0 <= a < ndim]
            if bad:
                raise ValueError(
                    f"flip={self.flip} has axes {bad} out of range for the {ndim}-D array "
                    f"remaining after channels={self.channels}. flip indexes the FINAL array."
                )
            rev = set(self.flip)
            arr.lazy_op(
                tuple(
                    slice(None, None, -1) if i in rev else slice(None) for i in range(ndim)
                )
            )
        return arr

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
    ome_norm: Path | str | None = None
    scale_shift: tuple[float, float] | None = None
    stack: Dataset | None = None

    @property
    def bounds(self) -> list[tuple[float, float]] | None:
        if self.ome_norm is not None:
            array = open_ds(self.store, mode="r", **self.zarr_kwargs)
            metadata_group = zarr.open_group(str(self.ome_norm))
            omero: dict = metadata_group.attrs["omero"]  # type: ignore[assignment]
            channels_meta: list[dict] = omero["channels"]
            bounds = [
                (channels_meta[c]["window"]["min"], channels_meta[c]["window"]["max"])
                for c in range(array.data.shape[0])
            ]
            return bounds
        else:
            return None

    @property
    def attrs(self):
        attrs = {}
        if self.ome_norm:
            attrs["bounds"] = self.bounds
        return attrs

    def lazy_ops(self, arr: Array) -> None:
        def scale_shift(data, scale_shift):
            data = data.astype(np.float32)
            scale, shift = scale_shift
            norm = data * scale + shift
            return norm

        def ome_norm(data, bounds):
            data = data.astype(np.float32)
            c, *shape = data.shape
            shift = np.array(
                [b_min for (b_min, _) in bounds], dtype=np.float32
            ).reshape(c, *((1,) * len(shape)))
            scale = np.array(
                [b_max - b_min for b_min, b_max in bounds], dtype=np.float32
            ).reshape(c, *((1,) * len(shape)))
            return (data - shift) / scale

        def stack(data, other_data):
            return np.concatenate([data, other_data], axis=0)

        if self.ome_norm:
            arr.lazy_op(lambda data: ome_norm(data, self.bounds))
        if self.scale_shift is not None:
            arr.lazy_op(lambda data: scale_shift(data, self.scale_shift))
        if self.stack is not None:
            arr.lazy_op(lambda data: stack(data, self.stack.array("r").data))  # type: ignore[possibly-missing-attribute]


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

        # This override does not call `lazy_ops` and does not apply `channels`, so neither can
        # take effect here. Refusing is loud; ignoring them would be a silently different array
        # from the one the same config produces for every other Dataset.
        unsupported = [
            n for n in ("flip", "channels") if getattr(self, n, None) not in (None, [])
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

        # da.from_array indexes from 0, but CloudVolume reads at absolute voxel
        # coords; shift each spatial slice by voxel_offset so volumes with a
        # non-zero offset are read at the correct location.
        offset = tuple(int(o) for o in vol.voxel_offset)  # type: ignore[unresolved-attribute]

        def getitem(a, index):
            idx = tuple(
                slice(
                    (s.start or 0) + (offset[i] if i < len(offset) else 0),
                    (s.stop if s.stop is not None else a.shape[i])
                    + (offset[i] if i < len(offset) else 0),
                    s.step,
                )
                if isinstance(s, slice)
                else s + (offset[i] if i < len(offset) else 0)
                for i, s in enumerate(index)
            )
            return a[idx]

        chunks = tuple(int(c) for c in vol.chunk_size) + (int(vol.num_channels),)  # type: ignore[unresolved-attribute]
        dask_arr = da.from_array(
            vol,
            chunks=chunks,
            getitem=getitem,
            meta=np.empty((0,) * len(vol.shape), dtype=vol.dtype),  # type: ignore[unresolved-attribute]
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
    Union[Raw, Affs, LSD, Labels, CloudVolumeWrapper],
    Field(discriminator="dataset_type"),
]
