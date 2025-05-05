import logging
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
from typing import Any, Literal, Sequence
import time
import numpy as np
import zarr
from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
from cloudvolume import CloudVolume

from .utils import PydanticCoordinate, StrictBaseModel

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Dataset(StrictBaseModel, ABC):
    """
    A Dataset base class that defines the common attributes and methods
    for all dataset types.
    """

    store: str | Path

    voxel_size: PydanticCoordinate | None = None
    offset: PydanticCoordinate | None = None
    axis_names: list[str] | None = None
    units: list[str] | None = None

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
            return self.store.split("/")[-1]

    def drop(self) -> None:
        """
        Delete this dataset
        """
        rmtree(self.store)

    def prepare(
        self,
        shape: Sequence[int],
        chunk_shape: Sequence[int],
        offset: Coordinate,
        voxel_size: Coordinate,
        units: list[str],
        axis_names: list[str],
        types: list[str],
        dtype,
        **ds_kwargs: dict[str, Any],
    ) -> None:
        # prepare ds
        array = prepare_ds(
            self.store,
            shape=shape,
            offset=offset,
            voxel_size=voxel_size,
            units=units,
            axis_names=axis_names,
            types=types,
            chunk_shape=chunk_shape,
            dtype=dtype,
            mode="a",
        )
        array._source_data.attrs.update(ds_kwargs)

    def array(self, mode: str = "r") -> Array:
        return open_ds(self.store, mode=mode)

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
    channels: list[int] | None = None
    ome_norm: Path | str | None = None
    scale_shift: tuple[float, float] | None = None
    stack: Dataset | None = None

    @property
    def bounds(self) -> list[tuple[float, float]] | None:
        if self.ome_norm is not None:
            array = open_ds(self.store, mode="r")
            metadata_group = zarr.open(self.ome_norm)
            channels_meta = metadata_group.attrs["omero"]["channels"]
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
        if self.channels is not None:
            attrs["channels"] = self.channels
        if self.ome_norm:
            attrs["bounds"] = self.bounds
        return attrs

    def array(self, mode="r"):
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

        metadata = {
            "voxel_size": self.voxel_size if self.voxel_size is not None else None,
            "offset": self.offset if self.offset is not None else None,
            "axis_names": self.axis_names if self.axis_names is not None else None,
            "units": self.units if self.units is not None else None,
        }

        array = open_ds(
            self.store,
            mode=mode,
            **{k: v for k, v in metadata.items() if v is not None},
        )

        if self.ome_norm:
            array.lazy_op(lambda data: ome_norm(data, self.bounds))
        if self.scale_shift is not None:
            array.lazy_op(lambda data: scale_shift(data, self.scale_shift))
        if self.channels is not None:
            array.lazy_op(np.s_[self.channels])
        if self.stack is not None:
            array.lazy_op(lambda data: stack(data, self.stack.array("r").data))

        return array


class Affs(Dataset):
    """
    Represents a dataset containing affinities.
    Requires the inclusion of the neighborhood for these
    affinities.
    """

    dataset_type: Literal["affs"] = "affs"
    neighborhood: list[PydanticCoordinate]

    @property
    def attrs(self):
        return {"neighborhood": self.neighborhood}


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
    data_name: str
    dataset_type: Literal["volume"] = "volume"
    mip: int = 0
    agglomerate: bool = True
    timestamp: int = int(time.time()) # default to current time

    @property
    def data(self) -> CloudVolume:
        vol = CloudVolume(self.store, mip=self.mip, use_https=True, agglomerate=self.agglomerate, timestamp=self.timestamp)
        return vol
    
    @property
    def name(self) -> str:
        return self.data_name
    
    @property
    def roi(self) -> Roi:
        shape = np.array(self.data.shape)[:-1].tolist()
        offset = self.data.voxel_offset
        self.voxel_size = Coordinate(np.array(self.data.resolution))
        return Roi(offset, shape)
    
    @property
    def attrs(self):
        return {}