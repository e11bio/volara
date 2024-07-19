import logging
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
from typing import Literal, Optional, Union

import numpy as np
import zarr
from funlib.persistence import open_ds, prepare_ds

from .utils import PydanticCoordinate, StrictBaseModel

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Dataset(ABC, StrictBaseModel):
    store: Union[str, Path]

    voxel_size: Optional[PydanticCoordinate] = None
    offset: Optional[PydanticCoordinate] = None
    axis_names: Optional[list[str]] = None
    units: Optional[list[str]] = None

    @property
    def name(self):
        if isinstance(self.store, Path):
            return self.store.name
        else:
            return self.store.split("/")[-1]

    def drop(self):
        rmtree(self.store)

    def prepare(self, shape, chunk_shape, offset, voxel_size, dtype, **ds_kwargs):
        # prepare ds
        array = prepare_ds(
            self.store,
            shape=shape,
            offset=offset,
            voxel_size=voxel_size,
            chunk_shape=chunk_shape,
            dtype=dtype,
            mode="a",
        )
        array._source_data.attrs.update(ds_kwargs)

    def array(self, mode="r"):
        return open_ds(self.store, mode=mode)

    @property
    @abstractmethod
    def attrs(self):
        pass


class Raw(Dataset):
    dataset_type: Literal["raw"] = "raw"
    channels: Optional[list[int]] = None
    ome_norm: Optional[Union[Path, str]] = None
    scale_shift: Optional[tuple[float, float]] = None

    @property
    def bounds(self) -> Optional[list[tuple[float, float]]]:
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
            scale, shift = scale_shift
            norm = data * scale + shift
            return norm

        def ome_norm(data, bounds):
            norm = np.zeros(data.shape, np.float32)
            for c, (b_min, b_max) in enumerate(bounds):
                norm[c] = (data[c] - b_min) / (b_max - b_min)
            return norm

        def select_channels(data, channels):
            channel_data = data[channels]
            return channel_data

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
            array.adapt(lambda data: ome_norm(data, self.bounds))
        if self.scale_shift is not None:
            array.adapt(lambda data: scale_shift(data, self.scale_shift))
        if self.channels is not None:
            array.adapt(np.s_[self.channels])

        return array


class Affs(Dataset):
    dataset_type: Literal["affs"] = "affs"
    neighborhood: list[PydanticCoordinate]

    @property
    def attrs(self):
        return {"neighborhood": self.neighborhood}


class LSD(Dataset):
    dataset_type: Literal["lsd"] = "lsd"

    @property
    def attrs(self):
        return {"lsds": True}


class Labels(Dataset):
    dataset_type: Literal["labels"] = "labels"

    @property
    def attrs(self):
        return {}
