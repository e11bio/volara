from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi

from volara.tmp import replace_values

from ..dataset import Dataset, Labels
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask


class LUT(BlockwiseTask):
    task_type: Literal["write-segments"] = "write-segments"
    frags_data: Labels
    seg_data: Labels
    lut: Path
    block_size: PydanticCoordinate

    _out_array_dtype: np.dtype = np.dtype(np.uint64)

    @property
    def task_name(self) -> str:
        return f"{self.frags_data.name}-{self.task_type}"

    @property
    def fit(self):
        return "shrink"

    @property
    def read_write_conflict(self):
        return False

    @property
    def write_roi(self) -> Roi:
        total_roi = self.frags_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.frags_data.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.voxel_size

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.write_size.dims)

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.seg_data]

    def drop_artifacts(self):
        rmtree(self.seg_data.store)

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        self.seg_data.prepare(
            self.write_roi.shape / self.voxel_size,
            self.write_size / self.voxel_size,
            self.write_roi.offset,
            self.voxel_size,
            self._out_array_dtype,
            kwargs=self.seg_data.attrs,
        )

    def map_block(self, block, frags, segs, mapping):
        segs[block.write_roi] = replace_values(
            frags.to_ndarray(block.write_roi), mapping[0], mapping[1]
        )

    @contextmanager
    def process_block_func(self):
        frags = self.frags_data.array("r")
        segs = self.seg_data.array("r+")

        mapping = np.load(Path(f"{self.lut}.npz"))["fragment_segment_lut"].astype(
            np.uint64
        )

        def process_block(block):
            self.map_block(block, frags, segs, mapping)

        yield process_block
