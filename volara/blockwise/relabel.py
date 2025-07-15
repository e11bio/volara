from contextlib import contextmanager
from shutil import rmtree
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi

from volara.lut import LUT
from volara.tmp import replace_values

from ..datasets import Dataset, Labels
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask


class Relabel(BlockwiseTask):
    """
    A task for blockwise relabelling of arrays using a lookup table from fragment
    to segment IDs.
    """

    task_type: Literal["relabel"] = "relabel"
    frags_data: Labels
    """
    The fragments dataset from which we read the fragment IDs.
    """
    seg_data: Labels
    """
    The segments dataset to which we write the relabeled segment IDs.
    """
    lut: LUT
    """
    The path to the lookup table (LUT) that maps fragment IDs to segment IDs.
    """
    block_size: PydanticCoordinate

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint64)

    @property
    def task_name(self) -> str:
        return f"{self.seg_data.name}-{self.task_type}"

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
        rmtree(self.seg_data.store, ignore_errors=True)

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        in_data = self.frags_data.array("r")
        self.seg_data.prepare(
            self.write_roi.shape / self.voxel_size,
            self.write_size / self.voxel_size,
            self.write_roi.offset,
            self.voxel_size,
            units=in_data.units,
            axis_names=in_data.axis_names,
            types=in_data.types,
            dtype=self._out_array_dtype,
        )

    def map_block(self, block, frags, segs, mapping):
        segs[block.write_roi] = replace_values(
            frags.to_ndarray(block.write_roi), mapping[0], mapping[1]
        )

    @contextmanager
    def process_block_func(self):
        frags = self.frags_data.array("r")
        segs = self.seg_data.array("r+")

        def process_block(block):
            mapping = self.lut.load()
            self.map_block(block, frags, segs, mapping)

        yield process_block
