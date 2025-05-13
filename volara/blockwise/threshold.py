from contextlib import contextmanager
from shutil import rmtree
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi

from ..datasets import Dataset, Labels, Raw
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask


class Threshold(BlockwiseTask):
    """
    Blockwise threshold an array
    """

    task_type: Literal["threshold"] = "threshold"
    in_data: Raw
    """
    The dataset to threshold.
    """
    mask: Labels
    """
    The output thresholded dataset saved as a mask.
    """
    threshold: float
    """
    The threshold value to apply to your dataset.
    """
    block_size: PydanticCoordinate

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint8)

    @property
    def task_name(self) -> str:
        return f"{self.mask.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.in_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.in_data.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.voxel_size

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.write_size.dims)

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.mask]

    def drop_artifacts(self):
        rmtree(self.mask.store, ignore_errors=True)

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        in_data = self.in_data.array("r")
        self.mask.prepare(
            self.write_roi.shape / self.voxel_size,
            self.write_size / self.voxel_size,
            offset=self.write_roi.offset,
            voxel_size=self.voxel_size,
            units=in_data.units,
            axis_names=in_data.axis_names,
            types=in_data.types,
            dtype=self._out_array_dtype,
        )

    @contextmanager
    def process_block_func(self):
        frags = self.in_data.array("r")
        segs = self.mask.array("r+")

        def process_block(block):
            segs[block.write_roi] = frags[block.write_roi] > self.threshold

        yield process_block
