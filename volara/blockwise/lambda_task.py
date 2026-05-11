from contextlib import contextmanager
from typing import Callable, Literal

import numpy as np
from funlib.geometry import Coordinate, Roi

from ..datasets import Dataset
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask


class LambdaTask(BlockwiseTask):
    """
    Generic blockwise task that applies a lambda function, operating on a numpy array, to a dataset.
    """

    task_type: Literal["lambda"] = "lambda"
    in_data: Dataset
    """
    The dataset to apply the lambda function to.
    """
    out_data: Dataset
    """
    The output dataset after applying the lambda function.
    """
    lambda_func: Callable
    """
    The lambda function to apply to your dataset.
    """
    block_size: PydanticCoordinate
    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint8)

    @property
    def task_name(self) -> str:
        return f"{self.out_data.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.in_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(self.roi)
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
        return [self.out_data]

    def drop_artifacts(self):
        self.out_data.drop()

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        in_data = self.in_data.array("r")
        self.out_data.prepare(
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
        source = self.in_data.array("r")
        destination = self.out_data.array("r+")

        def process_block(block):
            destination[block.write_roi] = self.lambda_func(source[block.write_roi])

        yield process_block
