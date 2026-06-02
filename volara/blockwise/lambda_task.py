from contextlib import contextmanager
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi

from ..datasets import Dataset, PydanticDataset
from ..utils import PydanticCallable, PydanticCoordinate
from .blockwise import BlockwiseTask


class LambdaTask(BlockwiseTask):
    """
    Generic blockwise task that applies a lambda function, operating on a numpy array, to a dataset.
    """

    task_type: Literal["lambda"] = "lambda"
    in_data: PydanticDataset
    """
    The dataset to apply the lambda function to.
    """
    out_data: PydanticDataset
    """
    The output dataset after applying the lambda function.
    """
    lambda_func: PydanticCallable
    """
    The lambda function to apply to your dataset.
    """
    out_array_dtype: np.dtype | None = None
    """
    The dtype of the output array. If None, it will be the same as the input array.
    """
    init_out: bool = True
    """
    If True (default), ``init`` calls ``out_data.prepare(...)`` sized to
    ``write_roi``. Set to False when ``out_data`` already exists on disk
    (e.g. a larger volume that this task patches a window of) -- the task
    will then skip preparation and write into the existing array in place.
    With ``init_out=False`` the iteration ROI also switches from
    ``in_data.roi`` to ``out_data.roi`` so the task scans the destination;
    blocks that don't overlap ``in_data`` are skipped at process time.
    """
    task_name_suffix: str | None = None
    """
    Optional suffix appended to ``task_name`` to disambiguate multiple
    invocations that share an ``out_data`` (e.g. when patching several
    disjoint windows of a pre-existing destination with ``init_out=False``).
    Daisy keys its block-done cache on ``task_name``, so without a suffix
    a second run would inherit the first run's meta dir and either skip
    legitimately-new blocks or raise on offset mismatches.
    """
    block_size: PydanticCoordinate | None = None
    """
    Block size in voxels. When omitted, defaults to ``out_data``'s
    chunk shape if it exists on disk (the usual case for
    ``init_out=False``), else falls back to ``in_data``'s chunk shape.
    """
    fit: str = "shrink"
    read_write_conflict: bool = False

    @property
    def task_name(self) -> str:
        base = f"{self.out_data.name}-{self.task_type}"
        if self.task_name_suffix is not None:
            return f"{base}-{self.task_name_suffix}"
        return base

    @property
    def write_roi(self) -> Roi:
        if self.init_out:
            total_roi = self.in_data.array("r").roi
        else:
            total_roi = self.out_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(self.roi)
        return total_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.in_data.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        if self.block_size is not None:
            block_voxels = Coordinate(self.block_size)
        else:
            try:
                block_voxels = Coordinate(self.out_data.array("r").chunk_shape)
            except FileNotFoundError:
                block_voxels = Coordinate(self.in_data.array("r").chunk_shape)
        return block_voxels * self.voxel_size

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.write_size.dims)

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.out_data]

    def drop_artifacts(self):
        if self.init_out:
            self.out_data.drop()

    def init(self):
        if self.out_array_dtype is not None:
            self._out_array_dtype = self.out_array_dtype
        else:
            self._out_array_dtype = self.in_data.array("r").dtype
        if self.init_out:
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
            write_roi = block.write_roi.intersect(source.roi).intersect(destination.roi)
            if write_roi.empty:
                return
            destination[write_roi] = self.lambda_func(source[write_roi])

        yield process_block
