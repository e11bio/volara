from contextlib import contextmanager
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from skimage.exposure import equalize_adapthist

from volara.blockwise.blockwise import BlockwiseTask
from volara.datasets import Dataset, Raw
from volara.utils import PydanticCoordinate


class CLAHE(BlockwiseTask):
    task_type: Literal["clahe"] = "clahe"

    in_arr: Raw
    out_arr: Raw

    block_size: PydanticCoordinate
    kernel: PydanticCoordinate

    clip_limit: float = 0.01

    fit: Literal["overhang"] = "overhang"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.out_arr.name}-{self.task_type}"

    @property
    def context(self) -> Coordinate:
        return self.kernel // 2

    @property
    def write_roi(self) -> Roi:
        roi = self.in_arr.array("r").roi
        if self.roi is not None:
            roi = roi.intersect(self.roi)
        return roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.in_arr.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.voxel_size

    @property
    def context_size(self) -> Coordinate:
        return self.voxel_size * self.context

    def drop_artifacts(self):
        self.out_arr.drop()

    def init(self):
        if self.out_arr is not None:
            in_data = self.in_arr.array("r")
            self.out_arr.prepare(
                in_data.shape,
                in_data.chunk_shape,
                in_data.roi.offset,
                voxel_size=in_data.voxel_size,
                units=in_data.units,
                axis_names=in_data.axis_names,
                types=in_data.types,
                dtype=np.uint8,
            )

    @contextmanager
    def process_block_func(self):
        in_arr = self.in_arr.array("r")
        out_arr = Dataset.array(self.out_arr, "r+")

        def process_block(block):
            # compute in read roi
            data = in_arr.to_ndarray(block.read_roi)
            if data.ndim == self.kernel.dims:
                data = equalize_adapthist(
                    data,
                    clip_limit=self.clip_limit,
                    kernel_size=self.kernel,
                )
            else:
                data = np.stack(
                    [
                        equalize_adapthist(
                            data[i],
                            clip_limit=self.clip_limit,
                            kernel_size=self.kernel,
                        )
                        for i in range(data.shape[0])
                    ],
                    axis=0,
                )

            # crop to write roi
            block_out_roi = block.write_roi.intersect(out_arr.roi)
            array = Array(data, block.read_roi.get_begin(), in_arr.voxel_size)
            write_data = array[block_out_roi]

            out_arr[block_out_roi] = write_data * 255

        yield process_block
