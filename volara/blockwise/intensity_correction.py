from contextlib import contextmanager
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from skimage.exposure import rescale_intensity
from skimage.exposure._adapthist import _clahe

from volara.blockwise.blockwise import BlockwiseTask
from volara.datasets import Dataset, Raw
from volara.utils import PydanticCoordinate


class CLAHE(BlockwiseTask):
    task_type: Literal["clahe"] = "clahe"

    in_arr: Raw
    out_arr: Raw
    mask_arr: Raw | None = None

    block_size: PydanticCoordinate
    kernel: PydanticCoordinate

    clip_limit: float = 0.01
    in_range: list[tuple[int, int]] | tuple[int, int] | str = "image"

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
                (
                    (in_data.shape[0],)
                    if in_data.types[0] not in ["time", "space"]
                    else ()
                )
                + tuple(self.block_size),
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
        mask_arr = self.mask_arr.array("r") if self.mask_arr else None

        def process_block(block):
            # compute in read roi
            read_roi = block.read_roi.intersect(in_arr.roi)
            data = in_arr.to_ndarray(read_roi)

            # rescale:
            if data.ndim == self.kernel.dims + 1 and not isinstance(self.in_range, str):
                assert len(self.in_range) == data.shape[0], (
                    "in_range must match channel dimension"
                )
                data = np.stack(
                    [
                        np.round(
                            rescale_intensity(
                                data[i],
                                out_range=(0, 2**14 - 1),
                                in_range=self.in_range[i],
                            )
                        ).astype(np.min_scalar_type(2**14))
                        for i in range(data.shape[0])
                    ],
                    axis=0,
                )
            else:
                data = np.round(
                    rescale_intensity(
                        data, out_range=(0, 2**14 - 1), in_range=self.in_range
                    )
                ).astype(np.min_scalar_type(2**14))

            if mask_arr is not None:
                mask_data = mask_arr.to_ndarray(block.read_roi)
                if data.ndim == self.kernel.dims:
                    data = data * (mask_data > 0)
                else:
                    data = data * (mask_data[None] > 0)
            if data.ndim == self.kernel.dims:
                data = _clahe(
                    data,
                    clip_limit=self.clip_limit,
                    kernel_size=self.kernel,
                    nbins=256,
                )
            else:
                data = np.stack(
                    [
                        _clahe(
                            data[i],
                            clip_limit=self.clip_limit,
                            kernel_size=self.kernel,
                            nbins=256,
                        )
                        for i in range(data.shape[0])
                    ],
                    axis=0,
                )

            # rescale to float
            data = rescale_intensity(data, out_range=(0, 1), in_range=(0, 2**14 - 1))

            # crop to write roi
            block_out_roi = block.write_roi.intersect(out_arr.roi)
            array = Array(data, read_roi.offset, in_arr.voxel_size)
            write_data = array[block_out_roi]

            out_arr[block_out_roi] = write_data * 255

        yield process_block
