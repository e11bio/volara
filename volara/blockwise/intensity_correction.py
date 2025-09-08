from contextlib import contextmanager
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi

from scipy.ndimage import gaussian_filter

from ..datasets import Dataset, Raw
from .blockwise import BlockwiseTask
from ..utils import PydanticCoordinate


class FlatFieldCorrection(BlockwiseTask):
    """ """

    task_type: Literal["flat_field_correction"] = "flat_field_correction"

    intensities: Raw
    gain: Raw
    bias: Raw

    stats_block_size: PydanticCoordinate

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint64)

    @property
    def task_name(self) -> str:
        return f"{self.gain.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        roi = self.intensities.array("r").roi
        if self.roi is not None:
            roi = roi.intersect(self.roi)
        return roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.intensities.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.stats_block_size * self.voxel_size

    @property
    def context_size(self) -> Coordinate:
        return self.voxel_size * 0

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.gain, self.bias]

    def drop_artifacts(self):
        self.gain.drop()
        self.bias.drop()

    def init(self):
        in_data = self.intensities.array("r")
        self.gain.prepare(
            self.write_roi.shape.ceil_division(self.write_size),
            self.write_size / self.write_size,
            self.write_roi.offset,
            self.write_size,
            units=in_data.units,
            axis_names=in_data.axis_names,
            types=in_data.types,
            dtype=self._out_array_dtype,
        )
        self.bias.prepare(
            self.write_roi.shape.ceil_division(self.write_size),
            self.write_size / self.write_size,
            self.write_roi.offset,
            self.write_size,
            units=in_data.units,
            axis_names=in_data.axis_names,
            types=in_data.types,
            dtype=self._out_array_dtype,
        )

    @contextmanager
    def process_block_func(self):
        intensities = self.intensities.array("r")
        gain_arr = self.gain.array("r+")
        bias_arr = self.bias.array("r+")

        def process_block(block):
            data = intensities.to_ndarray(block.write_roi)

            bias = np.median(data)  # or a low-percentile, see below
            data_zb = data - bias  # zero-baselined
            data_zb[data_zb < 0] = 0  # clamp tiny negatives from noise

            gain = float(data_zb.mean())

            gain_arr[block.write_roi] = gain
            bias_arr[block.write_roi] = bias

        yield process_block


class Blur(BlockwiseTask):
    task_type: Literal["blur"] = "blur"

    in_arr: Raw
    out_arr: Raw | None = None
    sigma_grid: tuple[int, ...]
    operation: Literal["scale_mean", "subtract_mean"] | None = None

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint64)

    @property
    def task_name(self) -> str:
        return f"{self.arr}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        roi = self.arr.array("r").roi
        if self.roi is not None:
            roi = roi.intersect(self.roi)
        return roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.arr.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.arr.array("r").roi.shape

    @property
    def context_size(self) -> Coordinate:
        return self.voxel_size * 0

    @property
    def output_datasets(self) -> list[Dataset]:
        return []

    def drop_artifacts(self):
        if self.out_arr:
            self.out_arr.drop()

    def init(self):
        in_data = self.in_arr.array("r")
        if self.out_arr is not None:
            self.out_arr.prepare(
                in_data.shape,
                in_data.chunk_shape,
                in_data.roi.offset,
                voxel_size=in_data.voxel_size,
                units=in_data.units,
                axis_names=in_data.axis_names,
                types=in_data.types,
                dtype=self._out_array_dtype,
            )

    @contextmanager
    def process_block_func(self):
        in_arr = self.in_arr.array("r")
        out_arr = self.out_arr.array("r+") if self.out_arr else in_arr.array("r+")

        def process_block(block):
            data = in_arr.to_ndarray(block.write_roi)

            # fill gaps
            filler = np.nanmean(data)
            data[np.isnan(data)] = filler

            # blur and normalise
            data = gaussian_filter(data, self.sigma_grid, mode="reflect")
            if self.operation is None:
                pass
            elif self.operation == "scale_mean":
                data /= data.mean()
            elif self.operation == "subtract_mean":
                data -= data.mean()

            out_arr[block.write_roi] = data

        yield process_block
