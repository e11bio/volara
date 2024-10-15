from contextlib import contextmanager
from shutil import rmtree
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from scipy.ndimage import shift
from skimage import registration

from volara.blockwise import BlockwiseTask
from volara.datasets import Dataset, Raw
from volara.utils import PydanticCoordinate


class ComputeShift(BlockwiseTask):
    task_type: Literal["compute_shift"] = "compute_shift"
    intensities: Raw
    shifts: Raw
    block_size: PydanticCoordinate
    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.shifts.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.intensities.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.intensities.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.voxel_size

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.shifts]

    def drop_artifacts(self):
        rmtree(self.shifts.store)

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        self.shifts.prepare(
            shape=(
                self.intensities.num_channels,
                len(self.voxel_size),
                *self.write_roi.shape / self.voxel_size,
            ),
            chunk_shape=(
                self.intensities.num_channels,
                len(self.voxel_size),
                *self.block_size,
            ),
            offset=self.write_roi.offset,
            voxel_size=self.voxel_size,
            units=self.intensities.units,
            dtype=np.float32,
            kwargs=self.shifts.attrs,
        )

    @staticmethod
    def compute_shift(array: np.ndarray):
        C, Z, Y, X = array.shape

        # todo: test average of channels rather than fixed reference channel
        reference_channel_index = 0
        fixed_image = array[reference_channel_index]

        shift_data = np.zeros(C, 3, Z, Y, X)

        for c in range(C):
            if c == reference_channel_index:
                continue  # skip reference channel

            moving_image = array[c]

            # compute shift between fixed image and moving image
            shift_xyz, error, diffphase = registration.phase_cross_correlation(
                fixed_image,
                moving_image,
                upsample_factor=10,  # more fine grained
            )
            shift_data[c] = shift_xyz
        return shift_data

    @contextmanager
    def process_block_func(self):
        # TODO: read from in_array_config
        in_array = self.intensities.array("r")
        out_array = self.shifts.array("a")

        def process_block(block):
            in_data = in_array.to_ndarray(roi=block.read_roi, fill_value=0)
            shifts = self.compute_shift(in_data)
            shift_array = Array(
                shifts,
                offset=block.read_roi.offset,
                voxel_size=in_array.voxel_size,
            )
            write_data = shift_array.to_ndarray(block.write_roi)
            out_array[block.write_roi] = write_data

        yield process_block


class ApplyShift(BlockwiseTask):
    task_type: Literal["apply_shifts"] = "apply_shifts"
    intensities: Raw
    shifts: Raw
    aligned_intensities: Raw
    block_size: PydanticCoordinate
    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.aligned_intensities.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.intensities.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.intensities.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.voxel_size

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.shifts]

    def drop_artifacts(self):
        rmtree(self.shifts.store)

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        self.aligned_intensities.prepare(
            shape=self.intensities.array().shape,
            chunk_shape=(
                self.intensities.num_channels,
                *self.block_size,
            ),
            offset=self.write_roi.offset,
            voxel_size=self.voxel_size,
            units=self.intensities.units,
            dtype=np.uint8,
            kwargs=self.aligned_intensities.attrs,
        )

    @staticmethod
    def apply_shift(intensities: np.ndarray, shifts: np.ndarray):
        C, Z, Y, X = intensities.shape

        aligned = np.zeros_like(intensities)
        aligned[0] = intensities[0]

        for c in range(1, C):
            # apply shift to align moving image
            aligned_moving_image = shift(
                intensities[0],
                shift=intensities[c],
                order=1,
                mode="constant",
                cval=0.0,  # zero border
            )

            aligned[c] = aligned_moving_image
        return aligned

    @contextmanager
    def process_block_func(self):
        # TODO: read from in_array_config
        in_array = self.intensities.array("r")
        shift_array = self.shifts.array("r")
        out_array = self.aligned_intensities.array("a")

        def process_block(block):
            in_data = in_array.to_ndarray(roi=block.read_roi, fill_value=0)
            in_shift = shift_array.to_ndarray(roi=block.read_roi, fill_value=0)
            aligned = self.apply_shift(in_data, in_shift)
            aligned_array = Array(
                aligned,
                offset=block.read_roi.offset,
                voxel_size=in_array.voxel_size,
            )
            write_data = aligned_array.to_ndarray(block.write_roi)
            out_array[block.write_roi] = write_data

        yield process_block
