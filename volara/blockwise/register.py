from contextlib import contextmanager
from shutil import rmtree
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from scipy.ndimage import map_coordinates
from skimage import registration

from volara.blockwise import BlockwiseTask
from volara.datasets import Dataset, Raw
from volara.utils import PydanticCoordinate
from augment.augment import apply_transformation

from daisy import Block


class ComputeShift(BlockwiseTask):
    task_type: Literal["compute_shift"] = "compute_shift"
    intensities: Raw
    shifts: Raw
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    fit: Literal["overhang"] = "overhang"
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
    def context_size(self):
        return self.context * self.voxel_size

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
                self.intensities.array().shape[0],
                len(self.voxel_size),
                *self.write_roi.shape.ceil_division(self.write_size),
            ),
            chunk_shape=(
                self.intensities.array().shape[0],
                len(self.voxel_size),
                *(1,) * len(self.voxel_size),
            ),
            offset=self.write_roi.offset,
            voxel_size=self.write_size,
            units=self.intensities.units,
            dtype=np.float32,
            kwargs=self.shifts.attrs,
        )

    @staticmethod
    def compute_shift(array: np.ndarray):
        C, Z, Y, X = array.shape

        shift_data = np.zeros((C, 3, 1, 1, 1))

        for c in range(1, C):
            # compute shift between fixed image and moving image
            shift_xyz, error, diffphase = registration.phase_cross_correlation(
                reference_image=array[0],
                moving_image=array[c],
                upsample_factor=10,
            )
            shift_data[c, :, 0, 0, 0] = shift_xyz
        return shift_data

    @contextmanager
    def process_block_func(self):
        # TODO: read from in_array_config
        in_array = self.intensities.array("r")
        out_array = self.shifts.array("a")

        def process_block(block: Block):
            valid_read_roi = block.read_roi.intersect(in_array.roi)
            in_data = in_array.to_ndarray(roi=valid_read_roi, fill_value=0)
            shifts = self.compute_shift(in_data)
            shift_array = Array(
                shifts,
                offset=block.write_roi.offset,
                voxel_size=block.write_roi.shape,
            )
            write_data = shift_array.to_ndarray(block.write_roi)
            out_array[block.write_roi] = write_data

        yield process_block


class ApplyShift(BlockwiseTask):
    task_type: Literal["apply_shifts"] = "apply_shifts"
    intensities: Raw
    shifts: Raw
    aligned: Raw
    fit: Literal["overhang"] = "overhang"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.aligned.name}-{self.task_type}"

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
    def context(self):
        return self.shifts.array("r").voxel_size // self.voxel_size

    @property
    def block_size(self) -> Coordinate:
        return self.shifts.array("r").voxel_size // self.voxel_size

    @property
    def context_size(self):
        return self.context * self.voxel_size

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
        self.aligned.prepare(
            shape=self.intensities.array().shape,
            chunk_shape=(
                self.intensities.array().shape[0],
                *self.block_size,
            ),
            offset=self.write_roi.offset,
            voxel_size=self.voxel_size,
            units=self.intensities.units,
            dtype=np.uint8,
            kwargs=self.aligned.attrs,
        )

    @staticmethod
    def apply_shift(intensities: np.ndarray, shifts: np.ndarray, voxel_write_roi: Roi):
        C, Z, Y, X = intensities.shape
        DZYX = 3  # shift in Z, Y, X
        BZ, BY, BX = 3, 3, 3
        assert shifts.shape == (C, DZYX, BZ, BY, BX)

        aligned = np.zeros((C, *voxel_write_roi.shape), dtype=intensities.dtype)

        for c in range(0, C):
            coordinates = np.meshgrid(
                np.linspace(0.0, 2.0, 3),
                *[np.linspace(2 / 3, 4 / 3, axis_len // 3) for axis_len in [Z, Y, X]],
                indexing="ij",
            )
            coordinates = np.stack(coordinates)

            # Interpolate the distances to the original pixel coordinates
            interpolated_shifts = map_coordinates(
                shifts[c],
                coordinates=coordinates,
                order=3,
            )

            print(
                "interpolated_shifts",
                interpolated_shifts.shape,
                interpolated_shifts[:, 32, 200, 200],
            )
            print("expected_shift", shifts[c, :, 1, 1, 1])

            coordinates = np.meshgrid(
                *[
                    np.linspace(axis_len // 3, (2 * axis_len) // 3 - 1, axis_len // 3)
                    for axis_len in [Z, Y, X]
                ],
                indexing="ij",
            )
            coordinates = np.stack(coordinates)

            interpolated_shifts += coordinates
            aligned_intensities = map_coordinates(
                intensities[c], interpolated_shifts, order=3
            )
            print("og_intensities", intensities[c].shape, intensities[c][96, 600, 600])
            print(
                "aligned_intensities",
                aligned_intensities.shape,
                aligned_intensities[32, 200, 200],
            )
            if c == 0:
                assert np.allclose(
                    aligned_intensities, intensities[c, 64:128, 400:800, 400:800]
                )
            aligned[c] = aligned_intensities
        return aligned

    @contextmanager
    def process_block_func(self):
        # TODO: read from in_array_config
        in_array = self.intensities.array("r")
        shift_array = self.shifts.array("r")
        out_array = self.aligned.array("a")

        def process_block(block: Block):
            in_data = in_array.to_ndarray(roi=block.read_roi, fill_value=0)
            in_shift = shift_array.to_ndarray(roi=block.read_roi, fill_value=0)
            aligned = self.apply_shift(
                in_data, in_shift, block.write_roi / self.voxel_size
            )
            aligned_array = Array(
                aligned,
                offset=block.write_roi.offset,
                voxel_size=in_array.voxel_size,
            )
            write_roi = block.write_roi.intersect(out_array.roi)
            write_data = aligned_array.to_ndarray(write_roi)
            out_array[write_roi] = write_data

        yield process_block
