from contextlib import contextmanager
from shutil import rmtree
from typing import Literal

import numpy as np
from daisy import Block
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from scipy.ndimage import map_coordinates
from skimage import registration

from volara.blockwise import BlockwiseTask
from volara.datasets import Dataset, Labels, Raw
from volara.utils import PydanticCoordinate


class ComputeShift(BlockwiseTask):
    task_type: Literal["compute_shift"] = "compute_shift"
    intensities: Raw
    shifts: Raw
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    mask: Labels | None = None
    overlap_ratio: float | None = None
    target: Raw | None = None
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
        in_data = self.intensities.array("r")
        # TODO: avoid hardcoding channels first :/
        # TODO: avoid assuming channel dim exists in intensities :/
        units = in_data.units
        axis_names = in_data.axis_names[-in_data.voxel_size.dims :]
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
            units=units,
            axis_names=[in_data.axis_names[0], "axis", *axis_names],
            types=[in_data.types[0], "axis", *in_data.types[1:]],
            dtype=np.float32,
        )

    @staticmethod
    def compute_shift(
        array: np.ndarray,
        target_data: np.ndarray,
        voxel_size: np.ndarray,
        mask: np.ndarray | None = None,
        overlap_ratio: float | None = None,
    ) -> np.ndarray:
        C, Z, Y, X = array.shape
        assert target_data.shape == (1, Z, Y, X) or target_data.shape == (Z, Y, X)
        if target_data.shape == (1, Z, Y, X):
            target_data = target_data[0]

        shift_data = np.zeros((C, 3, 1, 1, 1))
        if mask is not None and mask.max() == 0:
            return shift_data

        for c in range(0, C):
            # compute shift between fixed image and moving image
            shift_xyz, error, diffphase = registration.phase_cross_correlation(
                reference_image=target_data,
                moving_image=array[c],
                upsample_factor=3,
                moving_mask=mask,
                reference_mask=mask,
                overlap_ratio=overlap_ratio,
            )
            shift_data[c, :, 0, 0, 0] = shift_xyz * voxel_size
        return shift_data

    @contextmanager
    def process_block_func(self):
        # TODO: read from in_array_config
        in_array = self.intensities.array("r")
        out_array = self.shifts.array("a")

        if self.target is not None:
            target_array = self.target.array("r")

        if self.mask is not None:
            mask_array = self.mask.array("r")

        def process_block(block: Block):
            valid_read_roi = block.read_roi.intersect(in_array.roi)
            in_data = in_array.to_ndarray(roi=valid_read_roi, fill_value=0)
            if self.target is not None:
                target_data = target_array.to_ndarray(roi=valid_read_roi, fill_value=0)
            else:
                target_data = in_data[0]
            if self.mask is not None:
                mask_data = mask_array.to_ndarray(roi=valid_read_roi, fill_value=0)
            else:
                mask_data = None
            shifts = self.compute_shift(
                in_data,
                target_data,
                np.array(self.voxel_size),
                mask_data,
                self.overlap_ratio,
            )
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
    interp_shifts: Raw | None = None
    shift_threshold: float | None = None

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
        rmtree(self.aligned.store)
        if self.interp_shifts is not None:
            rmtree(self.interp_shifts.store)

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        in_array = self.intensities.array()

        voxel_shape = self.write_roi.shape / self.voxel_size

        self.aligned.prepare(
            shape=(in_array.shape[0], *voxel_shape),
            chunk_shape=(
                in_array.shape[0],
                *self.block_size,
            ),
            offset=self.write_roi.offset,
            voxel_size=self.voxel_size,
            units=self.intensities.units,
            axis_names=in_array.axis_names,
            dtype=np.uint8,
        )

        # TODO: avoid hardcoding channels first and assuming channel dim exists in intensities :/

        if self.interp_shifts is not None:
            self.interp_shifts.prepare(
                shape=(in_array.shape[0], in_array.voxel_size.dims, *voxel_shape),
                chunk_shape=(
                    in_array.shape[0],
                    in_array.voxel_size.dims,
                    *self.block_size,
                ),
                offset=self.write_roi.offset,
                voxel_size=self.voxel_size,
                units=self.intensities.units,
                axis_names=[in_array.axis_names[0], "axis", *in_array.axis_names[1:]],
                dtype=np.float32,
            )

    @staticmethod
    def apply_shift(
        intensities: np.ndarray,
        shifts: np.ndarray,
        voxel_write_roi: Roi,
        voxel_size: np.ndarray,
        shift_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        C, Z, Y, X = intensities.shape
        DZYX = 3  # shift in Z, Y, X
        BZ, BY, BX = 3, 3, 3
        assert shifts.shape == (C, DZYX, BZ, BY, BX)

        aligned = np.zeros((C, *voxel_write_roi.shape), dtype=intensities.dtype)
        interpolated_shifts = np.zeros(
            (C, DZYX, *voxel_write_roi.shape), dtype=np.float32
        )

        for c in range(0, C):
            channel_shifts = shifts[c]
            if shift_threshold is not None:
                max_shift = np.repeat(
                    np.abs(channel_shifts).max(axis=0, keepdims=True), 3, axis=0
                )
                mask = max_shift > shift_threshold
                channel_shifts[mask] = 0

            coordinates = np.meshgrid(
                np.linspace(0.0, 2.0, 3),
                *[np.linspace(2 / 3, 4 / 3, axis_len // 3) for axis_len in [Z, Y, X]],
                indexing="ij",
            )
            stacked_coordinates = np.stack(coordinates)

            # Interpolate the distances to the original pixel coordinates
            interp_shifts = map_coordinates(
                channel_shifts,
                coordinates=stacked_coordinates,
                order=1,
            )

            interpolated_shifts[c] = interp_shifts

            coordinates = np.meshgrid(
                *[
                    np.linspace(axis_len // 3, (2 * axis_len) // 3 - 1, axis_len // 3)
                    for axis_len in [Z, Y, X]
                ],
                indexing="ij",
            )
            stacked_coordinates = np.stack(coordinates)

            interpolated_coords = stacked_coordinates - (
                interp_shifts / voxel_size.reshape(-1, *((1,) * len(voxel_size)))
            )
            aligned_intensities = map_coordinates(
                intensities[c], interpolated_coords, order=1
            )
            aligned[c] = aligned_intensities
        return aligned, interpolated_shifts

    @contextmanager
    def process_block_func(self):
        # TODO: read from in_array_config
        in_array = self.intensities.array("r")
        shift_array = self.shifts.array("r")
        out_array = self.aligned.array("a")
        if self.interp_shifts is not None:
            out_interp_shifts = self.interp_shifts.array("a")

        def process_block(block: Block):
            in_shift = shift_array.to_ndarray(roi=block.read_roi, fill_value=0)

            in_data = in_array.to_ndarray(roi=block.read_roi, fill_value=0)
            aligned, interp_shifts = self.apply_shift(
                in_data,
                in_shift,
                block.write_roi / self.voxel_size,
                np.array(self.voxel_size),
                self.shift_threshold,
            )
            aligned = np.clip(
                aligned * 255,
                0,
                255,
            ).astype(np.uint8)
            aligned_array = Array(
                aligned,
                offset=block.write_roi.offset,
                voxel_size=in_array.voxel_size,
            )
            write_roi = block.write_roi.intersect(out_array.roi)
            write_data = aligned_array.to_ndarray(write_roi)
            out_array[write_roi] = write_data

            if self.interp_shifts is not None:
                interp_shifts_array = Array(
                    interp_shifts,
                    offset=block.write_roi.offset,
                    voxel_size=in_array.voxel_size,
                )
                write_data = interp_shifts_array.to_ndarray(write_roi)
                out_interp_shifts[write_roi] = write_data

        yield process_block
