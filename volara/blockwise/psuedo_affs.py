from contextlib import contextmanager
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array

from ..dataset import Affs, Dataset, Raw
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask


class PsuedoAff(BlockwiseTask):
    task_type: Literal["psuedo_affs"] = "psuedo_affs"
    embedding_data: Raw
    affs_data: Affs
    block_size: PydanticCoordinate

    @property
    def task_name(self) -> str:
        return f"{self.embedding_data.name}-{self.task_type}"

    @property
    def fit(self):
        return "shrink"

    @property
    def read_write_conflict(self):
        return False

    @property
    def write_roi(self) -> Roi:
        total_roi = self.embedding_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.embedding_data.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.voxel_size

    @property
    def context_size(self) -> tuple[Coordinate, Coordinate]:
        context_low = (
            Coordinate(
                [min(axis) for axis in zip(*self.affs_data.neighborhood)]
            )
            * self.voxel_size
        )

        context_high = (
            PydanticCoordinate(
                [max(axis) for axis in zip(*self.affs_data.neighborhood)]
            )
            * self.voxel_size
        )
        return context_low, context_high

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.affs_data]

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        # get data from in_array
        voxel_size = self.embedding_data.array("r").voxel_size

        self.affs_data.prepare(
            self.write_roi,
            voxel_size,
            self.write_size * voxel_size,
            self._out_array_dtype,
            len(self.affs_data.neighborhood),
            kwargs=self.affs_data.attrs,
        )

    @staticmethod
    def compute_psuedo_affs(
        array: np.ndarray, neighborhood: list[list[int]], norm=False, eps=1e-8
    ):
        cosine_sims = []
        for offset in neighborhood:
            offset_slices = [slice(None)] + [
                slice(max(o, 0), o if o < 0 else None) for o in offset
            ]
            base_slices = [slice(None)] + [
                slice(max(-o, 0), -o if o > 0 else None) for o in offset
            ]
            offset_array = array[tuple(offset_slices)]
            base_array = array[tuple(base_slices)]

            dot_prod = np.sum(base_array * offset_array, axis=0)
            if norm:
                dot_prod /= np.maximum(
                    np.linalg.norm(base_array, axis=0)
                    * np.linalg.norm(offset_array, axis=0),
                    eps,
                )

            dot_prod = np.pad(
                dot_prod,
                [(abs(o) if o < 0 else 0, o if o > 0 else 0) for o in offset],
                mode="constant",
            )

            cosine_sims.append(dot_prod)

        cosine_sim = np.array(cosine_sims)
        return cosine_sim

    @contextmanager
    def process_block_func(self):
        # TODO: read from in_array_config
        in_array = self.embedding_data.array("r")
        out_array = self.affs_data.array("a")

        def process_block(block):
            in_data = in_array.to_ndarray(roi=block.read_roi, fill_value=0)

            in_data = in_data.astype(np.float32) / 127.5 - 1.0
            affs_data = self.compute_psuedo_affs(in_data, self.affs_data.neighborhood)

            affs_array = Array(
                affs_data,
                offset=block.read_roi.offset,
                voxel_size=in_array.voxel_size,
            )

            write_data = affs_array.to_ndarray(block.write_roi)

            out_array[block.write_roi] = np.clip(write_data * 256, 0, 255).astype(
                np.uint8
            )

        yield process_block
