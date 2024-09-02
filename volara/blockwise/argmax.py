from contextlib import contextmanager
from shutil import rmtree
from typing import Literal

import numpy as np
from funlib.geometry import Coordinate, Roi

from ..datasets import Dataset, Labels, Raw
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask


class Argmax(BlockwiseTask):
    """
    A blockwise task that performs an argmax operation on a given set of
    probabilities and writes the result to a semantic segmentation dataset.
    """

    task_type: Literal["argmax"] = "argmax"
    probs_data: Raw
    sem_data: Labels
    combine_classes: list[list[int]] | None = None
    block_size: PydanticCoordinate
    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.probs_data.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.probs_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.probs_data.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.voxel_size

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.write_size.dims)

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.sem_data]

    def drop_artifacts(self):
        rmtree(self.sem_data.store)

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        # get data from in_array
        voxel_size = self.probs_data.array("r").voxel_size

        self.sem_data.prepare(
            self.write_roi,
            voxel_size,
            self.write_size,
            self._out_array_dtype,
            None,
            kwargs=self.sem_data.attrs,
        )

    def argmax_block(self, block, probabilities, semantic):
        probs = probabilities.to_ndarray(block.write_roi)
        if self.combine_classes is not None:
            combined = np.zeros(
                (len(self.combine_classes),) + probs.shape[1:], dtype=probs.dtype
            )
            for i, classes in enumerate(self.combine_classes):
                combined[i] = np.sum(probs[classes], axis=0)
            probs = combined
        semantic[block.write_roi] = np.argmax(probs, 0)

    @contextmanager
    def process_block_func(self):
        probabilities = self.probs_data.array("r")
        semantic = self.sem_data.array("r+")

        def process_block(block):
            self.argmax_block(block, probabilities, semantic)

        yield process_block
