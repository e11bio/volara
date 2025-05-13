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
    """
    The dataset containing raw probabilities for which you want to
    compute the argmax.
    """
    sem_data: Labels
    """
    The dataset in which we will store the final semantic labels.
    """
    combine_classes: list[list[int]] | None = None
    """
    A list of lists containing the ids to combine. All channels in `combine_classes[i]`
    will be summed into a new channel `i` before computing the argmax.
    """
    block_size: PydanticCoordinate
    """
    The block size with which to chunk our argmax task.
    """
    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.sem_data.name}-{self.task_type}"

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
        shape = self.write_roi.shape // self.voxel_size
        chunk_shape = self.write_size // self.voxel_size

        in_data = self.probs_data.array("r")

        self.sem_data.prepare(
            shape,
            chunk_shape,
            self.write_roi.offset,
            self.voxel_size,
            units=in_data.units,
            axis_names=in_data.axis_names[1:],
            types=in_data.types[1:],
            dtype=self._out_array_dtype,
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
