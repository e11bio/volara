import logging
from contextlib import contextmanager
from shutil import rmtree
from typing import Annotated, Literal

import daisy
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from pydantic import Field
from scipy.ndimage import median_filter
from skimage.measure import label as relabel
from skimage.morphology import remove_small_objects

from ..datasets import Labels, Raw
from ..dbs import PostgreSQL, SQLite
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

logger = logging.getLogger(__file__)


class BinaryExtractFrags(BlockwiseTask):
    task_type: Literal["binary-extract-frags"] = "binary-extract-frags"
    db: Annotated[
        PostgreSQL | SQLite,
        Field(discriminator="db_type"),
    ]
    binary_probs_data: Raw
    frags_data: Labels
    mask_data: Raw | None = None
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    save_intensities: dict[str, Raw] | None = None
    remove_debris: int = 0

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint64)

    @property
    def task_name(self) -> str:
        return f"{self.frags_data.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.binary_probs_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.binary_probs_data.array("r").voxel_size

    @property
    def context_size(self) -> Coordinate:
        return self.context * self.binary_probs_data.array("r").voxel_size

    @property
    def num_voxels_in_block(self) -> int:
        return int(np.prod(self.block_size))

    @property
    def voxel_size(self) -> Coordinate:
        return self.binary_probs_data.array("r").voxel_size

    def drop_artifacts(self):
        try:
            rmtree(self.frags_data.store)
        except FileNotFoundError:
            pass
        self.db.drop()

    def init(self):
        self.db.init()
        self.init_out_array()

    def init_out_array(self):
        self.frags_data.prepare(
            self.write_roi.shape / self.voxel_size,
            self.write_size / self.voxel_size,
            self.write_roi.offset,
            self.voxel_size,
            self._out_array_dtype,
            kwargs=self.frags_data.attrs,
        )

    def get_fragments(self, binary_probs):
        fragments_data = self.compute_fragments(binary_probs)

        # remove small debris
        if self.remove_debris > 0:
            fragments_dtype = fragments_data.dtype
            fragments_data = fragments_data.astype(np.int64)
            fragments_data = remove_small_objects(
                fragments_data, min_size=self.remove_debris
            )
            fragments_data = fragments_data.astype(fragments_dtype)

        return fragments_data

    def apply_median_filter(self, image, size=3):
        filtered_image = np.zeros_like(image)
        for c in range(image.shape[0]):
            filtered_image[c] = median_filter(image[c], size=size)
        return filtered_image

    def compute_channel_codes(self, image):
        C, Z, Y, X = image.shape
        image_flat = image.reshape(C, -1)
        # generate bit positions
        bits = 1 << np.arange(C)[::-1]
        # compute codes
        codes = np.dot(bits, image_flat).reshape(Z, Y, X)
        return codes

    def compute_fragments(self, binary_probs):
        # todo: add size parameter
        binary_probs = self.apply_median_filter(binary_probs, size=3)

        # todo: add channel threshold parameter
        threshold_probs = (binary_probs == 1).astype(np.uint8)

        channel_codes = self.compute_channel_codes(threshold_probs)

        # todo: change to 26-connectivity relabeling?
        fragments_data = relabel(channel_codes, connectivity=1)

        return fragments_data

    def generate_fragments_in_block(
        self,
        block: daisy.Block,
        binary_probs: Array,
        frags: Array,
        rag_provider,
        mask: Array | None = None,
    ):
        # todo: simplify or break into more functions

        binary_probs_data = binary_probs.to_ndarray(block.read_roi, fill_value=0)

        if binary_probs.dtype == np.uint8:
            max_probs_value = 255.0
            binary_probs_data = binary_probs_data.astype(np.float64)
        else:
            max_probs_value = 1.0

        if binary_probs_data.max() < 1e-3:
            return

        binary_probs_data /= max_probs_value

        if mask is not None:
            logger.debug("reading mask from %s", block.read_roi)
            mask_data = mask.to_ndarray(block.read_roi, fill_value=0)

            if len(mask_data.shape) == block.read_roi.dims + 1:
                # assume masking with raw data where data > 0
                mask_data = (np.min(mask_data, axis=0) > 0).astype(np.uint8)

            if np.max(mask_data) == 255:
                # should be ones
                mask_data = (mask_data > 0).astype(np.uint8)

            logger.debug("masking affinities")
            binary_probs_data *= mask_data

        fragments_data = self.get_fragments(binary_probs_data)

        logger.info(f"fragments data: {fragments_data}")

        fragments = Array(
            fragments_data, offset=block.read_roi.offset, voxel_size=frags.voxel_size
        )

        # crop fragments to write_roi
        fragments_data = fragments.to_ndarray(block.write_roi)
        max_id = fragments_data.max()

        fragments_data, max_id = relabel(fragments_data, return_num=True)
        assert max_id < self.num_voxels_in_block, f"max_id: {max_id}"

        # ensure unique IDs
        id_bump = block.block_id[1] * self.num_voxels_in_block
        fragments_data[fragments_data > 0] += id_bump

        # store fragments
        frags[block.write_roi] = fragments_data

    @contextmanager
    def process_block_func(self):
        binary_probs = self.binary_probs_data.array("r")
        frags = self.frags_data.array("r+")
        mask = self.mask_data.array("r") if self.mask_data else None

        rag_provider = self.db.open("r+")

        def process_block(block):
            self.generate_fragments_in_block(
                block,
                binary_probs,
                frags,
                rag_provider,
                mask=mask,
            )

        yield process_block
