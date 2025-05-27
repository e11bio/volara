import logging
from contextlib import contextmanager
from shutil import rmtree
from typing import Annotated, Literal

import daisy
import mwatershed as mws
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from pydantic import Field
from scipy.ndimage import gaussian_filter, measurements
from skimage.measure import label as relabel
from skimage.morphology import remove_small_objects

from ..datasets import Affs, Labels, Raw
from ..dbs import PostgreSQL, SQLite
from ..tmp import replace_values
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

logger = logging.getLogger(__file__)


class ExtractFrags(BlockwiseTask):
    """
    A task for extracting fragments from affinities.
    Internally it uses mutex watershed to agglomerate fragments.
    """

    task_type: Literal["extract-frags"] = "extract-frags"
    db: Annotated[
        PostgreSQL | SQLite,
        Field(discriminator="db_type"),
    ]
    """
    The database into which we will store node centers along with statistics
    such as fragment size, mean intensity, etc.
    """
    affs_data: Affs
    """
    The affinities dataset that we will use to extract fragments.
    """
    frags_data: Labels
    """
    The output dataset that will contain the extracted fragments.
    """
    mask_data: Raw | None = None
    """
    An optional mask that will be used to ignore some affinities.
    """
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    bias: list[float]
    """
    The merge/split bias for the affinities. This should be a vector of length equal to the
    size of the neighborhood with one bias per offset. This allows you to have a merge preferring
    bias for short range affinites and a split preferring bias for long range affinities.

    Example:
    Assuming you trained affs [(0, 1), (1, 0), (0, 4), (4, 0)] for a 2D dataset, you can set the bias
    to [-0.2, -0.2, -0.8, -0.8]. This will bias you towards merging on short range affinities and splitting on
    long range affinities which has been shown to work well.
    """
    strides: list[PydanticCoordinate] | None = None
    """
    The strides to use for each affinity offset in the mutex watershed algorithm. If you have
    long range affinities it can be heplful to ignore some percentage of them to avoid excessive
    splits, so you may want to use only every other voxel in the z direction for example.

    Example:
    Assuming you trained affs [(0, 1), (1, 0), (0, 4), (4, 0)] for a 2D dataset, you can set the strides
    to [(1, 1), (1, 1), (2, 2), (2, 2)]. This will result in only 1 in every 4 long range affinities
    being used in the mutex watershed algorithm resulting in fewer splits (assuming you biased long
    range affinities towards splitting).
    """
    sigma: PydanticCoordinate | None = None
    """
    The amplitude of the smoothing kernel to apply to the affinities before watershed.
    This can help agglomerate fragments from the inside out to avoid a small merge error
    causing a large fragment to split in half.
    """
    noise_eps: float | None = None
    """
    The amplitude of the random noise to add to the affinities before watershed. This
    also helps avoid streak like fragment artifacts from processing affinities in a fifo order.
    """
    filter_fragments: float = 0.0
    """
    The minimum average affinity value for a fragment to be considered valid. If the average
    affinity value is below this threshold the fragment will be removed.
    """
    remove_debris: int = 0
    """
    The minimum size of a fragment to be considered valid. If the fragment is smaller than this
    value it will be removed.
    """
    randomized_strides: bool = False
    """
    If using strides, you may want to switch from a grided stride to a random probability of
    filtering out an affinity. This can help avoid grid artifacts in the fragments.
    """

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint64)

    @property
    def neighborhood(self):
        return self.affs_data.neighborhood

    @property
    def task_name(self) -> str:
        return f"{self.frags_data.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.affs_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.affs_data.array("r").voxel_size

    @property
    def context_size(self) -> Coordinate:
        return self.context * self.affs_data.array("r").voxel_size

    @property
    def num_voxels_in_block(self) -> int:
        return int(np.prod(self.block_size))

    @property
    def voxel_size(self) -> Coordinate:
        return self.affs_data.array("r").voxel_size

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
        in_data = self.affs_data.array("r")
        self.frags_data.prepare(
            self.write_roi.shape / self.voxel_size,
            self.write_size / self.voxel_size,
            self.write_roi.offset,
            self.voxel_size,
            units=in_data.units,
            axis_names=in_data.axis_names[1:],
            types=in_data.types[1:],
            dtype=self._out_array_dtype,
        )

    def filter_avg_fragments(self, affs, fragments_data, filter_value):
        # tmp (think about this)
        average_affs = np.mean(affs[0:3], axis=0)

        filtered_fragments = []

        fragment_ids = np.unique(fragments_data)

        for fragment, mean in zip(
            fragment_ids, measurements.mean(average_affs, fragments_data, fragment_ids)
        ):
            if mean < filter_value:
                filtered_fragments.append(fragment)

        filtered_fragments = np.array(filtered_fragments, dtype=fragments_data.dtype)
        replace = np.zeros_like(filtered_fragments)
        replace_values(fragments_data, filtered_fragments, replace)

    def get_fragments(self, affs_data):
        fragments_data = self.compute_fragments(affs_data)

        # # mask fragments if provided
        # if mask is not None:
        #     fragments_data *= mask_data.astype(np.uint64)

        # filter fragments
        if self.filter_fragments > 0:
            self.filter_avg_fragments(affs_data, fragments_data, self.filter_fragments)

        # remove small debris
        if self.remove_debris > 0:
            fragments_dtype = fragments_data.dtype
            fragments_data = fragments_data.astype(np.int64)
            fragments_data = remove_small_objects(
                fragments_data, min_size=self.remove_debris
            )
            fragments_data = fragments_data.astype(fragments_dtype)

        return fragments_data

    def compute_fragments(self, affs_data):
        if self.sigma is not None:
            # add 0 for channel dim
            sigma = (0, *self.sigma)
        else:
            sigma = None

        # add some random noise to affs (this is particularly necessary if your affs are
        #  stored as uint8 or similar)
        # If you have many affinities of the exact same value the order they are processed
        # in may be fifo, so you can get annoying streaks.

        ### tmp comment out ###

        shift = np.zeros_like(affs_data)

        if self.noise_eps is not None:
            shift += np.random.randn(*affs_data.shape) * self.noise_eps

        #######################

        # add smoothed affs, to solve a similar issue to the random noise. We want to bias
        # towards processing the central regions of objects first.

        ### tmp comment out ###

        if sigma is not None:
            shift += gaussian_filter(affs_data, sigma=sigma) - affs_data

        #######################
        shift += np.array([self.bias]).reshape(
            (-1, *((1,) * (len(affs_data.shape) - 1)))
        )

        fragments_data = mws.agglom(
            (affs_data + shift).astype(np.float64),
            offsets=self.neighborhood,
            strides=self.strides,
            randomized_strides=self.randomized_strides,
        )

        return fragments_data

    def watershed_in_block(
        self,
        block: daisy.Block,
        affs: Array,
        frags: Array,
        rag_provider,
        mask: Array | None = None,
    ):
        # todo: simplify or break into more functions

        affs_data = affs.to_ndarray(block.read_roi, fill_value=0)

        if affs.dtype == np.uint8:
            max_affinity_value = 255.0
            affs_data = affs_data.astype(np.float64)
        else:
            max_affinity_value = 1.0

        if affs_data.max() < 1e-3:
            return

        affs_data /= max_affinity_value

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
            affs_data *= mask_data

        fragments_data = self.get_fragments(affs_data)

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

        # following only makes a difference if fragments were found
        if fragments_data.max() == 0:
            return

        fragment_ids, counts = np.unique(fragments_data, return_counts=True)
        logger.info("Found %d fragments", len(fragment_ids))
        fragment_ids, counts = zip(
            *[(f, c) for f, c in zip(fragment_ids, counts) if f > 0]
        )
        centers_of_masses = measurements.center_of_mass(
            np.ones_like(fragments_data), fragments_data, fragment_ids
        )

        fragment_centers = {
            fragment_id: {
                "center": block.write_roi.get_offset()
                + affs.voxel_size * Coordinate(center),
                "size": count,
            }
            for fragment_id, center, count in zip(
                fragment_ids, centers_of_masses, counts
            )
            if fragment_id > 0
        }

        # store nodes
        rag = rag_provider[block.write_roi]

        for node, data in fragment_centers.items():
            # centers
            node_attrs = {
                "position": data["center"],
            }

            node_attrs["size"] = int(data["size"])

            rag.add_node(int(node), **node_attrs)

        rag_provider.write_graph(
            rag,
            block.write_roi,
        )

    @contextmanager
    def process_block_func(self):
        affs = self.affs_data.array("r")
        frags = self.frags_data.array("r+")
        mask = self.mask_data.array("r") if self.mask_data else None

        rag_provider = self.db.open("r+")

        def process_block(block):
            self.watershed_in_block(
                block,
                affs,
                frags,
                rag_provider,
                mask=mask,
            )

        yield process_block
