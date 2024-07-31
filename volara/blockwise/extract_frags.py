import logging
from contextlib import contextmanager
from shutil import rmtree
from typing import Annotated, Literal, Optional, Union

import daisy
import mwatershed as mws
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from pydantic import Field
from scipy.ndimage import measurements
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label as relabel

from ..dataset import Affs, Labels, Raw
from ..dbs import PostgreSQL, SQLite
from ..tmp import replace_values
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

logger = logging.getLogger(__file__)


class ExtractFrags(BlockwiseTask):
    task_type: Literal["extract-frags"] = "extract-frags"
    db: Annotated[
        Union[PostgreSQL, SQLite],
        Field(discriminator="db_type"),
    ]
    affs_data: Affs
    frags_data: Labels
    mask_data: Optional[Raw] = None
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    save_intensities: Optional[dict[str, Raw]] = None
    bias: list[float]
    sigma: Optional[PydanticCoordinate] = None
    noise_eps: Optional[float] = None
    filter_fragments: float = 0.0
    remove_debris: int = 0

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint64)

    @property
    def neighborhood(self):
        return self.affs_data.neighborhood

    @property
    def task_name(self) -> str:
        return f"{self.affs_data.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.affs_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi

    @property
    def write_size(self) -> PydanticCoordinate:
        return self.block_size * self.affs_data.array("r").voxel_size

    @property
    def context_size(self) -> PydanticCoordinate:
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
        self.frags_data.prepare(
            self.write_roi.shape / self.voxel_size,
            self.write_size / self.voxel_size,
            self.write_roi.offset,
            self.voxel_size,
            self._out_array_dtype,
            kwargs=self.frags_data.attrs,
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
        replace_values(fragments_data, filtered_fragments, replace, inplace=True)

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
            self.remove_small_objects(fragments_data, min_size=self.remove_debris)
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
            # strides=self.strides,
        )

        return fragments_data

    def watershed_in_block(
        self,
        block: daisy.Block,
        affs: Array,
        frags: Array,
        rag_provider,
        mask: Optional[Array] = None,
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

        save_intensities = (
            self.save_intensities if self.save_intensities is not None else {}
        )
        intensities = []
        for data_config in save_intensities.values():
            embedding_data = data_config.array("r").to_ndarray(block.write_roi)

            mean_intensities = np.stack(
                [
                    measurements.mean(embedding_data[ch], fragments_data, fragment_ids)
                    for ch in range(embedding_data.shape[0])
                ],
                axis=1,
            )
            intensities.append(mean_intensities)

        fragment_centers = {
            fragment_id: {
                **{
                    "center": block.write_roi.get_offset()
                    + affs.voxel_size * Coordinate(center),
                    "size": count,
                },
                **{
                    bar_name: bar
                    for bar_name, bar in zip(save_intensities.keys(), bars)
                },
            }
            for fragment_id, center, count, *bars in zip(
                fragment_ids, centers_of_masses, counts, *intensities
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

            for bar_name in save_intensities.keys():
                node_attrs[bar_name] = data[bar_name]

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
