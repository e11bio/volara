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
from scipy.ndimage import gaussian_filter

from ..datasets import Affs, Dataset, Labels
from ..dbs import PostgreSQL, SQLite
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

logger = logging.getLogger(__file__)


class SeededExtractFrags(BlockwiseTask):
    """
    Extract fragments from affinities using a set of skeletons as a supervising signal.
    Any voxel that intersects with a node placed on a skeleton is guaranteed to be assigned
    the label of the skeleton it intersects with. The affinities are used to fill out
    the rest of the segment to get a full volume representation of your skeletons.

    Any fragment that does not intersect with a skeleton is discarded.
    """

    task_type: Literal["seeded-extract-frags"] = "seeded-extract-frags"
    affs_data: Affs
    """
    The affinities dataset that will be used to expand the skeletons to full segments.
    """
    segs_data: Labels
    """
    The segmentations dataset that will contain the final segmentations.
    """
    mask_data: Labels | None = None
    """
    An optional mask for masking out the affinities.
    """
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    bias: list[float]
    """
    The bias terms to be used for each offset in the affinities neighborhoood.
    """
    strides: list[PydanticCoordinate] | None = None
    """
    The strides with which to filter each offset in the affinities neighborhood.
    """
    graph_db: Annotated[
        PostgreSQL | SQLite,
        Field(discriminator="db_type"),
    ]
    """
    The graph database containing the skeletons to use as a supervising signale.
    """
    randomized_strides: bool = False
    """
    Whether or not to convert the strides from a grid like filter to a random probability
    of filtering out each affinity edge.
    """

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint64)

    @property
    def task_name(self) -> str:
        return f"{self.segs_data.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        if self.roi is not None:
            return self.affs_data.array("r").roi.intersect(
                Roi(self.roi[0], self.roi[1])
            )
        else:
            return self.affs_data.array("r").roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.affs_data.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.affs_data.array("r").voxel_size

    @property
    def context_size(self) -> Coordinate:
        return self.context * self.voxel_size

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.segs_data]

    def drop_artifacts(self):
        rmtree(self.segs_data.store)

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        # get data from in_array
        in_data = self.affs_data.array("r")
        voxel_size = in_data.voxel_size

        self.segs_data.prepare(
            self.write_roi.shape / voxel_size,
            self.block_write_roi.shape / voxel_size,
            offset=self.write_roi.offset,
            voxel_size=voxel_size,
            units=in_data.units,
            axis_names=in_data.axis_names[1:],
            types=in_data.types[1:],
            dtype=self._out_array_dtype,
        )

    @contextmanager
    def process_block_func(self):
        affs_array = self.affs_data.array("r")
        segs_array = self.segs_data.array("r+")
        mask_array = self.mask_data.array("r") if self.mask_data is not None else None

        graph_provider = self.graph_db.open("r")

        def process_block(block: daisy.Block):
            affs = affs_array.to_ndarray(block.read_roi, fill_value=0)
            if mask_array is not None:
                affs *= mask_array.to_ndarray(block.read_roi, fill_value=0) > 0
            if affs.max() == 0:
                return
            graph = graph_provider.read_graph(block.read_roi)
            seeds = np.zeros(affs.shape[1:], dtype=np.uint64)
            unique_seeds = set()
            for _, node_attrs in graph.nodes(data=True):
                if "position" not in node_attrs:
                    continue
                pos = Coordinate(node_attrs["position"])
                if block.read_roi.contains(pos):
                    pos -= block.read_roi.offset
                    pos /= affs_array.voxel_size
                    seeds[tuple(pos)] = int(node_attrs["skeleton_id"])
                    unique_seeds.add(int(node_attrs["skeleton_id"]))

            if len(unique_seeds) == 0:
                return

            if affs.dtype == np.uint8:
                max_affinity_value = 255.0
                affs = affs.astype(np.float64)
            else:
                max_affinity_value = 1.0

            if affs.max() < 1e-3:
                return

            affs /= max_affinity_value

            sigma = (0, 6, 9, 9)

            random_noise = np.random.randn(*affs.shape) * 0.001

            smoothed_affs = (
                gaussian_filter(affs, sigma=sigma) - 0.5
            ) * 0.01  # todo: parameterize?

            #######################

            shift = np.array(
                self.bias,
            ).reshape((-1, *((1,) * (len(affs.shape) - 1))))

            logger.error(
                f"unique seeds ({unique_seeds}) and seed counts: {np.unique(seeds, return_counts=True)}"
            )

            segs = mws.agglom(
                affs + shift + random_noise + smoothed_affs,
                offsets=self.affs_data.neighborhood,
                strides=self.strides,
                seeds=seeds,
                randomized_strides=self.randomized_strides,
            )

            logger.error(
                f"unique seeds ({unique_seeds}) and frag counts: {np.unique(segs, return_counts=True)}"
            )

            segs = segs * np.isin(segs, list(unique_seeds))

            logger.error(
                f"unique seeds ({unique_seeds}) and seg counts: {np.unique(segs, return_counts=True)}"
            )

            segs = Array(segs, block.read_roi.offset, segs_array.voxel_size)

            # store fragments
            segs_array[block.write_roi] = segs[block.write_roi]

            logger.info(f"releasing block: {block}")

        yield process_block
