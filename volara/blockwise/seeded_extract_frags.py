import logging
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from typing import Annotated, Literal

import daisy
import mwatershed as mws
import networkx as nx
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from pydantic import Field
from scipy.ndimage.filters import gaussian_filter

from ..datasets import Affs, Dataset, Labels
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
    nml_file: Path
    """
    The nml file containing the skeletons.
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
        voxel_size = self.affs_data.array("r").voxel_size

        self.segs_data.prepare(
            self.write_roi,
            voxel_size,
            self.write_size,
            self._out_array_dtype,
            None,
            kwargs=self.segs_data.attrs,
        )

    @contextmanager
    def process_block_func(self):
        affs_array = self.affs_data.array("r")
        segs_array = self.segs_data.array("r+")

        nx_graph = nml_to_networkx_graph(
            self.nml_file, voxel_size=affs_array.voxel_size
        )

        def process_block(block: daisy.Block):
            affs = affs_array.to_ndarray(block.read_roi, fill_value=0)
            graph = nx_graph
            seeds = np.zeros(affs.shape[1:], dtype=np.uint64)
            unique_seeds = set()
            for _, node_attrs in graph.nodes(data=True):
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


def nml_to_networkx_graph(
    path_to_nml, position_attribute=["z", "y", "x"], voxel_size=[400, 150, 150]
):
    tree = ET.parse(path_to_nml)
    root = tree.getroot()

    G = nx.Graph()

    # Loop through each 'thing' in the NML file (these are skeletons)
    for skeleton in root.findall("thing"):
        skeleton_id = skeleton.attrib["id"]

        # Add nodes
        nodes = skeleton.find("nodes").findall("node")

        # Skip skeleton if it contains only a single node (not sure why these exist)
        if len(nodes) <= 1:
            continue

        for node in nodes:
            node_id = node.attrib["id"]

            position = [
                float(node.attrib[p]) * v
                for p, v in zip(position_attribute, voxel_size)
            ]

            G.add_node(node_id, position=position, skeleton_id=skeleton_id)

        # Add edges
        for edge in skeleton.find("edges").findall("edge"):
            source = edge.attrib["source"]
            target = edge.attrib["target"]
            G.add_edge(source, target)

    return G
