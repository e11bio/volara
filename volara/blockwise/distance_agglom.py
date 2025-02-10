import logging
from contextlib import contextmanager
from itertools import chain, combinations
from typing import Annotated, Literal

import numpy as np
from funlib.geometry import Coordinate, Roi
from pydantic import Field
from scipy.ndimage import laplace
from scipy.spatial import cKDTree

from volara.lut import LUT

from ..datasets import Dataset, Labels
from ..dbs import PostgreSQL, SQLite
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

logger = logging.getLogger(__file__)


class DistanceAgglom(BlockwiseTask):
    """
    Distance based edge creation for fragment to segment agglomeration.
    In this context distance is defined as the distance between the embeddings
    of two fragments. Given some attributes saved or computable on the fragments
    we compute the distance between pairs of fragments and then save the edges
    with their costs.
    Because the distance is computed between embeddings, there is no spatial limit
    to the distance between fragments. This means that without a spatial constraint
    we would need to compute the distance between all pairs of fragments. To avoid
    this we can set a connection radius that will limit the euclidean distance between
    connectable fragments.
    """

    task_type: Literal["distance-agglom"] = "distance-agglom"

    storage: (
        Annotated[
            PostgreSQL | SQLite,
            Field(discriminator="db_type"),
        ]
        | LUT
    )
    """
    Where to store the edges and or the final Look Up Table.
    If a Database is provided, it is assumed that each fragment in the
    fragments dataset has a node in the graph already saved.
    """
    frags_data: Labels
    """
    The labels dataset that contains the fragments to agglomerate
    """
    distance_keys: list[str] | None = None
    background_intensities: list[float] | None = None
    eps: float = 1e-8
    distance_threshold: float | None = None
    distance_metric: Literal["euclidean", "cosine", "max"] = "cosine"

    block_size: PydanticCoordinate
    context: PydanticCoordinate

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.frags_data.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.frags_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(Roi(self.roi[0], self.roi[1]))
        return total_roi

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.frags_data.array("r").voxel_size

    @property
    def context_size(self) -> Coordinate:
        return self.context * self.frags_data.array("r").voxel_size

    @property
    def output_datasets(self) -> list[Dataset]:
        return []

    def drop_artifacts(self):
        self.db.drop_edges()

    def label_distances(self, labels, voxel_size, dist_threshold=0.0):
        # First 0 out all voxel where the laplace is 0 (not an edge voxel)
        output = np.zeros_like(labels, dtype=np.float32)
        object_filter = laplace(labels, output=output)
        labels *= abs(object_filter) > 0
        coordinates = np.nonzero(labels)
        labels = labels[*coordinates]
        coords = np.column_stack(coordinates) * np.array(voxel_size)

        trees = []
        for label in np.unique(labels):
            trees.append(
                (label, cKDTree(coords[labels == label]), coords[labels == label])
            )
        min_dists = {}

        for (label_a, tree_a, tree_coords_a), (
            label_b,
            tree_b,
            tree_coords_b,
        ) in combinations(trees, 2):
            pairs = tree_a.query_ball_tree(tree_b, dist_threshold)
            if len(list(chain(*pairs))) > 0:
                min_dists[(label_a, label_b)] = min(
                    np.linalg.norm(tree_coords_a[i] - tree_coords_b[j])
                    for i, matches in enumerate(pairs)
                    for j in matches
                )

        return list(min_dists.keys()), list(min_dists.values())

    def agglomerate_in_block(self, block, frags, rag_provider):
        voxel_size = frags.voxel_size
        frags = frags.to_ndarray(block.read_roi, fill_value=0)
        rag = rag_provider[block.read_roi]

        distance_threshold = (
            self.distance_threshold
            if self.distance_threshold is not None
            else min(self.context * voxel_size)
        )
        pairs, distances = self.label_distances(frags, voxel_size, distance_threshold)
        distance_keys = [] if self.distance_keys is None else self.distance_keys
        background_intensities = (
            [0.0] * len(distance_keys)
            if self.background_intensities is None
            else self.background_intensities
        )
        assert len(background_intensities) == len(distance_keys)
        for (frag_i, frag_j), dist in zip(pairs, distances):
            if frag_i in rag.nodes and frag_j in rag.nodes:
                node_attrs_a = rag.nodes[frag_i]
                node_attrs_b = rag.nodes[frag_j]
                attr_dict = {"distance": dist}
                for distance_key, background_intensity in zip(
                    distance_keys, background_intensities
                ):
                    if (
                        distance_key not in node_attrs_a
                        or distance_key not in node_attrs_b
                    ):
                        continue
                    distance_a = (
                        np.array(node_attrs_a[distance_key]) - background_intensity
                    )
                    distance_b = (
                        np.array(node_attrs_b[distance_key]) - background_intensity
                    )
                    if self.distance_metric == "cosine":
                        similarity = np.dot(distance_a, distance_b) / max(
                            np.linalg.norm(distance_a) * np.linalg.norm(distance_b),
                            self.eps,
                        )
                    elif self.distance_metric == "euclidean":
                        similarity = -np.linalg.norm(distance_a - distance_b)
                    elif self.distance_metric == "max":
                        similarity = -np.max(np.abs(distance_a - distance_b))
                    attr_dict[f"{distance_key}_similarity"] = similarity
                rag.add_edge(
                    int(frag_i),
                    int(frag_j),
                    **attr_dict,
                )

        rag_provider.write_graph(rag, block.write_roi, write_nodes=False)

    @contextmanager
    def process_block_func(self):
        frags = self.frags_data.array("r")
        rag_provider = self.db.open("r+")

        def process_block(block):
            self.agglomerate_in_block(
                block,
                frags,
                rag_provider,
            )

        yield process_block
