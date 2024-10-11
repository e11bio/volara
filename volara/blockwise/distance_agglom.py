import logging
from contextlib import contextmanager
from itertools import chain, combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

import mwatershed as mws
import numpy as np
from funlib.geometry import Coordinate, Roi
from pydantic import Field
from scipy.ndimage import laplace
from scipy.spatial import cKDTree

from ..datasets import Dataset, Labels, Raw
from ..dbs import PostgreSQL, SQLite
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

logger = logging.getLogger(__file__)


class DistanceAgglom(BlockwiseTask):
    task_type: Literal["distance-agglom"] = "distance-agglom"
    db: Annotated[
        PostgreSQL | SQLite,
        Field(discriminator="db_type"),
    ]
    frags_data: Labels
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    distance_keys: list[str] | None = None
    background_intensities: list[float] | None = None
    eps: float = 1e-8
    distance_threshold: float | None = None
    distance_metric: Literal["euclidean", "cosine", "max"] = "cosine"

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


class DistanceAgglomSimple(BlockwiseTask):
    task_type: Literal["distance-agglom-simple"] = "distance-agglom-simple"
    lut: Path
    frags_data: Labels
    raw_data: Raw
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    background_intensity: float | None = None
    eps: float = 1e-8
    distance_threshold: float
    emb_distance_threshold: float = 0.05
    distance_metric: Literal["euclidean", "cosine", "max"] = "cosine"
    small_size_threshold: int = 512
    large_size_threshold: int = 512
    size_filter: int = 64

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
        self.lut.unlink(missing_ok=True)

    def label_distances(self, labels, frags_a, frags_b, voxel_size, dist_threshold=0.0):
        # First 0 out all voxel where the laplace is 0 (not an edge voxel)
        output = np.zeros_like(labels, dtype=np.float32)
        object_filter = laplace(labels, output=output)
        labels *= abs(object_filter) > 0
        coordinates = np.nonzero(labels)
        labels = labels[*coordinates]
        coords = np.column_stack(coordinates) * np.array(voxel_size)

        trees_a, trees_b = [], []
        for label in frags_a:
            trees_a.append((label, cKDTree(coords[labels == label])))
        for label in frags_b:
            trees_b.append((label, cKDTree(coords[labels == label])))

        min_dists = {}

        for (label_a, tree_a), (label_b, tree_b) in product(trees_a, trees_b):
            pairs = tree_a.query_ball_tree(tree_b, dist_threshold)
            if len(list(chain(*pairs))) > 0:
                min_dists[(label_a, label_b)] = dist_threshold

        return list(min_dists.keys())

    def agglomerate_in_block(self, block, frags, raw, tmpdir):
        voxel_size = frags.voxel_size
        intensities = raw.to_ndarray(block.read_roi, fill_value=0)
        small_frags = frags.to_ndarray(block.write_roi, fill_value=0)
        write_frags = set([x for x in np.unique(small_frags) if x != 0])
        frags = frags.to_ndarray(block.read_roi, fill_value=0)

        def calculate_label_intensities(image, labels, size_filter=64):
            unique_labels, counts = np.unique(labels, return_counts=True)
            small_frags = unique_labels[counts < size_filter]
            unique_labels, counts = (
                unique_labels[counts > size_filter],
                counts[counts > size_filter],
            )
            unique_labels = unique_labels[unique_labels != 0]  # ignore background
            label_intensities = np.zeros((len(unique_labels), image.shape[0]))
            label_sizes = np.zeros(len(unique_labels))

            for i, (label, count) in enumerate(zip(unique_labels, counts)):
                mask = labels == label
                masked_image = image[:, mask]
                label_intensities[i] = np.mean(masked_image, axis=1)  # channel wise
                label_sizes[i] = count

            return unique_labels, label_intensities, label_sizes, small_frags

        unique_labels, label_intensities, label_sizes, small_frags = (
            calculate_label_intensities(
                intensities, frags, size_filter=self.size_filter
            )
        )
        unique_label_mapping = {
            label: (intensity, size)
            for label, intensity, size in zip(
                unique_labels, label_intensities, label_sizes
            )
        }

        distance_threshold = (
            self.distance_threshold
            if self.distance_threshold is not None
            else min(self.context_size)
        )
        pairs = self.label_distances(
            frags,
            unique_labels[label_sizes > self.large_size_threshold],
            unique_labels[label_sizes < self.small_size_threshold],
            voxel_size,
            distance_threshold,
        )
        # pairs = list(
        #     product(
        #         unique_labels[label_sizes > self.large_size_threshold],
        #         unique_labels[label_sizes < self.small_size_threshold],
        #     )
        # )
        edges = []
        for frag_i, frag_j in pairs:
            if frag_i in write_frags and frag_j in write_frags:
                intensity_i, size_i = unique_label_mapping[frag_i]
                intensity_j, size_j = unique_label_mapping[frag_j]

                intensity_a, size_a = (
                    (intensity_i, size_i) if size_i < size_j else (intensity_j, size_j)
                )
                intensity_b, size_b = (
                    (intensity_j, size_j) if size_i < size_j else (intensity_i, size_i)
                )

                if (
                    size_a < self.small_size_threshold
                    and size_b > self.large_size_threshold
                ):
                    if self.distance_metric == "cosine":
                        emb_distance = 1 - np.dot(intensity_a, intensity_b) / max(
                            np.linalg.norm(intensity_a) * np.linalg.norm(intensity_b),
                            self.eps,
                        )
                    elif self.distance_metric == "euclidean":
                        emb_distance = np.linalg.norm(intensity_a - intensity_b)
                    elif self.distance_metric == "max":
                        emb_distance = np.max(np.abs(intensity_a - intensity_b))
                    else:
                        raise ValueError("Invalid distance metric")
                    if emb_distance < self.emb_distance_threshold:
                        edges.append((frag_i, frag_j, emb_distance))

        np.savez(
            tmpdir / f"{block.block_id[1]}",
            pos=edges,
            neg=list(
                combinations(unique_labels[label_sizes > self.large_size_threshold], 2)
            ),
            small_frags=[frag for frag in small_frags if frag in write_frags],
        )

    @contextmanager
    def process_block_func(self):
        frags = self.frags_data.array("r")
        raw = self.raw_data.array("r")

        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tmpdir.mkdir(exist_ok=True)

            def process_block(block):
                self.agglomerate_in_block(
                    block,
                    frags,
                    raw,
                    tmpdir,
                )

            if self.lut.exists():
                pos_edges = [
                    (a, b) for a, b in np.load(self.lut)["fragment_segment_lut"]
                ]
            else:
                pos_edges = []

            try:
                yield process_block
            finally:
                neg_edges = []
                small_frags = []
                for file in tmpdir.iterdir():
                    edges = np.load(file)
                    pos_edges.extend(edges["pos"])
                    neg_edges.extend(edges["neg"])
                    small_frags.extend(edges["small_frags"])

                pos_edges = [
                    (True, int(edge[0]), int(edge[1]))
                    for edge in sorted(pos_edges, key=lambda x: x[2])
                ]
                neg_edges = [(False, int(edge[0]), int(edge[1])) for edge in neg_edges]

                lut = mws.cluster(neg_edges + pos_edges)
                lut.extend([(small_frag, 0) for small_frag in small_frags])
                np.savez(
                    self.lut, fragment_segment_lut=np.array(lut, dtype=np.uint64).T
                )
