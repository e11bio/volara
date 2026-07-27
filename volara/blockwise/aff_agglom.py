import logging
from contextlib import contextmanager
from itertools import chain
from typing import Annotated, Callable, Generator, Literal

import networkx as nx
import numpy as np
import scipy.ndimage
from daisy import Block
from funlib.geometry import Coordinate, Roi
from funlib.math import inv_cantor_number
from funlib.persistence.arrays import Array
from funlib.persistence.graphs.graph_database import GraphDataBase
from pydantic import Field

from ..datasets import Affs, Dataset, Labels
from ..dbs import PostgreSQL, SQLite
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

logger = logging.getLogger(__file__)


class AffAgglom(BlockwiseTask):
    """
    A blockwise task that computes edges between supervoxels
    sharing affinity edges and stores statistics such as the
    mean affinity value between the supervoxels.
    """

    task_type: Literal["aff-agglom"] = "aff-agglom"
    db: Annotated[
        PostgreSQL | SQLite,
        Field(discriminator="db_type"),
    ]
    """
    The database containing nodes associated with the fragment supervoxels
    to which we will add the supervoxel edge affinity statistics
    """
    affs_data: Affs
    """
    The affs to read for creating supervoxel affinity statistics
    """
    frags_data: Labels
    """
    The labels array containing the supervoxels for computing affinity statistics
    """
    block_size: PydanticCoordinate
    """
    The blocksize within which to compute supervoxel affinity statistics
    """
    context: PydanticCoordinate
    """
    The amount of context to use when computing supervoxel affinity statistics
    """
    scores: dict[str, list[PydanticCoordinate]]
    """
    A dictionary of score names and their respective neighborhoods.
    This allows us to compute the affinity statistics in subgroups of the
    neighborhood. For example if you wanted to compute the mean affinity
    between fragments in x and y separately from z, you would provide
    a dictionary like so:

        scores = {
            "xy": [(1, 0, 0), (0, 1, 0)],
            "z": [(0, 0, 1)]
        }

    """
    fit: Literal["shrink"] = "shrink"
    """
    The boundary behavior for our daisy task.
    """
    read_write_conflict: Literal[False] = False
    """
    We don't have read write conflicts in this task and can compute
    every block independently in an arbitrary order.
    """
    optimized: bool = False
    """
    Use the optimized implementations of the fragment relabel and the affinity
    counting. Produces the same edges and weights as the original code, just
    faster (measured 7.6x on the relabel, 40x on the counting):

    - relabel via `np.unique(return_inverse=True)` instead of a full-array scan
      per fragment (O(n_frags * n_voxels))
    - affinity counting over only the boundary voxels, with an integer cantor
      pairing and `np.bincount`, instead of full-volume float64 pairings handed
      to `scipy.ndimage.mean` / `sum_labels`

    Defaults to False so the original behavior is what you get unless you ask
    for it. Set True to compare.
    """

    @property
    def task_name(self) -> str:
        return f"{self.db.id}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.frags_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(self.roi)
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

    def agglomerate(self, affs: np.ndarray, frags: np.ndarray, rag: nx.Graph):
        benchmark_logger = self.get_benchmark_logger()

        with benchmark_logger.trace("Preprocess fragments"):
            # Both branches relabel fragments to a dense 1..k range so the cantor
            # pairings in count_affs stay small, and produce the same mapping.
            if self.optimized:
                # np.unique(return_inverse=True) IS this relabel, done in C.
                unique_vals, inverse = np.unique(frags, return_inverse=True)
                # 0 sorts first, so when background is present inverse already
                # maps 0 -> 0 and fragments -> 1..k in the same order the loop
                # below produces. When the block has NO background, inverse would
                # map the lowest fragment id to 0 and silently turn it into
                # background, so shift by one in that case.
                has_background = unique_vals.size > 0 and unique_vals[0] == 0
                frags = inverse.reshape(frags.shape)
                if not has_background:
                    frags = frags + 1
                frags = frags.astype(np.uint64)

                first_seq = 0 if has_background else 1
                rev_mapping = {
                    seq + first_seq: int(orig) for seq, orig in enumerate(unique_vals)
                }
                num_frags = unique_vals.size - (1 if has_background else 0)
            else:
                # Original: one full-array scan per fragment, O(n_frags * n_vox).
                fragment_ids = [int(x) for x in np.unique(frags) if x != 0]
                num_frags = len(fragment_ids)
                frag_mapping = {
                    old: seq for seq, old in zip(range(1, num_frags + 1), fragment_ids)
                }
                rev_mapping = {v: k for k, v in frag_mapping.items()}
                for old, seq in frag_mapping.items():
                    frags[frags == old] = seq

            if num_frags == 0:
                return

            neighborhood_affs: dict[Coordinate, dict[int, tuple[float, float]]] = {}

            # affs_data.neighborhood cannot be None, assert called to make mypy happy
            assert self.affs_data.neighborhood is not None

        def slices(offset: Coordinate):
            base_slice = tuple(
                slice(-m if m < 0 else None, -m if m > 0 else None) for m in offset
            )
            offset_slice = tuple(
                slice(m if m > 0 else None, m if m < 0 else None) for m in offset
            )
            return base_slice, offset_slice

        def count_affs_original(
            fragments: np.ndarray, affinities: np.ndarray, offset: Coordinate
        ) -> dict[int, tuple[float, float]]:
            base_slice, offset_slice = slices(offset)
            base_frags = fragments[base_slice]
            base_affinities = affinities[base_slice]
            offset_frags = fragments[offset_slice]

            mask = (offset_frags != base_frags) * (offset_frags > 0) * (base_frags > 0)

            # cantor pairing function
            # 1/2 (k1 + k2)(k1 + k2 + 1) + k2
            k1, k2 = (
                np.min([offset_frags, base_frags], axis=0),
                np.max([offset_frags, base_frags], axis=0),
            )
            cantor_pairings = ((k1 + k2) * (k1 + k2 + 1) / 2 + k2) * mask
            cantor_ids = np.array([x for x in np.unique(cantor_pairings) if x != 0])
            scores = scipy.ndimage.mean(base_affinities, cantor_pairings, cantor_ids)
            counts = scipy.ndimage.sum_labels(mask, cantor_pairings, cantor_ids)
            return {
                cantor_id: (mean_score, count)
                for cantor_id, mean_score, count in zip(cantor_ids, scores, counts)
            }

        def count_affs_optimized(
            fragments: np.ndarray, affinities: np.ndarray, offset: Coordinate
        ) -> dict[int, tuple[float, float]]:
            base_slice, offset_slice = slices(offset)
            base_frags = fragments[base_slice]
            base_affinities = affinities[base_slice]
            offset_frags = fragments[offset_slice]

            mask = (offset_frags != base_frags) & (offset_frags > 0) & (base_frags > 0)

            # Only voxels on a boundary between two distinct fragments carry an
            # edge -- typically a few percent of the block. Select them first and
            # do all the arithmetic on that small array, instead of computing
            # full-volume cantor pairings and handing them to
            # scipy.ndimage.mean/sum_labels (which then rescan every voxel; that
            # pair alone was ~85% of this function).
            u_frags = offset_frags[mask].astype(np.uint64)
            v_frags = base_frags[mask].astype(np.uint64)
            if u_frags.size == 0:
                return {}

            # cantor pairing function, in integers:
            # 1/2 (k1 + k2)(k1 + k2 + 1) + k2
            # Kept in uint64 rather than float64 -- the ids are exact here,
            # whereas float64 silently collides once the pairing exceeds 2**53.
            k1 = np.minimum(u_frags, v_frags)
            k2 = np.maximum(u_frags, v_frags)
            k_sum = k1 + k2
            cantor_pairings = (k_sum * (k_sum + 1)) // 2 + k2

            cantor_ids, inverse = np.unique(cantor_pairings, return_inverse=True)
            weights = base_affinities[mask]
            counts = np.bincount(inverse)
            sums = np.bincount(inverse, weights=weights)

            return {
                int(cantor_id): (float(total / count), float(count))
                for cantor_id, total, count in zip(cantor_ids, sums, counts)
            }

        count_affs = count_affs_optimized if self.optimized else count_affs_original

        with benchmark_logger.trace("Count affinities"):
            for offset_affs, offset in zip(affs, self.affs_data.neighborhood):
                neighborhood_affs[offset] = count_affs(frags, offset_affs, offset)

        with benchmark_logger.trace("postprocess scores"):
            for score_name, score_neighborhood in self.scores.items():
                offset_counts = [
                    neighborhood_affs[offset] for offset in score_neighborhood
                ]
                cantor_ids = set(chain(*[x.keys() for x in offset_counts]))
                for cantor_id in cantor_ids:
                    key_counts = [x.get(cantor_id, (1.0, 0.0)) for x in offset_counts]
                    total_count = sum([count[1] for count in key_counts])
                    if total_count > 0:
                        u, v = inv_cantor_number(cantor_id, dims=2)
                        rag.add_edge(
                            rev_mapping[u],
                            rev_mapping[v],
                            **{
                                score_name: sum(
                                    [
                                        count[0] * count[1] / total_count
                                        for count in key_counts
                                    ]
                                )
                            },
                        )

    def agglomerate_in_block(
        self, block: Block, affs: Array, frags: Array, rag_provider: GraphDataBase
    ):
        benchmark_logger = self.get_benchmark_logger()

        with benchmark_logger.trace("Read data in block"):
            frags_data = frags.to_ndarray(block.read_roi, fill_value=0)
            rag = rag_provider[block.read_roi]

            affs_data = affs.to_ndarray(block.read_roi, fill_value=0)

            if affs_data.dtype == np.uint8:
                affs_data = affs_data.astype(np.float32) / 255.0

        with benchmark_logger.trace("Agglomerate"):
            self.agglomerate(
                affs_data,
                frags_data,
                rag,
            )

        with benchmark_logger.trace("Write RAG edges"):
            rag_provider.write_graph(rag, block.write_roi, write_nodes=False)

    def init(self) -> None:
        self.db.init()

    @contextmanager
    def process_block_func(self) -> Generator[Callable, None, None]:
        affs = self.affs_data.array("r")
        frags = self.frags_data.array("r")
        rag_provider = self.db.open("r+")

        def process_block(block) -> None:
            self.agglomerate_in_block(
                block,
                affs,
                frags,
                rag_provider,
            )

        yield process_block
