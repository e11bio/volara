import logging
from contextlib import contextmanager, nullcontext
from typing import Annotated, Iterator, Literal

import daisy
import mwatershed as mws
import networkx as nx
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from pydantic import Field, model_validator
from scipy.ndimage import gaussian_filter, label, maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label as relabel
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from ..datasets import Affs, Labels, Raw
from ..dbs import PostgreSQL, SQLite
from ..segment_utils import replace_values
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

logger = logging.getLogger(__file__)

# Written onto seam affinities to force a split. mws reads only the sign, so any
# large negative does; not `-inf`, since `inf * bool` is nan, which mws rejects.
HARD_SPLIT = -1e9


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
    min_seed_distance: int | None = None
    """
    Seed with one spacing everywhere: the size of the maximum filter footprint
    computed on the boundary distances, in voxels. Mutually exclusive with
    `adaptive_seed_spacing`; set neither to skip seeding entirely.
    """
    adaptive_seed_spacing: float | None = None
    """
    Seed with a spacing that follows the local object size: each candidate's
    suppression radius is this multiple of its own distance to the boundary,
    capped at the block context. Candidates sit on the medial axis, where that
    distance is the local radius, so `2.0` is roughly one seed per local
    diameter. Lower is finer everywhere. Mutually exclusive with
    `min_seed_distance`.
    """
    boundary_mask_offsets: Literal["direct", "all"] | list[int] = "direct"
    """
    Which affinity offsets are averaged into the boundary mask that seeds are
    placed in: `"direct"` for the unit offsets, `"all"` for the whole
    neighborhood, or explicit indices into it. Including long range offsets
    erodes each object by roughly the offset size, which drops objects thinner
    than those offsets out of the mask entirely, leaving them unseeded.
    """
    seed_eps: float | None = None
    """
    If using seeds, this will decay the affs based on distance from the seeds.
    The seed_eps is the scale to apply to the seed distance transform which is
    then subtracted from the shift. This is useful if increased fragmentation of
    the supervoxels is desired.
    """
    max_seed_decay_distance: int | None = None
    """
    If using seeds and `seed_eps` is set, optionally cap the seed distance
    transform used for affinity decay. This prevents the decay from growing
    without bound deep inside large objects, which can otherwise lead to
    unlabeled (`0`) interiors. If None, no cap is applied.
    """
    voronoi: Literal["geodesic", "euclidean"] | None = None
    """
    If using seeds, split every unit-offset affinity whose two voxels belong to
    different seeds, so fragment boundaries land on the surface where two seeds'
    territories meet rather than wherever the affinity field happens to dip.

    `"geodesic"` grows territory by flooding the boundary distances within the
    mask, so a seed claims only tissue it connects to and the seam is a cross
    section, settling at a constriction where there is one. `"euclidean"` takes
    the nearest seed in space, which is cheaper but lets a seed claim tissue it
    is not connected to. Long range offsets are never cut, each spanning many
    territories, so one can still merge across a seam.
    """

    bulk_write: bool = False
    """
    Whether to bulk-write to database (false by default). This removes/rebuilds
    indexes, and sets other useful flags for writing large amounts of data
    quickly, which can be useful for large runs to prevent database bottlenecks.
    """

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint64)

    @model_validator(mode="after")
    def _one_seed_spacing_rule(self) -> "ExtractFrags":
        if (
            self.min_seed_distance is not None
            and self.adaptive_seed_spacing is not None
        ):
            raise ValueError(
                "`min_seed_distance` and `adaptive_seed_spacing` are two ways to set "
                "the same thing - how far apart seeds go - so set one or the other. "
                "Whichever is more restrictive at a voxel would simply win."
            )
        return self

    @model_validator(mode="after")
    def _boundary_mask_offsets_are_usable(self) -> "ExtractFrags":
        channels = self.boundary_mask_channels
        n = len(self.neighborhood)
        if not channels or min(channels) < 0 or max(channels) >= n:
            raise ValueError(
                f"`boundary_mask_offsets={self.boundary_mask_offsets!r}` must select "
                f"at least one offset from the {n}-offset neighborhood."
            )
        return self

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
            total_roi = total_roi.intersect(self.roi)
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
        self.frags_data.drop()
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
        # Average over the direct-neighbour offsets only. Those are the first
        # `ndim` entries of the neighborhood (one per axis), so take the slice
        # from the neighborhood's dimensionality rather than hardcoding 3 -
        # otherwise a 2D neighborhood averages a long-range channel in here.
        ndim = len(self.neighborhood[0])
        average_affs = np.mean(affs[0:ndim], axis=0)

        # Per-fragment mean affinity via unique + bincount, rather than
        # scipy.ndimage.mean with an index array, which rescans the whole volume
        # per label. Same values, same order (np.unique is sorted), one pass.
        fragment_ids, inverse = np.unique(fragments_data, return_inverse=True)

        # on numpy >= 2 `inverse` comes back shaped like
        # the input, and bincount rejects anything but 1-D. On 1.x it was already
        # flat and this is a no-op, so pinned to >=2 in toml.
        inverse = inverse.reshape(-1)
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=average_affs.reshape(-1))
        means = sums / counts
        filtered_fragments = fragment_ids[means < filter_value].astype(
            fragments_data.dtype
        )

        if filtered_fragments.size == 0:
            # Nothing to drop -skip the full-volume relabel pass, which would
            # otherwise be an identity copy of the whole block.
            return fragments_data

        replace = np.zeros_like(filtered_fragments)

        # `replace_values` builds and returns a new array; it does not mutate
        # `fragments_data`. Discarding this return made `filter_fragments` a
        # silent no-op.
        return replace_values(fragments_data, filtered_fragments, replace)

    def get_fragments(
        self,
        affs_data,
        voxel_size: Coordinate | None = None,
    ):
        fragments_data = self.compute_fragments(affs_data, voxel_size=voxel_size)

        # TODO: also mask out the fragments themselves when a mask is provided.
        # `process_block` currently applies the mask to the affinities before
        # fragments are computed, so masked-out regions are only indirectly
        # suppressed. Zeroing the fragments here (and skipping fully-masked
        # blocks outright) is part of the generalized masking support tracked in
        # https://github.com/e11bio/volara/issues/9

        # filter fragments
        if self.filter_fragments > 0:
            fragments_data = self.filter_avg_fragments(
                affs_data, fragments_data, self.filter_fragments
            )

        # remove small debris
        if self.remove_debris > 0:
            fragments_dtype = fragments_data.dtype
            fragments_data = fragments_data.astype(np.int64)
            fragments_data = remove_small_objects(
                fragments_data, min_size=self.remove_debris
            )
            fragments_data = fragments_data.astype(fragments_dtype)

        return fragments_data

    def fragment_centers(
        self,
        fragments_data: np.ndarray,
        offset: Coordinate,
        voxel_size: Coordinate,
    ) -> dict[int, dict]:
        """
        The world-space centroid and voxel count of every non-zero fragment.

        A single np.unique plus one bincount per axis, rather than
        `np.unique(return_counts=True)` alongside `scipy.ndimage.center_of_mass`
        with an index array, which rescans the whole volume per label.
        """
        fragment_ids, inverse = np.unique(fragments_data, return_inverse=True)

        # np >= 2 shapes `inverse` like the input and bincount only takes 1-D.
        inverse = inverse.reshape(-1)
        counts = np.bincount(inverse)

        centers = np.empty((fragment_ids.size, fragments_data.ndim), dtype=np.float64)
        for d in range(fragments_data.ndim):
            # The coordinate ramp along axis `d`, broadcast rather than
            # materialized - bincount reads it as a flat view.
            bcast_shape = [1] * fragments_data.ndim
            bcast_shape[d] = fragments_data.shape[d]
            coord = np.broadcast_to(
                np.arange(fragments_data.shape[d]).reshape(bcast_shape),
                fragments_data.shape,
            )
            centers[:, d] = np.bincount(inverse, weights=coord.reshape(-1)) / counts

        return {
            int(fragment_id): {
                "center": offset + voxel_size * Coordinate(center),
                "size": int(count),
            }
            for fragment_id, center, count in zip(fragment_ids, centers, counts)
            if fragment_id > 0
        }

    def get_seeds(
        self,
        boundary_distances,
        min_seed_distance=10,
    ):
        max_filtered = maximum_filter(boundary_distances, min_seed_distance)
        maxima = max_filtered == boundary_distances

        seeds, n = label(maxima)

        if n == 0:
            return np.zeros(boundary_distances.shape, dtype=np.uint64)

        return seeds

    @property
    def boundary_mask_channels(self) -> list[int]:
        """
        `boundary_mask_offsets` resolved to channel indices. The neighborhood is
        assumed to list the direct neighbors first, then the long range offsets.
        """
        spec = self.boundary_mask_offsets
        if spec == "all":
            return list(range(len(self.neighborhood)))
        if spec == "direct":
            return list(range(len(self.neighborhood[0])))
        return list(spec)

    def _seed_candidates(self, boundary_distances, min_spacing, voxel_size):
        """
        Medial-axis voxels, thinned to one per `min_spacing` cell, deepest first.

        Candidates are voxels rather than connected components: along a smooth
        process the boundary distance is flat down the whole medial axis, so one
        representative per component would put a single seed on a process of any
        length, and the spacing would never be applied.

        Candidates within one voxel of the boundary are rejected outright: their
        distance carries no thickness information, so spacing them out instead
        packs seeds down a sub-resolution wisp, which `voronoi` then shatters
        into fragments small enough for `remove_debris` to delete.
        """
        radius = np.maximum(1, np.rint(min_spacing / np.array(voxel_size))).astype(int)
        grid = np.ogrid[tuple(slice(-r, r + 1) for r in radius)]
        footprint = sum((g / r) ** 2 for g, r in zip(grid, radius)) <= 1.0
        maxima = (
            maximum_filter(boundary_distances, footprint=footprint)
            == boundary_distances
        ) & (boundary_distances > max(voxel_size))

        coords = np.array(np.nonzero(maxima))
        if coords.shape[1] == 0:
            return coords, np.zeros(0)

        distances = boundary_distances[tuple(coords)]
        order = np.argsort(-distances)
        coords, distances = coords[:, order], distances[order]

        # nothing closer together than min_spacing can survive the greedy pass, so
        # keeping one candidate per cell only bounds the work
        cell = np.maximum(1, np.floor(min_spacing / np.array(voxel_size))).astype(int)
        cell_idx = coords.T // cell
        keys = np.ravel_multi_index(cell_idx.T, cell_idx.max(axis=0) + 1)
        _, first = np.unique(keys, return_index=True)
        first.sort()
        return coords[:, first], distances[first]

    def get_adaptive_seeds(self, boundary_distances, voxel_size) -> np.ndarray:
        """
        Seeds whose spacing follows the local object size. `boundary_distances`
        must be in world units, so the spacing is physical under anisotropy.

        Suppression is greedy, deepest candidate first, dropping a candidate if an
        accepted seed lies within *the candidate's own* spacing. Using the accepted
        seed's larger spacing instead would let a soma swallow the seeds of every
        thin process passing nearby.
        """
        # a noise guard rather than a knob; thin objects are spaced by their radius
        floor = 2.0 * max(voxel_size)
        # past the context, seed placement would depend on where blocks fell
        cap = max(float(min(self.context * voxel_size)), floor)

        coords, distances = self._seed_candidates(boundary_distances, floor, voxel_size)
        seeds = np.zeros(boundary_distances.shape, dtype=np.uint64)
        if coords.shape[1] == 0:
            return seeds

        positions = coords.T * np.array(voxel_size)
        spacings_sq = np.clip(self.adaptive_seed_spacing * distances, floor, cap) ** 2

        nearest = np.full(coords.shape[1], np.inf)
        accepted = []
        for i in range(coords.shape[1]):  # already sorted by descending distance
            if nearest[i] < spacings_sq[i]:
                continue
            accepted.append(i)
            offsets = positions - positions[i]
            np.minimum(nearest, np.einsum("ij,ij->i", offsets, offsets), out=nearest)

        logger.debug(
            "adaptive seeding kept %d of %d candidates", len(accepted), coords.shape[1]
        )
        seeds[tuple(coords[:, accepted])] = np.arange(
            1, len(accepted) + 1, dtype=np.uint64
        )
        return seeds

    def seed_cells(
        self,
        seeds: np.ndarray,
        boundary_distances: np.ndarray,
        boundary_mask,
        voxel_size=None,
    ) -> np.ndarray:
        """
        Every masked voxel labelled with the seed that owns it.

        `"geodesic"` floods `-boundary_distances` from the seeds within the mask,
        so territory spreads along tissue and the surface where two seeds meet
        settles at a constriction, the flood reaching a neck last. The partition
        is then a function of the boundary mask alone. `"euclidean"` takes the
        nearest seed in space, for which the distance transform's indices are
        already the answer.
        """
        if self.voronoi == "euclidean":
            indices = distance_transform_edt(
                seeds == 0,
                sampling=tuple(voxel_size),
                return_distances=False,
                return_indices=True,
            )
            return seeds[tuple(indices)]

        return watershed(
            -boundary_distances,
            markers=seeds.astype(np.int32),
            mask=boundary_mask,
        )

    def cut_seams(
        self, shift, seeds: np.ndarray, boundary_distances, boundary_mask, voxel_size
    ):
        """
        Split every unit-offset affinity that crosses a boundary between two seed
        territories.

        Any path between two territories crosses such an edge, so the set is a
        complete cut and the seam is watertight. Edges leaving the mask are
        skipped, already being low affinity, as are long range offsets.
        """
        cells = self.seed_cells(seeds, boundary_distances, boundary_mask, voxel_size)

        for channel, offset in enumerate(self.neighborhood):
            if max(abs(d) for d in offset) > 1:
                continue
            # the voxel pairs this offset connects, both ends in bounds
            src = tuple(
                slice(max(0, -d), size - max(0, d))
                for d, size in zip(offset, cells.shape)
            )
            dst = tuple(
                slice(max(0, d), size - max(0, -d))
                for d, size in zip(offset, cells.shape)
            )
            crosses = (
                (cells[src] != cells[dst]) & boundary_mask[src] & boundary_mask[dst]
            )
            shift[(channel, *src)][crosses] = HARD_SPLIT

        return shift

    def boundary_mask(self, affs_data: np.ndarray) -> np.ndarray:
        """
        The object interiors that seeds may be placed in. See
        `boundary_mask_offsets` for which offsets are averaged.
        """
        return np.mean(affs_data[self.boundary_mask_channels], axis=0) > 0.5

    def compute_fragments(
        self,
        affs_data,
        rng: np.random.Generator | None = None,
        voxel_size: Coordinate | None = None,
    ):
        """
        Mutex watershed on `affs_data`, returning the fragment labels.

        `rng` supplies the `noise_eps` noise; pass a seeded generator to make a
        call reproducible. Note that `randomized_strides=True` draws from its own
        generator inside `mws.agglom` which this does not reach, so seeding here
        only pins the noise.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.sigma is not None:
            # add 0 for channel dim
            sigma = (0, *self.sigma)
        else:
            sigma = None

        # add some random noise to affs (this is particularly necessary if your
        # affs are stored as uint8 or similar). If you have many affinities of
        # the exact same value the order they are processed in may be fifo, so
        # you can get annoying streaks.

        if self.noise_eps is not None:
            # Generate the noise straight into `shift` rather than
            # `zeros_like` + `randn(*shape) * eps` + `+=`, which allocates three
            # full (C, Z, Y, X) float64 volumes (~2.2 GB each at full context)
            # to end up with one.
            shift = np.empty_like(affs_data)
            if shift.dtype in (np.float32, np.float64):
                rng.standard_normal(shift.shape, dtype=shift.dtype, out=shift)
            else:
                shift[:] = rng.standard_normal(shift.shape)
            shift *= self.noise_eps
        else:
            shift = np.zeros_like(affs_data)

        #######################

        # add smoothed affs, to solve a similar issue to the random noise. We
        # want to bias towards processing the central regions of objects first.

        if sigma is not None:
            shift += gaussian_filter(affs_data, sigma=sigma) - affs_data

        #######################
        shift += np.array([self.bias]).reshape(
            (-1, *((1,) * (len(affs_data.shape) - 1)))
        )

        if self.min_seed_distance is not None or self.adaptive_seed_spacing is not None:
            boundary_mask = self.boundary_mask(affs_data)
            if voxel_size is None:
                voxel_size = self.voxel_size

            # in world units, so spacings are physical under anisotropy
            boundary_distances = distance_transform_edt(
                boundary_mask, sampling=tuple(voxel_size)
            )

            if self.adaptive_seed_spacing is not None:
                seeds = self.get_adaptive_seeds(boundary_distances, voxel_size)
            else:
                seeds = self.get_seeds(
                    distance_transform_edt(boundary_mask),
                    min_seed_distance=self.min_seed_distance,
                ).astype(np.uint64)

            seeds[~boundary_mask] = 0

            if self.seed_eps is not None:
                D = distance_transform_edt(seeds == 0)

                if self.max_seed_decay_distance is not None:
                    D = np.minimum(D, self.max_seed_decay_distance)

                shift -= self.seed_eps * D

            if self.voronoi is not None:
                shift = self.cut_seams(
                    shift, seeds, boundary_distances, boundary_mask, voxel_size
                )

        else:
            seeds = None

        # `shift` is our own temporary, so fold the affinities into it in place
        # rather than allocating `affs_data + shift` and copying that again via
        # `.astype(np.float64)` (which copies even when already float64).
        # `affs_data` itself must survive - filter_avg_fragments still reads it.
        shift += affs_data

        fragments_data = mws.agglom(
            shift.astype(np.float64, copy=False),
            offsets=self.neighborhood,
            strides=self.strides,
            seeds=seeds,
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
        benchmark_logger = self.get_benchmark_logger()

        with benchmark_logger.trace("Read Affs"):
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
            with benchmark_logger.trace("Read Mask"):
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

        with benchmark_logger.trace("Compute Fragments"):
            fragments_data = self.get_fragments(affs_data, voxel_size=affs.voxel_size)

            fragments = Array(
                fragments_data,
                offset=block.read_roi.offset,
                voxel_size=frags.voxel_size,
            )

        with benchmark_logger.trace("Relabel Fragments"):
            fragments_data = fragments.to_ndarray(block.write_roi)
            max_id = fragments_data.max()

            fragments_data, max_id = relabel(fragments_data, return_num=True)
            assert max_id < self.num_voxels_in_block, f"max_id: {max_id}"

            # ensure unique IDs
            id_bump = block.block_id[1] * self.num_voxels_in_block

            fragments_data[fragments_data > 0] += id_bump

        with benchmark_logger.trace("Write Fragments"):
            frags[block.write_roi] = fragments_data

        # following only makes a difference if fragments were found
        if fragments_data.max() == 0:
            return

        with benchmark_logger.trace("Compute Fragment Centers"):
            fragment_centers = self.fragment_centers(
                fragments_data,
                block.write_roi.get_offset(),
                affs.voxel_size,
            )
            logger.info("Found %d fragments", len(fragment_centers))

        with benchmark_logger.trace("Update RAG"):
            if self.bulk_write:
                # since we drop indexes, the read_graph query has to check each
                # node for containment which grows with cost as the db grows
                rag = nx.Graph()
            else:
                rag = rag_provider[block.write_roi]
                assert len(rag) == 0, "RAG should be empty"

            for node, data in fragment_centers.items():
                # centers
                node_attrs = {
                    "position": data["center"],
                }

                node_attrs["size"] = int(data["size"])

                rag.add_node(int(node), **node_attrs)

            if self.bulk_write:
                rag_provider.bulk_write_graph(
                    rag,
                    block.write_roi,
                )
            else:
                rag_provider.write_graph(
                    rag,
                    block.write_roi,
                )

    def _task_context(self, worker):
        if self.bulk_write:
            return self.db.open("r+").bulk_write_mode(
                worker=worker, node_writes=True, edge_writes=False
            )
        return nullcontext()

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

        with self._task_context(worker=True):
            yield process_block

    @contextmanager
    def task(
        self,
        upstream_tasks: daisy.Task | list[daisy.Task] | None = None,
        multiprocessing: bool = True,
    ) -> Iterator[daisy.Task]:

        # temporary workaround since bulk_write_mode needs to modify the db
        self.init()

        with self._task_context(worker=False):
            with super().task(upstream_tasks, multiprocessing) as task:
                yield task
