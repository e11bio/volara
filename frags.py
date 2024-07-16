import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional

import mwatershed as mws
import numpy as np
from funlib.persistence import Array
from funlib.segment.arrays import replace_values
from scipy.ndimage import binary_dilation, center_of_mass, label, measurements
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label as relabel
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

# from volara.utils.agglomeration import waterz_agglomerate
# from volara.utils.watershed import (
#     filter_avg_fragments,
# )
from .volara.utils import PydanticCoordinate, StrictBaseModel

logging.basic(level=logging.INFO)


def watershed_from_boundary_distance(
    boundary_distances,
    boundary_mask,
    return_seeds=False,
    id_offset=0,
    min_seed_distance=10,
):
    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances

    seeds, n = label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=boundary_mask
    )

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret


def watershed_from_affinities(
    affs,
    max_affinity_value=1.0,
    fragments_in_xy=False,
    return_seeds=False,
    min_seed_distance=10,
):
    """Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.
    Returns:
        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True"""

    if fragments_in_xy:
        # mean_affs = 0.5 * (affs[1] + affs[2])

        mean_affs = (1 / 3) * (
            affs[0] + affs[1] + affs[2]
        )  # todo: other affinities? *0.5

        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):
            boundary_mask = mean_affs[z] > 0.5 * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            ret = watershed_from_boundary_distance(
                boundary_distances,
                boundary_mask,
                return_seeds=return_seeds,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance,
            )

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:
        boundary_mask = np.mean(affs, axis=0) > 0.5 * max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        ret = watershed_from_boundary_distance(
            boundary_distances,
            boundary_mask,
            return_seeds,
            min_seed_distance=min_seed_distance,
        )

    return ret


def mwatershed_from_affinities(
    affs,
    neighborhood,
    sigma,
    adjacent_edge_bias,
    lr_edge_bias,
    fragments_in_xy=False,
    strides=None,
):
    # todo: handle this automatically for different neighborhoods
    if fragments_in_xy:
        neighborhood = np.array(neighborhood)

        affs.data = affs.data[[0, 1, 3, 4, 6, 7]]
        neighborhood = list(neighborhood[[0, 1, 3, 4, 6, 7]])

    # add some random noise to affs (this is particularly necessary if your affs are
    #  stored as uint8 or similar)
    # If you have many affinities of the exact same value the order they are processed
    # in may be fifo, so you can get annoying streaks.

    ### tmp comment out ###

    random_noise = np.random.randn(*affs.shape) * 0.001  # todo: parameterize?

    #######################

    # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # towards processing the central regions of objects first.

    ### tmp comment out ###

    smoothed_affs = (
        gaussian_filter(affs, sigma=sigma) - 0.5
    ) * 0.01  # todo: parameterize?

    #######################

    shift = np.array(
        [
            adjacent_edge_bias if max(offset) <= 1 else lr_edge_bias
            for offset in neighborhood
        ]
    ).reshape((-1, *((1,) * (len(affs.data.shape) - 1))))

    fragments_data = mws.agglom(
        affs + shift + random_noise + smoothed_affs,
        offsets=neighborhood,
        strides=strides,
    )

    return fragments_data


def filter_avg_fragments(affs, fragments_data, filter_value):
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


def upsample(a, factor):
    for d, f in enumerate(factor):
        a = np.repeat(a, f, axis=d)

    return a


def get_mask_data_in_roi(mask, roi, target_voxel_size):
    assert mask.voxel_size.is_multiple_of(target_voxel_size), (
        "Can not upsample from %s to %s" % (mask.voxel_size, target_voxel_size)
    )

    aligned_roi = roi.snap_to_grid(mask.voxel_size, mode="grow")
    aligned_data = mask.to_ndarray(aligned_roi, fill_value=0)

    if mask.voxel_size == target_voxel_size:
        return aligned_data

    factor = mask.voxel_size / target_voxel_size

    upsampled_aligned_data = upsample(aligned_data, factor)

    upsampled_aligned_mask = Array(
        upsampled_aligned_data, roi=aligned_roi, voxel_size=target_voxel_size
    )

    return upsampled_aligned_mask.to_ndarray(roi)


def compute_mean_intensities(
    image_data, requested_channels, fragments_data, fragment_id
):
    data_channels = image_data.shape[0]

    assert (
        requested_channels is not None and requested_channels <= data_channels
    ), f"Number of channels to compute mean intensities must be not None and less than or equal to the number of channels in the data. Requested channels: {requested_channels}, data channels: {data_channels}"

    # Calculate mean intensities for each channel
    mean_intensities = [
        measurements.mean(image_data[ch], fragments_data, fragment_id)
        for ch in range(requested_channels)
    ]

    return mean_intensities


def embedding_frag_gen(
    embedding,
    affs,
    mask_threshold,
    distance_threshold,
    max_affinity_value=1.0,
    min_seed_distance=10,
    sampling_rate=1.0,
    merge_segments=True,
    affs_seeds=True,
    relabel_fragments=False,
):
    seed_array = affs if affs_seeds else embedding

    boundary_mask = np.mean(seed_array, axis=0) > mask_threshold * max_affinity_value

    boundary_distances = distance_transform_edt(boundary_mask)

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances

    seeds, n = label(maxima)

    unique_labels = np.unique(seeds * maxima)

    centroids = center_of_mass(maxima, seeds, unique_labels[unique_labels != 0])

    maxima_coords = np.array(
        [
            np.round(centroid).astype(int)
            for centroid in centroids
            if not np.isnan(centroid).any()
        ]
    )

    num_sampled_points = int(len(maxima_coords) * sampling_rate)
    sampled_maxima_coords = maxima_coords[
        np.random.choice(len(maxima_coords), num_sampled_points, replace=False)
    ]

    fragments = np.zeros(embedding.shape[1:], dtype=np.uint64)
    start_label = 1

    for coord in sampled_maxima_coords:
        current_label = fragments[coord[0], coord[1], coord[2]]

        point = embedding[:, coord[0], coord[1], coord[2]]

        # todo add dot product option
        distance = np.sum(
            (embedding - point[:, np.newaxis, np.newaxis, np.newaxis]) ** 2, axis=0
        )

        mask = distance <= distance_threshold**2

        if merge_segments:
            if current_label == 0:
                # This seed is not part of any segment yet, create a new one
                fragments[mask] = start_label
                start_label += 1
            else:
                # This seed is part of an existing segment, merge the new segment
                fragments[mask] = current_label

        else:
            # Check for adjacency with existing segments
            dilated_mask = binary_dilation(mask)
            overlapping_labels = np.unique(fragments[dilated_mask])
            overlapping_labels = overlapping_labels[
                overlapping_labels != 0
            ]  # Exclude background

            if len(overlapping_labels) > 0:
                start_label += (
                    1  # Increment label to avoid merging with adjacent segment
                )

            fragments[mask] = start_label

    if relabel_fragments:
        fragments = relabel(fragments).astype(fragments.dtype)

    return fragments


from e11_post.utils.merge_tree import MergeTree

try:
    import waterz
except ImportError:
    pass
from funlib.math import inv_cantor_number

waterz_merge_functions = {
    "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
    "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
    "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
    "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
    "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
    "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
    "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
    "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
    "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
    "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
    "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
}


def waterz_agglomerate(
    affinities,
    fragments,
    threshold,
    merge_function="hist_quant_75",
    fragment_relabel_map=None,
    rag=None,
):
    if fragment_relabel_map is not None:
        if rag is None:
            raise ValueError(
                "If fragment_relabel_map is provided, rag should also be provided."
            )
        thresholds_list = [0, threshold]
    else:
        thresholds_list = [threshold]

    generator = waterz.agglomerate(
        affs=affinities.astype(np.float32),
        fragments=fragments,
        thresholds=thresholds_list,
        scoring_function=waterz_merge_functions[merge_function],
        discretize_queue=256 if fragment_relabel_map is not None else 0,
        return_merge_history=True if fragment_relabel_map is not None else False,
        return_region_graph=True if fragment_relabel_map is not None else False,
    )

    if fragment_relabel_map is not None:
        # add edges to RAG
        _, _, initial_rag = next(generator)
        for edge in initial_rag:
            u, v = fragment_relabel_map[edge["u"]], fragment_relabel_map[edge["v"]]
            rag.add_edge(u, v, merge_score=None, agglomerated=True)

        # Collect merge history
        _, merge_history, _ = next(generator)

        # Cleanup generator
        for _, _, _ in generator:
            pass

        # create a merge tree from the merge history
        merge_tree = MergeTree(fragment_relabel_map)

        for merge in merge_history:
            a, b, c, score = merge["a"], merge["b"], merge["c"], merge["score"]
            merge_tree.merge(
                fragment_relabel_map[a],
                fragment_relabel_map[b],
                fragment_relabel_map[c],
                score,
            )

        # mark edges in original RAG with score at time of merging
        num_merged = 0
        for u, v, data in rag.edges(data=True):
            merge_score = merge_tree.find_merge(u, v)
            data["adj_weight"] = merge_score
            if merge_score is not None:
                num_merged += 1

    else:
        data = next(generator)
        # Cleanup generator
        for _ in generator:
            pass

    return data if fragment_relabel_map is None else None


def mwatershed_agglomerate(affs, fragments, neighborhood, rag):
    fragment_ids = [int(x) for x in np.unique(fragments) if x != 0]
    num_frags = len(fragment_ids)
    frag_mapping = {old: seq for seq, old in zip(range(1, num_frags + 1), fragment_ids)}
    rev_mapping = {v: k for k, v in frag_mapping.items()}
    for old, seq in frag_mapping.items():
        fragments[fragments == old] = seq

    if len(fragment_ids) == 0:
        return

    # COMPUTE EDGE SCORES
    # mutex watershed has shown good results when using short range edges
    # for merging objects and long range edges for splitting. So we compute
    # these scores separately

    # separate affinities and neighborhood by range
    adjacents = [offset for offset in neighborhood if max(offset) <= 1]
    lr_neighborhood = neighborhood[len(adjacents) :]
    affs, lr_affs = affs[: len(adjacents)], affs[len(adjacents) :]

    # COMPUTE EDGE SCORES FOR ADJACENT FRAGMENTS
    max_offset = [max(axis) for axis in zip(*adjacents)]
    base_frags = fragments[tuple(slice(0, -m) for m in max_offset)]
    base_affs = affs[(slice(None, None),) + tuple(slice(0, -m) for m in max_offset)]
    offset_frags = []
    for offset in adjacents:
        offset_frags.append(
            fragments[
                tuple(
                    slice(o, (-m + o) if m != o else None)
                    for o, m in zip(offset, max_offset)
                )
            ]
        )

    offset_frags = np.stack(offset_frags)
    stacked_base_frags = np.stack([base_frags] * offset_frags.shape[0])
    mask = (
        (offset_frags != stacked_base_frags)
        * (offset_frags > 0)
        * (stacked_base_frags > 0)
    )

    # cantor pairing function
    # 1/2 (k1 + k2)(k1 + k2 + 1) + k2
    k1, k2 = (
        np.min(
            [
                offset_frags,
                stacked_base_frags,
            ],
            axis=0,
        ),
        np.max(
            [
                offset_frags,
                stacked_base_frags,
            ],
            axis=0,
        ),
    )
    cantor_pairings = ((k1 + k2) * (k1 + k2 + 1) / 2 + k2) * mask
    cantor_ids = np.array([x for x in np.unique(cantor_pairings) if x != 0])
    adjacent_score = measurements.median(
        base_affs,
        cantor_pairings,
        cantor_ids,
    )
    adjacent_map = {
        cantor_id: float(med_score)
        for cantor_id, med_score in zip(cantor_ids, adjacent_score)
    }

    # COMPUTE LONG RANGE EDGE SCORES
    lr_max_offset = [max(axis) for axis in zip(*lr_neighborhood)]
    lr_base_frags = fragments[tuple(slice(0, -m) for m in lr_max_offset)]
    lr_base_affs = lr_affs[
        (slice(None, None),) + tuple(slice(0, -m) for m in lr_max_offset)
    ]
    lr_offset_frags = []
    for offset in lr_neighborhood:
        lr_offset_frags.append(
            fragments[
                tuple(
                    slice(o, (-m + o) if m != o else None)
                    for o, m in zip(offset, lr_max_offset)
                )
            ]
        )
    lr_offset_frags = np.stack(lr_offset_frags)
    stacked_lr_base_frags = np.stack([lr_base_frags] * lr_offset_frags.shape[0])
    lr_mask = (
        (lr_offset_frags != stacked_lr_base_frags)
        * (lr_offset_frags > 0)
        * (stacked_lr_base_frags > 0)
    )
    # cantor pairing function
    k1, k2 = (
        np.min(
            [
                lr_offset_frags,
                stacked_lr_base_frags,
            ],
            axis=0,
        ),
        np.max(
            [lr_offset_frags, stacked_lr_base_frags],
            axis=0,
        ),
    )
    lr_cantor_pairings = ((k1 + k2) * (k1 + k2 + 1) / 2 + k2) * lr_mask
    lr_cantor_ids = np.array([x for x in np.unique(lr_cantor_pairings) if x != 0])
    lr_adjacent_score = measurements.median(
        lr_base_affs,
        lr_cantor_pairings,
        lr_cantor_ids,
    )
    lr_adjacent_map = {
        cantor_id: float(med_score)
        for cantor_id, med_score in zip(lr_cantor_ids, lr_adjacent_score)
    }

    cantor_ids = set(adjacent_map.keys()).union(set(lr_adjacent_map.keys()))

    for cantor_id in cantor_ids:
        u, v = inv_cantor_number(cantor_id, dims=2)
        assert u > v and v > 0, (
            u,
            v,
            adjacent_map.get(cantor_id, None),
            lr_adjacent_map.get(cantor_id, None),
        )
        adj_weight = adjacent_map.get(cantor_id, None)
        lr_adj_weight = lr_adjacent_map.get(cantor_id, None)
        rag.add_edge(
            rev_mapping[u],
            rev_mapping[v],
            adj_weight=adj_weight,
            lr_weight=lr_adj_weight,
        )


class Agglom(ABC, StrictBaseModel):
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    fragments_in_xy: bool = False
    filter_fragments: float = 0.0
    remove_debris: int = 0
    epsilon_agglomerate: Optional[float] = None

    @abstractmethod
    def compute_fragments(self, affs_data):
        pass

    @abstractmethod
    def agglomerate(self, affs, frags, rag):
        pass

    def get_fragments(self, affs_data):
        fragments_data = self.compute_fragments(affs_data)

        # # mask fragments if provided
        # if mask is not None:
        #     fragments_data *= mask_data.astype(np.uint64)

        # filter fragments
        if self.filter_fragments > 0:
            filter_avg_fragments(affs_data, fragments_data, self.filter_fragments)

        # remove small debris
        if self.remove_debris > 0:
            fragments_dtype = fragments_data.dtype
            fragments_data = fragments_data.astype(np.int64)
            remove_small_objects(fragments_data, min_size=self.remove_debris)
            fragments_data = fragments_data.astype(fragments_dtype)

        # epsilon agglomeration
        if self.epsilon_agglomerate is not None:
            waterz_agglomerate(
                affinities=affs_data,
                threshold=self.epsilon_agglomerate,
                fragments=fragments_data,
            )

        return fragments_data


class MWatershed(Agglom):
    agglom_type: Literal["mwatershed"] = "mwatershed"
    bias: list[float]
    neighborhood: list[PydanticCoordinate]
    strides: Optional[list[PydanticCoordinate]] = None
    sigma: Optional[PydanticCoordinate] = None
    noise_eps: Optional[float] = None

    def agglomerate(self, affs, frags, rag):
        from funlib.math import inv_cantor_number
        from scipy.ndimage import measurements

        fragment_ids = [int(x) for x in np.unique(frags) if x != 0]
        num_frags = len(fragment_ids)
        frag_mapping = {
            old: seq for seq, old in zip(range(1, num_frags + 1), fragment_ids)
        }
        rev_mapping = {v: k for k, v in frag_mapping.items()}
        for old, seq in frag_mapping.items():
            frags[frags == old] = seq

        if len(fragment_ids) == 0:
            return

        # COMPUTE EDGE SCORES
        # mutex watershed has shown good results when using short range edges
        # for merging objects and long range edges for splitting. So we compute
        # these scores separately

        # separate affinities and neighborhood by range
        adjacents = [offset for offset in self.neighborhood if max(offset) <= 1]
        lr_neighborhood = self.neighborhood[len(adjacents) :]
        affs, lr_affs = affs[: len(adjacents)], affs[len(adjacents) :]

        # COMPUTE EDGE SCORES FOR ADJACENT FRAGMENTS
        if len(adjacents) > 0:
            max_offset = [max(axis) for axis in zip(*adjacents)]
            base_frags = frags[tuple(slice(0, -m) for m in max_offset)]
            base_affs = affs[
                (slice(None, None),) + tuple(slice(0, -m) for m in max_offset)
            ]
            offset_frags = []
            for offset in adjacents:
                offset_frags.append(
                    frags[
                        tuple(
                            slice(o, (-m + o) if m != o else None)
                            for o, m in zip(offset, max_offset)
                        )
                    ]
                )

            offset_frags = np.stack(offset_frags)
            stacked_base_frags = np.stack([base_frags] * offset_frags.shape[0])
            mask = (
                (offset_frags != stacked_base_frags)
                * (offset_frags > 0)
                * (stacked_base_frags > 0)
            )

            # cantor pairing function
            # 1/2 (k1 + k2)(k1 + k2 + 1) + k2
            k1, k2 = (
                np.min(
                    [
                        offset_frags,
                        stacked_base_frags,
                    ],
                    axis=0,
                ),
                np.max(
                    [
                        offset_frags,
                        stacked_base_frags,
                    ],
                    axis=0,
                ),
            )
            cantor_pairings = ((k1 + k2) * (k1 + k2 + 1) / 2 + k2) * mask
            cantor_ids = np.array([x for x in np.unique(cantor_pairings) if x != 0])
            adjacent_score = measurements.median(
                base_affs,
                cantor_pairings,
                cantor_ids,
            )
            adjacent_map = {
                cantor_id: float(med_score)
                for cantor_id, med_score in zip(cantor_ids, adjacent_score)
            }
        else:
            adjacent_map = {}

        # COMPUTE LONG RANGE EDGE SCORES
        if len(lr_neighborhood) > 0:
            lr_max_offset = [max(axis) for axis in zip(*lr_neighborhood)]
            lr_base_frags = frags[tuple(slice(0, -m) for m in lr_max_offset)]
            lr_base_affs = lr_affs[
                (slice(None, None),) + tuple(slice(0, -m) for m in lr_max_offset)
            ]
            lr_offset_frags = []
            for offset in lr_neighborhood:
                lr_offset_frags.append(
                    frags[
                        tuple(
                            slice(o, (-m + o) if m != o else None)
                            for o, m in zip(offset, lr_max_offset)
                        )
                    ]
                )
            lr_offset_frags = np.stack(lr_offset_frags)
            stacked_lr_base_frags = np.stack([lr_base_frags] * lr_offset_frags.shape[0])
            lr_mask = (
                (lr_offset_frags != stacked_lr_base_frags)
                * (lr_offset_frags > 0)
                * (stacked_lr_base_frags > 0)
            )
            # cantor pairing function
            k1, k2 = (
                np.min(
                    [
                        lr_offset_frags,
                        stacked_lr_base_frags,
                    ],
                    axis=0,
                ),
                np.max(
                    [lr_offset_frags, stacked_lr_base_frags],
                    axis=0,
                ),
            )
            lr_cantor_pairings = ((k1 + k2) * (k1 + k2 + 1) / 2 + k2) * lr_mask
            lr_cantor_ids = np.array(
                [x for x in np.unique(lr_cantor_pairings) if x != 0]
            )
            lr_adjacent_score = measurements.median(
                lr_base_affs,
                lr_cantor_pairings,
                lr_cantor_ids,
            )
            lr_adjacent_map = {
                cantor_id: float(med_score)
                for cantor_id, med_score in zip(lr_cantor_ids, lr_adjacent_score)
            }
        else:
            lr_adjacent_map = {}

        cantor_ids = set(adjacent_map.keys()).union(set(lr_adjacent_map.keys()))

        for cantor_id in cantor_ids:
            u, v = inv_cantor_number(cantor_id, dims=2)
            assert u > v and v > 0, (
                u,
                v,
                adjacent_map.get(cantor_id, None),
                lr_adjacent_map.get(cantor_id, None),
            )
            adj_weight = adjacent_map.get(cantor_id, None)
            lr_adj_weight = lr_adjacent_map.get(cantor_id, None)
            rag.add_edge(
                rev_mapping[u],
                rev_mapping[v],
                adj_weight=adj_weight,
                lr_weight=lr_adj_weight,
            )

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


class Waterz(Agglom):
    agglom_type: Literal["waterz"] = "waterz"
    merge_function: str
    epsilon_agglomerate: float
    thresholds_minmax: list[float]
    thresholds_step: float

    def agglomerate(self, affs, frags, rag):
        import waterz
        from funlib.segment.arrays import relabel
        from volara.utils.merge_tree import MergeTree

        waterz_merge_functions = {
            "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
            "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
            "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
            "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
            "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
            "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
            "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
            "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
            "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
            "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
            "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
        }

        fragments_relabelled, n, fragment_relabel_map = relabel(
            frags, return_backwards_map=True
        )

        if fragment_relabel_map is not None:
            if rag is None:
                raise ValueError(
                    "If fragment_relabel_map is provided, rag should also be provided."
                )
            thresholds_list = [0, self.threshold]
        else:
            thresholds_list = [self.threshold]

        generator = waterz.agglomerate(
            affs=affs.astype(np.float32),
            fragments=fragments_relabelled,
            thresholds=thresholds_list,
            scoring_function=waterz_merge_functions[self.merge_function],
            discretize_queue=256 if fragment_relabel_map is not None else 0,
            return_merge_history=True if fragment_relabel_map is not None else False,
            return_region_graph=True if fragment_relabel_map is not None else False,
        )

        if fragment_relabel_map is not None:
            # add edges to RAG
            _, _, initial_rag = next(generator)
            for edge in initial_rag:
                u, v = fragment_relabel_map[edge["u"]], fragment_relabel_map[edge["v"]]
                rag.add_edge(u, v, merge_score=None, agglomerated=True)

            # Collect merge history
            _, merge_history, _ = next(generator)

            # Cleanup generator
            for _, _, _ in generator:
                pass

            # create a merge tree from the merge history
            merge_tree = MergeTree(fragment_relabel_map)

            for merge in merge_history:
                a, b, c, score = merge["a"], merge["b"], merge["c"], merge["score"]
                merge_tree.merge(
                    fragment_relabel_map[a],
                    fragment_relabel_map[b],
                    fragment_relabel_map[c],
                    score,
                )

            # mark edges in original RAG with score at time of merging
            num_merged = 0
            for u, v, data in rag.edges(data=True):
                merge_score = merge_tree.find_merge(u, v)
                data["adj_weight"] = merge_score
                if merge_score is not None:
                    num_merged += 1

        else:
            data = next(generator)
            # Cleanup generator
            for _ in generator:
                pass

        return data if fragment_relabel_map is None else None

    def compute_fragments(self, affs):
        """Extract initial fragments from affinities using a watershed
        transform. Returns the fragments and the maximal ID in it.
        Returns:
            (fragments, max_id)
            or
            (fragments, max_id, seeds) if return_seeds == True"""
        raise NotImplementedError("Not implemented yet.")
        """
        if self.fragments_in_xy:
            # mean_affs = 0.5 * (affs[1] + affs[2])

            mean_affs = (1 / 3) * (
                affs[0] + affs[1] + affs[2]
            )  # todo: other affinities? *0.5

            depth = mean_affs.shape[0]

            fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
            if self.return_seeds:
                seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

            id_offset = 0
            for z in range(depth):
                boundary_mask = mean_affs[z] > 0.5 * max_affinity_value
                boundary_distances = distance_transform_edt(boundary_mask)

                ret = watershed_from_boundary_distance(
                    boundary_distances,
                    boundary_mask,
                    return_seeds=return_seeds,
                    id_offset=id_offset,
                    min_seed_distance=min_seed_distance,
                )

                fragments[z] = ret[0]
                if return_seeds:
                    seeds[z] = ret[2]

                id_offset = ret[1]

            ret = (fragments, id_offset)
            if return_seeds:
                ret += (seeds,)

        else:
            boundary_mask = np.mean(affs, axis=0) > 0.5 * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            ret = watershed_from_boundary_distance(
                boundary_distances,
                boundary_mask,
                return_seeds,
                min_seed_distance=min_seed_distance,
            )

        return ret
        """


class EmbeddingFrags(Agglom):
    agglom_type: Literal["embeddings"] = "embeddings"
    mask_threshold: float
    distance_threshold: float
    min_seed_distance: int
    sampling_rate: float
    merge_segments: bool
    affs_seeds: bool
    relabel_fragments: bool

    def compute_fragments(self, affs_data):
        raise NotImplementedError("Not implemented yet.")
        """
        return embedding_frag_gen(
            raw.data,
            affs_data,
            mask_threshold=self.mask_threshold,
            distance_threshold=self.distance_threshold,
            # todo: other params
        )
        """
