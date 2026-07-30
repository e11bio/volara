import numba
import numpy as np


def seg_to_affgraph(seg, nhood):
    nhood = np.array(nhood)

    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    dims = nhood.shape[1]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    if dims == 2:
        for e in range(nEdge):
            aff[
                e,
                max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
            ] = (
                (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                    ]
                    == seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                    ]
                )
                * (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                    ]
                    > 0
                )
                * (
                    seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                    ]
                    > 0
                )
            )

    elif dims == 3:
        for e in range(nEdge):
            aff[
                e,
                max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
            ] = (
                (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                        max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
                    ]
                    == seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                        max(0, nhood[e, 2]) : min(shape[2], shape[2] + nhood[e, 2]),
                    ]
                )
                * (
                    seg[
                        max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                        max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                        max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
                    ]
                    > 0
                )
                * (
                    seg[
                        max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                        max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                        max(0, nhood[e, 2]) : min(shape[2], shape[2] + nhood[e, 2]),
                    ]
                    > 0
                )
            )

    else:
        raise RuntimeError(f"AddAffinities works only in 2 or 3 dimensions, not {dims}")

    return aff


@numba.njit(parallel=True)
def replace_values(arr, src, dst):
    shape = arr.shape
    arr = arr.ravel()
    label_map = {src[i]: dst[i] for i in range(len(src))}
    relabeled_arr = np.zeros_like(arr)

    for i in numba.prange(arr.shape[0]):  # type: ignore[non-iterable]
        relabeled_arr[i] = label_map.get(arr[i], arr[i])

    return relabeled_arr.reshape(shape)


def prepare_mapping(src, dst):
    src = np.asarray(src, dtype=np.uint64)
    dst = np.asarray(dst, dtype=np.uint64)

    order = np.argsort(src)

    return (
        np.ascontiguousarray(src[order]),
        np.ascontiguousarray(dst[order]),
    )


def filter_mapping_to_block(in_frags, src_sorted, dst_sorted):
    block_ids = np.unique(in_frags)

    idx = np.searchsorted(src_sorted, block_ids)

    valid = idx < src_sorted.size
    idx = idx[valid]
    block_ids = block_ids[valid]

    matched = src_sorted[idx] == block_ids

    return (
        np.ascontiguousarray(src_sorted[idx[matched]]),
        np.ascontiguousarray(dst_sorted[idx[matched]]),
    )


@numba.njit(parallel=True)
def replace_values_sorted(arr, src_sorted, dst_sorted):
    shape = arr.shape
    flat = arr.ravel()
    out = np.empty_like(flat)

    for i in numba.prange(flat.size):
        v = flat[i]
        j = np.searchsorted(src_sorted, v)

        if j < src_sorted.size and src_sorted[j] == v:
            out[i] = dst_sorted[j]
        else:
            out[i] = v

    return out.reshape(shape)


def warmup_replace_values_sorted():
    _ = replace_values_sorted(
        np.zeros((4, 4, 4), dtype=np.uint64),
        np.array([0], dtype=np.uint64),
        np.array([0], dtype=np.uint64),
    )
