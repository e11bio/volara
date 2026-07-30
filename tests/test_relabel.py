import daisy
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi

from volara.blockwise import Relabel
from volara.datasets import Labels
from volara.lut import LUT
from volara.tmp import (
    filter_mapping_to_block,
    prepare_mapping,
    replace_values,
    replace_values_sorted,
    warmup_replace_values_sorted,
)


def relabel_sorted(arr, src, dst):
    """The faster relabel pipeline, chained exactly as `Relabel.map_block` does."""
    src_sorted, dst_sorted = prepare_mapping(src, dst)
    block_src, block_dst = filter_mapping_to_block(arr, src_sorted, dst_sorted)
    return replace_values_sorted(np.ascontiguousarray(arr), block_src, block_dst)


def test_relabel_init_and_drop(labels_2d, tmp_path):
    """init() creates output zarr, drop_artifacts() removes it."""
    frags_path, _ = labels_2d
    seg_path = tmp_path / "test.zarr" / "seg"
    lut = LUT(path=tmp_path / "lut.npz")
    lut.save(np.array([[1, 2, 3, 4], [10, 20, 30, 40]]))

    task = Relabel(
        frags_data=Labels(store=frags_path),
        seg_data=Labels(store=seg_path),
        lut=lut,
        block_size=Coordinate(10, 10),
    )
    task.init()
    assert seg_path.exists()
    task.drop_artifacts()
    assert not seg_path.exists()


def test_relabel_basic(labels_2d, block_2d, tmp_path):
    """Fragments [1,2,3,4] mapped to segments [10,20,30,40] via LUT."""
    frags_path, frags_data = labels_2d
    seg_path = tmp_path / "test.zarr" / "seg"
    lut = LUT(path=tmp_path / "lut.npz")
    lut.save(np.array([[1, 2, 3, 4], [10, 20, 30, 40]]))

    task = Relabel(
        frags_data=Labels(store=frags_path),
        seg_data=Labels(store=seg_path),
        lut=lut,
        block_size=Coordinate(10, 10),
    )
    task.init()

    with task.process_block_func() as process_block:
        process_block(block_2d)

    result = task.seg_data.array("r")[:]
    expected = np.zeros_like(frags_data, dtype=np.uint64)
    for frag_id, seg_id in [(1, 10), (2, 20), (3, 30), (4, 40)]:
        expected[frags_data == frag_id] = seg_id
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# faster LUT relabel
#
# `replace_values` builds a numba typed dict of the whole LUT and probes it once
# per voxel. `map_block` now instead sorts the LUT once per task
# (`prepare_mapping`), narrows it to the ids actually present in the block
# (`filter_mapping_to_block`), and binary-searches per voxel
# (`replace_values_sorted`). Different lookup structure, a different mapping
# size per block, and a different miss path -so what these tests pin is the one
# thing that shouldnt change: the output array. `replace_values` is still in
# `tmp.py` and serves as the reference.
#
# The speedup only shows on volumes far larger than a unit test should build, so
# it is not asserted here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr, src, dst, case",
    [
        # every value mapped
        ([1, 2, 3, 4], [1, 2, 3, 4], [10, 20, 30, 40], "exhaustive"),
        # values with no LUT entry pass through untouched
        ([1, 2, 3, 99], [1, 2], [10, 20], "unmapped_passthrough"),
        # background is just another id -- mapped if listed, kept if not
        ([0, 1, 0, 2], [1, 2], [10, 20], "background_unmapped"),
        ([0, 1, 0, 2], [0, 1, 2], [7, 10, 20], "background_mapped"),
        # prepare_mapping sorts, so an unsorted LUT must behave identically
        ([1, 2, 3, 4], [4, 1, 3, 2], [40, 10, 30, 20], "unsorted_lut"),
        # LUT far wider than the block: filter_mapping_to_block drops the rest
        ([5, 6], list(range(1, 21)), [i * 100 for i in range(1, 21)], "lut_superset"),
        # ids entirely below / above the LUT, exercising both searchsorted ends
        ([1, 2], [50, 51], [500, 510], "all_below_lut"),
        ([90, 91], [50, 51], [500, 510], "all_above_lut"),
        # empty LUT -> identity
        ([1, 2, 3], [], [], "empty_lut"),
        # full uint64 range, where a float64 detour would silently lose bits
        (
            [2**63 + 1, 2**63 + 2],
            [2**63 + 1, 2**63 + 2],
            [2**63 + 10, 2**63 + 20],
            "uint64_range",
        ),
        # a mapping that permutes ids among themselves
        ([1, 2, 3], [1, 2, 3], [3, 1, 2], "permutation"),
        ([1, 2, 3], [1, 2, 3], [1, 2, 3], "identity"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_replace_values_sorted_matches_replace_values(arr, src, dst, case):
    arr = np.array(arr, dtype=np.uint64)
    src = np.array(src, dtype=np.uint64)
    dst = np.array(dst, dtype=np.uint64)

    expected = replace_values(arr, src, dst)
    actual = relabel_sorted(arr, src, dst)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("shape", [(10,), (4, 5), (3, 4, 5), (2, 3, 4, 5)])
def test_replace_values_sorted_matches_replace_values_nd(shape):
    """Shape is preserved and the values agree, 1D through 4D."""
    arr = np.random.default_rng(0).integers(0, 12, size=shape).astype(np.uint64)
    # Map only some of the ids, so both the hit and the miss path are exercised.
    src = np.array([0, 2, 3, 5, 7, 11], dtype=np.uint64)
    dst = np.array([100, 102, 103, 105, 107, 111], dtype=np.uint64)

    actual = relabel_sorted(arr, src, dst)

    assert actual.shape == shape
    np.testing.assert_array_equal(actual, replace_values(arr, src, dst))


def test_replace_values_sorted_matches_replace_values_sparse_lut():
    """A wide, sparse LUT over many voxels - the regime `Relabel` runs in, where
    most of the LUT is irrelevant to any single block."""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 5_000, size=(20, 30, 40)).astype(np.uint64)
    src = rng.choice(20_000, size=8_000, replace=False).astype(np.uint64)
    dst = (src + 1_000_000).astype(np.uint64)

    np.testing.assert_array_equal(
        relabel_sorted(arr, src, dst), replace_values(arr, src, dst)
    )


def test_prepare_mapping_sorts_and_keeps_pairs_together():
    """The sort permutes both arrays, not just src."""
    src = np.array([9, 3, 7, 1], dtype=np.uint64)
    dst = np.array([90, 30, 70, 10], dtype=np.uint64)

    src_sorted, dst_sorted = prepare_mapping(src, dst)

    np.testing.assert_array_equal(src_sorted, [1, 3, 7, 9])
    np.testing.assert_array_equal(dst_sorted, [10, 30, 70, 90])
    assert src_sorted.flags["C_CONTIGUOUS"] and dst_sorted.flags["C_CONTIGUOUS"]


def test_filter_mapping_to_block_keeps_only_present_ids():
    """The per-block narrowing keeps exactly the entries whose src id occurs in the
    block, so the binary search runs against the smallest possible table."""
    src_sorted, dst_sorted = prepare_mapping(
        np.array([1, 2, 3, 4, 5], dtype=np.uint64),
        np.array([10, 20, 30, 40, 50], dtype=np.uint64),
    )
    block = np.array([[2, 2], [4, 99]], dtype=np.uint64)

    block_src, block_dst = filter_mapping_to_block(block, src_sorted, dst_sorted)

    np.testing.assert_array_equal(block_src, [2, 4])
    np.testing.assert_array_equal(block_dst, [20, 40])


def test_filter_mapping_to_block_empty_block_mapping():
    """A block sharing no ids with the LUT narrows to an empty mapping, and the
    relabel then leaves it alone."""
    src_sorted, dst_sorted = prepare_mapping(
        np.array([1, 2], dtype=np.uint64), np.array([10, 20], dtype=np.uint64)
    )
    block = np.array([[7, 8], [9, 9]], dtype=np.uint64)

    block_src, block_dst = filter_mapping_to_block(block, src_sorted, dst_sorted)

    assert block_src.size == 0 and block_dst.size == 0
    np.testing.assert_array_equal(
        replace_values_sorted(block, block_src, block_dst), block
    )


def test_warmup_replace_values_sorted_is_callable():
    """`process_block_func` calls this before the block loop so the numba compile
    does not land inside a timed block. It must not raise."""
    warmup_replace_values_sorted()


def test_relabel_task_matches_replace_values(labels_2d, block_2d, tmp_path):
    """End to end: the task's output equals feeding the whole volume through
    `replace_values` in one shot."""
    frags_path, frags_data = labels_2d
    lut = LUT(path=tmp_path / "lut.npz")
    src = np.array([1, 2, 3, 4], dtype=np.uint64)
    dst = np.array([10, 20, 30, 40], dtype=np.uint64)
    lut.save(np.array([src, dst]))

    task = Relabel(
        frags_data=Labels(store=frags_path),
        seg_data=Labels(store=tmp_path / "test.zarr" / "seg_oracle"),
        lut=lut,
        block_size=Coordinate(10, 10),
    )
    task.init()

    with task.process_block_func() as process_block:
        process_block(block_2d)

    np.testing.assert_array_equal(
        task.seg_data.array("r")[:],
        replace_values(frags_data.astype(np.uint64), src, dst),
    )


def test_relabel_task_blockwise_matches_single_block(labels_2d, tmp_path):
    """Splitting the volume into blocks changes nothing.

    `map_block` narrows the LUT per block, so each block sees a *different*
    mapping array. The stitched result must still equal the whole-volume answer.
    """
    frags_path, frags_data = labels_2d
    lut = LUT(path=tmp_path / "lut.npz")
    src = np.array([1, 2, 3, 4], dtype=np.uint64)
    dst = np.array([10, 20, 30, 40], dtype=np.uint64)
    lut.save(np.array([src, dst]))

    total_roi = Roi((0, 0), (10, 10))
    task = Relabel(
        frags_data=Labels(store=frags_path),
        seg_data=Labels(store=tmp_path / "test.zarr" / "seg_blockwise"),
        lut=lut,
        block_size=Coordinate(5, 5),
    )
    task.init()

    with task.process_block_func() as process_block:
        for z in range(0, 10, 5):
            for y in range(0, 10, 5):
                write_roi = Roi((z, y), (5, 5))
                process_block(
                    daisy.Block(
                        total_roi=total_roi, read_roi=write_roi, write_roi=write_roi
                    )
                )

    np.testing.assert_array_equal(
        task.seg_data.array("r")[:],
        replace_values(frags_data.astype(np.uint64), src, dst),
    )
