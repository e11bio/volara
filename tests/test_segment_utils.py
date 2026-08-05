import numpy as np

from volara.segment_utils import replace_values, seg_to_affgraph


def test_seg_to_affgraph_2d():
    """4x4 labels with 2 regions, check affinity edges."""
    seg = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.uint64,
    )
    nhood = [[1, 0], [0, 1]]
    affs = seg_to_affgraph(seg, nhood=nhood)
    assert affs.shape == (2, 4, 4)
    # Within-region affinities should be 1 where both voxels are same nonzero label
    # Cross-boundary affinities should be 0
    # Offset [0,1]: column-wise neighbors. Columns 0-1 are label 1, 2-3 are label 2
    # boundary at col 1->2: affs[1, :, 1] should be 0
    assert np.all(affs[1, :, 1] == 0)
    # within label 1: affs[1, :, 0] should be 1
    assert np.all(affs[1, :, 0] == 1)


def test_seg_to_affgraph_3d():
    """4x4x4 labels, verify 3D neighborhoods."""
    seg = np.ones((4, 4, 4), dtype=np.uint64)
    seg[:2] = 1
    seg[2:] = 2
    nhood = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    affs = seg_to_affgraph(seg, nhood=nhood)
    assert affs.shape == (3, 4, 4, 4)
    # Boundary at z=1->2: offset [1,0,0] should produce 0 at z=1
    assert np.all(affs[0, 1, :, :] == 0)
    # Within label 1: z=0->1 should be 1
    assert np.all(affs[0, 0, :, :] == 1)


def test_seg_to_affgraph_background():
    """Background (0) produces no affinity edges."""
    seg = np.zeros((4, 4), dtype=np.uint64)
    nhood = [[1, 0], [0, 1]]
    affs = seg_to_affgraph(seg, nhood=nhood)
    assert np.all(affs == 0)


def test_replace_values():
    """Basic label remapping."""
    arr = np.array([1, 2, 3, 4], dtype=np.int64)
    src = np.array([1, 2, 3, 4], dtype=np.int64)
    dst = np.array([10, 20, 30, 40], dtype=np.int64)
    result = replace_values(arr, src, dst)
    np.testing.assert_array_equal(result, [10, 20, 30, 40])


def test_replace_values_unmapped():
    """Unmapped values are preserved."""
    arr = np.array([1, 2, 3, 99], dtype=np.int64)
    src = np.array([1, 2], dtype=np.int64)
    dst = np.array([10, 20], dtype=np.int64)
    result = replace_values(arr, src, dst)
    np.testing.assert_array_equal(result, [10, 20, 3, 99])
