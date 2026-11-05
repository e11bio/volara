import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi

from volara.blockwise.graph_cluster import Octree


def test_single_level_is_one_block():
    roi = Roi((10, 20, 30), (100, 90, 70))
    t = Octree(roi, 1)
    assert t.num_blocks(0) == 1
    assert t.block_roi(0, (0, 0, 0)) == roi
    assert t.index_of_roi(0, roi) == (0, 0, 0)
    assert t.is_root(0)


def test_grid_shapes_and_clipping_on_non_divisible_roi():
    roi = Roi((0, 0, 0), (100, 90, 70))
    t = Octree(roi, 3)
    assert tuple(t.leaf_shape) == (25, 23, 18)
    assert [t.num_blocks(level) for level in range(3)] == [64, 8, 1]
    # last block along each axis is clipped
    assert t.block_roi(0, (3, 3, 3)) == Roi((75, 69, 54), (25, 21, 16))
    # the union of all blocks of a level is the roi, blocks do not overlap
    for level in range(3):
        covered = np.zeros(tuple(roi.shape), dtype=np.uint8)
        for index in t.blocks(level):
            b = t.block_roi(level, index)
            sl = tuple(slice(o, o + s) for o, s in zip(b.begin, b.shape, strict=False))
            covered[sl] += 1
            assert t.index_of_roi(level, b) == index
        assert covered.min() == 1 and covered.max() == 1


def test_children_are_the_blocks_inside_the_parent():
    t = Octree(Roi((0, 0, 0), (100, 90, 70)), 3)
    for level in (1, 2):
        for index in t.blocks(level):
            parent = t.block_roi(level, index)
            children = t.children(level, index)
            assert 1 <= len(children) <= 8
            for child in children:
                c = t.block_roi(level - 1, child)
                assert parent.contains(c)
                assert t.ancestor(level - 1, child, level) == index
    # an edge block at level 1 has fewer than 8 children when the leaf grid is odd
    t2 = Octree(Roi((0, 0, 0), (30, 30, 30)), 3, leaf_shape=Coordinate(10, 10, 10))
    assert [t2.num_blocks(level) for level in range(3)] == [27, 8, 1]
    assert len(t2.children(1, (1, 1, 1))) == 1
    assert len(t2.children(1, (0, 0, 0))) == 8
    assert len(t2.children(2, (0, 0, 0))) == 8


def test_leaf_index_is_half_open():
    t = Octree(Roi((10, 0, 0), (40, 40, 40)), 3)  # leaf (10, 10, 10)
    pos = np.array(
        [
            [10, 0, 0],  # first leaf, inclusive begin
            [19.999, 9.999, 9.999],  # still first leaf
            [20, 10, 10],  # exclusive end -> next leaf
            [49.999, 39.999, 39.999],  # last leaf
            [50, 0, 0],  # outside (end is exclusive)
            [9.999, 0, 0],  # outside (before begin)
            [np.nan, 0, 0],  # no position
        ]
    )
    idx = t.leaf_index(pos)
    assert idx.tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
        [3, 3, 3],
        [-1, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ]


def test_lca():
    t = Octree(Roi((0, 0, 0), (80, 80, 80)), 4)  # leaf 10, leaf grid 8
    a = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [3, 0, 0], [4, 0, 0], [7, 7, 7]])
    b = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [4, 0, 0], [5, 0, 0], [0, 0, 0]])
    level, index = t.lca(a, b)
    assert level.tolist() == [0, 1, 1, 3, 1, 3]
    assert index.tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [2, 0, 0],
        [0, 0, 0],
    ]
    for (la, lb), lvl, idx in zip(zip(a, b, strict=False), level, index, strict=False):
        block = t.block_roi(int(lvl), tuple(idx))
        assert block.contains(t.block_roi(0, tuple(la)))
        assert block.contains(t.block_roi(0, tuple(lb)))


def test_rejects_unusable_depths():
    with pytest.raises(ValueError, match="too small"):
        Octree(Roi((0, 0, 0), (4, 4, 4)), 4)
    with pytest.raises(ValueError, match="no-op"):
        # leaf 10 over 20 -> 2 leaves per axis -> root at level 1, level 2 is a no-op
        Octree(Roi((0, 0, 0), (20, 20, 20)), 3, leaf_shape=Coordinate(10, 10, 10))
    with pytest.raises(ValueError, match="more than"):
        Octree(Roi((0, 0, 0), (100, 100, 100)), 2, leaf_shape=Coordinate(10, 10, 10))
    with pytest.raises(ValueError):
        Octree(Roi((0, 0, 0), (10, 10, 10)), 0)
