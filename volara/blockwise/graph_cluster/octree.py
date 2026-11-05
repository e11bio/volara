"""Spatial 2^d-tree over a ROI: block grids per level, node ownership, and
lowest-common-ancestor (LCA) lookup for edges.

Every level is a regular grid of blocks anchored at ``roi.begin``. The block
shape doubles per level, so a block at level ``L`` with grid index ``g`` is
exactly the union of the level ``L-1`` blocks with indices ``2g .. 2g+1`` per
dimension (clipped to the ROI). Node ownership is half-open: a node belongs to
the leaf whose ``[begin, end)`` box contains its position, matching the
database's spatial query. Nothing here knows about graphs or clustering.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product

import numpy as np
from funlib.geometry import Coordinate, Roi

Index = tuple[int, ...]


class Octree:
    def __init__(
        self,
        roi: Roi,
        num_levels: int,
        leaf_shape: Coordinate | None = None,
    ):
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        if any(b is None for b in roi.begin) or any(s is None for s in roi.shape):
            raise ValueError(f"GraphCluster needs a bounded roi, got {roi}")

        self.roi = roi
        self.dims = roi.dims
        self.num_levels = num_levels
        self.begin = np.asarray(roi.begin, dtype=np.int64)
        self.shape = np.asarray(roi.shape, dtype=np.int64)

        splits = 2 ** (num_levels - 1)
        if leaf_shape is None:
            if np.any(self.shape < splits):
                raise ValueError(
                    f"roi shape {tuple(roi.shape)} is too small for num_levels="
                    f"{num_levels}: leaves would be below one unit along some "
                    "axis. Use fewer levels or a larger roi."
                )
            leaf = -(-self.shape // splits)  # ceil
        else:
            leaf = np.asarray(tuple(leaf_shape), dtype=np.int64)
            if leaf.shape != (self.dims,) or np.any(leaf < 1):
                raise ValueError(f"invalid leaf shape {leaf_shape}")
        self.leaf_shape = leaf
        self.leaf_grid = -(-self.shape // leaf)

        if np.any(self.grid_shape(num_levels - 1) != 1):
            raise ValueError(
                f"leaf shape {tuple(int(x) for x in leaf)} needs more than {num_levels} levels "
                f"to reach a single root block over roi shape {tuple(roi.shape)} "
                f"(leaf grid {tuple(int(x) for x in self.leaf_grid)})."
            )
        if num_levels > 1 and np.all(self.leaf_grid <= 2 ** (num_levels - 2)):
            raise ValueError(
                f"num_levels={num_levels} but the leaf grid {tuple(int(x) for x in self.leaf_grid)} "
                f"already reaches a single root block at level {num_levels - 2}; "
                "the top level would be a no-op. Reduce num_levels or the leaf "
                "shape rather than silently changing what num_levels means."
            )

    @staticmethod
    def levels_for(roi: Roi, leaf_shape: Coordinate) -> int:
        """The smallest number of levels for which ``leaf_shape`` leaves over
        ``roi`` reach a single root block."""
        shape = np.asarray(roi.shape, dtype=np.int64)
        leaf = np.asarray(tuple(leaf_shape), dtype=np.int64)
        grid = -(-shape // leaf)
        levels = 1
        while np.any(-(-grid // (2 ** (levels - 1))) > 1):
            levels += 1
        return levels

    # -- grids -------------------------------------------------------------

    def block_shape(self, level: int) -> np.ndarray:
        """Unclipped block shape at ``level`` (leaf shape times 2**level)."""
        return self.leaf_shape * (2**level)

    def grid_shape(self, level: int) -> np.ndarray:
        return -(-self.leaf_grid // (2**level))

    def num_blocks(self, level: int) -> int:
        return int(np.prod(self.grid_shape(level)))

    def blocks(self, level: int) -> Iterator[Index]:
        yield from product(*(range(int(n)) for n in self.grid_shape(level)))

    def block_roi(self, level: int, index: Index) -> Roi:
        """The ROI of a block, clipped to the tree's roi."""
        idx = np.asarray(index, dtype=np.int64)
        begin = self.begin + idx * self.block_shape(level)
        end = np.minimum(begin + self.block_shape(level), self.begin + self.shape)
        return Roi(Coordinate(*begin.tolist()), Coordinate(*(end - begin).tolist()))

    def index_of_roi(self, level: int, roi: Roi) -> Index:
        """Grid index of the block whose (clipped) roi is ``roi``."""
        idx = (np.asarray(roi.begin, dtype=np.int64) - self.begin) // self.block_shape(
            level
        )
        index = tuple(int(i) for i in idx)
        if self.block_roi(level, index) != roi:
            raise ValueError(
                f"{roi} is not a level {level} block of this octree "
                f"(closest index {index} -> {self.block_roi(level, index)})"
            )
        return index

    def is_root(self, level: int) -> bool:
        return level == self.num_levels - 1

    # -- ownership -----------------------------------------------------------

    def leaf_index(self, positions: np.ndarray) -> np.ndarray:
        """Leaf grid index per position, ``-1`` on every axis for positions
        outside the roi or with a NaN coordinate. Half-open on each axis."""
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, self.dims)
        finite = np.all(np.isfinite(pos), axis=1)
        pos = np.where(finite[:, None], pos, self.begin)  # placeholder, masked below
        idx = np.floor((pos - self.begin) / self.leaf_shape).astype(np.int64)
        # guard against float rounding at block boundaries: the authoritative
        # test is the half-open comparison, exactly as the database does it
        lo = self.begin + idx * self.leaf_shape
        idx -= pos < lo
        idx += pos >= lo + self.leaf_shape
        inside = (
            finite
            & np.all(pos >= self.begin, axis=1)
            & np.all(pos < self.begin + self.shape, axis=1)
        )
        idx[~inside] = -1
        return idx

    def lca(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lowest common ancestor of leaf indices ``a`` and ``b`` (``(n, d)``
        arrays): the level and grid index of the smallest block containing
        both leaves."""
        a = np.asarray(a, dtype=np.int64).reshape(-1, self.dims)
        b = np.asarray(b, dtype=np.int64).reshape(-1, self.dims)
        x = a ^ b
        bits = np.zeros_like(x)
        for k in range(1, self.num_levels + 1):
            bits = np.where((x >> (k - 1)) > 0, k, bits)
        level = bits.max(axis=1)
        index = a >> level[:, None]
        return level, index

    def ancestor(self, level: int, index: Index, target_level: int) -> Index:
        shift = target_level - level
        assert shift >= 0
        return tuple(int(i) >> shift for i in index)

    def children(self, level: int, index: Index) -> list[Index]:
        """Existing children of a block (fewer than 2**d at the roi edge)."""
        if level == 0:
            return []
        grid = self.grid_shape(level - 1)
        out = []
        for offset in product((0, 1), repeat=self.dims):
            child = tuple(2 * int(i) + o for i, o in zip(index, offset, strict=False))
            if all(c < g for c, g in zip(child, grid, strict=False)):
                out.append(child)
        return out

    @staticmethod
    def key(level: int, index: Index) -> str:
        return f"L{level}_" + "_".join(str(int(i)) for i in index)

    def describe(self) -> dict:
        return {
            "roi": (tuple(self.roi.begin), tuple(self.roi.shape)),
            "num_levels": self.num_levels,
            "leaf_shape": tuple(int(x) for x in self.leaf_shape),
            "blocks_per_level": [
                self.num_blocks(level) for level in range(self.num_levels)
            ],
        }
