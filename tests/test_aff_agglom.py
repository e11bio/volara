"""Tests for AffAgglom.

`AffAgglom.agglomerate` speedups - relabels fragments to a dense 1..k range with
a single `np.unique(return_inverse=True)` rather than one full-array scan per
fragment, and accumulates per-pair affinity statistics over only the boundary
voxels using an integer cantor pairing and `np.bincount` rather than full-volume
pairings handed to `scipy.ndimage.mean` / `sum_labels`. `reference_edges` below
is a brute force test that walks every neighbour voxel pair and accumulates into
a plain dict, using neither the relabel nor the cantor pairing, so agreement
with it is should provide evidence that the optimization gives the same output.
"""

import networkx as nx
import numpy as np
import pytest
from funlib.geometry import Coordinate
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import AffAgglom
from volara.datasets import Affs, Labels

NEIGHBORHOOD_2D = [[1, 0], [0, 1]]


def write_affs(path, data, neighborhood=NEIGHBORHOOD_2D):
    arr = prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )
    arr[:] = data
    arr._source_data.attrs["neighborhood"] = neighborhood
    return path


def make_task(db, frags_path, affs_path, scores) -> AffAgglom:
    return AffAgglom(
        db=db,
        frags_data=Labels(store=frags_path),
        affs_data=Affs(store=affs_path),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        scores=scores,
    )


def agglomerate(task, affs, frags) -> dict:
    """Run `agglomerate` on a fresh graph and return {(u, v): attrs}, with the
    endpoints sorted so comparisons don't depend on edge orientation."""
    rag = nx.Graph()
    task.agglomerate(affs, frags, rag)
    return {(min(u, v), max(u, v)): data for u, v, data in rag.edges(data=True)}


def slices(offset):
    base = tuple(slice(-m if m < 0 else None, -m if m > 0 else None) for m in offset)
    shifted = tuple(slice(m if m > 0 else None, m if m < 0 else None) for m in offset)
    return base, shifted


def reference_edges(neighborhood, scores, affs, frags) -> dict:
    """Brute-force per-pair affinity means.

    Walks every voxel pair in the neighborhood one at a time and accumulates
    (sum, count) into a dict keyed by the original fragment ids. A score's weight
    is the mean affinity over all boundary voxels of all its offsets, which is
    what `agglomerate`'s count-weighted combination reduces to.
    """
    totals = {name: {} for name in scores}
    for score_name, offsets in scores.items():
        for offset in offsets:
            channel = list(neighborhood).index(offset)
            base_slice, offset_slice = slices(offset)
            base_frags = frags[base_slice]
            offset_frags = frags[offset_slice]
            base_affs = affs[channel][base_slice]
            for idx in np.ndindex(base_frags.shape):
                u, v = int(offset_frags[idx]), int(base_frags[idx])
                if u == v or u == 0 or v == 0:
                    continue
                acc = totals[score_name].setdefault((min(u, v), max(u, v)), [0.0, 0])
                acc[0] += float(base_affs[idx])
                acc[1] += 1
    return {
        name: {pair: acc[0] / acc[1] for pair, acc in pairs.items()}
        for name, pairs in totals.items()
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_aff_agglom_drop_edges(frags_2d, sqlite_db_2d, block_2d, tmp_path):
    """drop_artifacts() removes edges but preserves nodes."""
    frags_path, _ = frags_2d

    # Create 1-channel affs (needed for the task config, not for the drop test)
    affs_path = write_affs(
        tmp_path / "test.zarr" / "affs",
        np.zeros((1, 10, 10), dtype=np.float32),
        neighborhood=[[1, 0]],
    )

    # Seed the DB with nodes and an edge
    db = sqlite_db_2d.open("r+")
    g = db.read_graph()
    g.add_node(1, position=(1, 5), size=1, raw_intensity=(1,))
    g.add_node(2, position=(2, 5), size=1, raw_intensity=(2,))
    g.add_edge(1, 2, y_aff=0.5)
    db.write_graph(g)

    # Verify edges exist before drop
    g = sqlite_db_2d.open("r").read_graph()
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1

    task = make_task(sqlite_db_2d, frags_path, affs_path, {"y_aff": [Coordinate(1, 0)]})
    task.drop_artifacts()

    # Edges should be gone, nodes should remain
    g = sqlite_db_2d.open("r").read_graph()
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 0


def test_aff_agglom_basic(frags_2d, sqlite_db_2d, block_2d, tmp_path):
    """10 horizontal stripe fragments, alternating affinities -> 9 edges with correct scores."""
    frags_path, _ = frags_2d

    # Seed DB with 10 fragment nodes matching the stripe labels (1..10)
    db = sqlite_db_2d.open("r+")
    g = db.read_graph()
    for i in range(1, 11):
        g.add_node(i, position=(i, 5), size=1, raw_intensity=(i,))
    db.write_graph(g)

    # Create 1-channel affs with offset [1,0]: 1 in every other row, 0 elsewhere
    affs_data = np.zeros((1, 10, 10), dtype=np.uint32)
    affs_data[0, ::2, :] = 1
    affs_path = write_affs(
        tmp_path / "test.zarr" / "affs", affs_data, neighborhood=[[1, 0]]
    )

    task = make_task(sqlite_db_2d, frags_path, affs_path, {"y_aff": [Coordinate(1, 0)]})

    with task.process_block_func() as process_block:
        process_block(block_2d)

    g = sqlite_db_2d.open("r").read_graph(block_2d.write_roi)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 9
    for u, v, data in g.edges(data=True):
        if u % 2 == 0:
            assert data["y_aff"] == 0.0
        else:
            assert data["y_aff"] == 1.0


# ---------------------------------------------------------------------------
# edge weights
# ---------------------------------------------------------------------------


def test_agglomerate_weight_is_mean_over_all_boundary_voxels(
    frags_2d, sqlite_db_2d, tmp_path
):
    """When one score groups several offsets, its weight is the mean affinity over
    the boundary voxels of all of them pooled, not a mean of per-offset means.

    Fragment 1 fills the top-left 5x5 corner of a block that is otherwise
    fragment 2, so each offset contributes 5 boundary voxels: 5 at 0.2 along
    (1, 0) and 5 at 0.6 along (0, 1). The pooled mean is 4.0 / 10 = 0.4, whereas
    averaging the two per-offset means would also give 0.4 only because the counts
    happen to match, so the asymmetric case is covered by the reference tests
    below.
    """
    frags_path, _ = frags_2d

    affs = np.zeros((2, 10, 10), dtype=np.float32)
    affs[0, 4, 0:5] = 0.2  # (1, 0) boundary voxels
    affs[1, 0:5, 4] = 0.6  # (0, 1) boundary voxels
    affs_path = write_affs(tmp_path / "test.zarr" / "affs", affs)

    frags = np.full((10, 10), 2, dtype=np.uint64)
    frags[:5, :5] = 1

    task = make_task(
        sqlite_db_2d,
        frags_path,
        affs_path,
        {"aff": [Coordinate(1, 0), Coordinate(0, 1)]},
    )

    edges = agglomerate(task, affs, frags)

    assert set(edges) == {(1, 2)}
    assert edges[(1, 2)]["aff"] == pytest.approx(0.4)


def test_agglomerate_separate_scores_stay_separate(frags_2d, sqlite_db_2d, tmp_path):
    """Each score name aggregates only its own offsets."""
    frags_path, _ = frags_2d

    affs = np.zeros((2, 10, 10), dtype=np.float32)
    affs[0, 4, 0:5] = 0.2
    affs[1, 0:5, 4] = 0.6
    affs_path = write_affs(tmp_path / "test.zarr" / "affs", affs)

    frags = np.full((10, 10), 2, dtype=np.uint64)
    frags[:5, :5] = 1

    task = make_task(
        sqlite_db_2d,
        frags_path,
        affs_path,
        {"z_aff": [Coordinate(1, 0)], "y_aff": [Coordinate(0, 1)]},
    )

    edges = agglomerate(task, affs, frags)

    assert edges[(1, 2)]["z_aff"] == pytest.approx(0.2)
    assert edges[(1, 2)]["y_aff"] == pytest.approx(0.6)


def test_agglomerate_ignores_background(frags_2d, sqlite_db_2d, tmp_path):
    """Two fragments separated by a row of background share no edge, even though
    both touch the background."""
    frags_path, _ = frags_2d

    affs = np.ones((2, 10, 10), dtype=np.float32)
    affs_path = write_affs(tmp_path / "test.zarr" / "affs", affs)

    frags = np.zeros((10, 10), dtype=np.uint64)
    frags[0:4] = 1
    frags[5:10] = 2  # row 4 is background

    task = make_task(
        sqlite_db_2d,
        frags_path,
        affs_path,
        {"aff": [Coordinate(1, 0), Coordinate(0, 1)]},
    )

    assert agglomerate(task, affs, frags) == {}


def test_agglomerate_empty_block(frags_2d, sqlite_db_2d, tmp_path):
    """An all-background block adds no edges and does not raise."""
    frags_path, _ = frags_2d

    affs = np.ones((2, 10, 10), dtype=np.float32)
    affs_path = write_affs(tmp_path / "test.zarr" / "affs", affs)

    task = make_task(sqlite_db_2d, frags_path, affs_path, {"aff": [Coordinate(1, 0)]})

    assert agglomerate(task, affs, np.zeros((10, 10), dtype=np.uint64)) == {}


# ---------------------------------------------------------------------------
# dense relabel: edges must come back keyed by the original fragment ids
# ---------------------------------------------------------------------------


def test_agglomerate_reports_original_ids_without_background(
    frags_2d, sqlite_db_2d, tmp_path
):
    """A block with no background at all.

    `np.unique(return_inverse=True)` maps the lowest value to 0, so on a block
    where nothing is background the relabel has to shift by one or the lowest
    fragment is silently swallowed as background and loses all its edges.
    """
    frags_path, _ = frags_2d

    affs = np.full((2, 10, 10), 0.5, dtype=np.float32)
    affs_path = write_affs(tmp_path / "test.zarr" / "affs", affs)

    frags = np.full((10, 10), 7, dtype=np.uint64)
    frags[:5] = 5  # ids 5 and 7, no zeros anywhere

    task = make_task(sqlite_db_2d, frags_path, affs_path, {"aff": [Coordinate(1, 0)]})

    edges = agglomerate(task, affs, frags)

    assert set(edges) == {(5, 7)}
    assert edges[(5, 7)]["aff"] == pytest.approx(0.5)


def test_agglomerate_reports_original_ids_for_large_labels(
    frags_2d, sqlite_db_2d, tmp_path
):
    """Real fragment ids are block-bumped into the billions. The relabel exists so
    the cantor pairing stays small; the edges still have to come back keyed by the
    original ids."""
    frags_path, _ = frags_2d

    affs = np.full((2, 10, 10), 0.25, dtype=np.float32)
    affs_path = write_affs(tmp_path / "test.zarr" / "affs", affs)

    big = np.uint64(4_000_000_000)
    frags = np.zeros((10, 10), dtype=np.uint64)
    frags[0:5] = big
    frags[5:10] = big + np.uint64(1)

    task = make_task(sqlite_db_2d, frags_path, affs_path, {"aff": [Coordinate(1, 0)]})

    edges = agglomerate(task, affs, frags)

    assert set(edges) == {(int(big), int(big) + 1)}


# ---------------------------------------------------------------------------
# equivalence against the brute-force reference
# ---------------------------------------------------------------------------


def make_frags(layout) -> np.ndarray:
    frags = np.zeros((10, 10), dtype=np.uint64)
    if layout == "quadrants":
        frags[:5, :5], frags[:5, 5:] = 1, 2
        frags[5:, :5], frags[5:, 5:] = 3, 4
    elif layout == "stripes":
        frags[:] = np.arange(1, 11, dtype=np.uint64)[:, None]
    elif layout == "no_background":
        frags[:] = np.arange(11, 21, dtype=np.uint64)[None, :]
    elif layout == "with_background_holes":
        frags[:] = 1
        frags[5:, :] = 2
        frags[3:6, 3:6] = 0
    elif layout == "random":
        frags[:] = np.random.default_rng(2).integers(0, 5, size=(10, 10))
    elif layout == "random_dense":
        # No zeros, and many distinct pairs - the worst case for the relabel.
        frags[:] = np.random.default_rng(3).integers(1, 8, size=(10, 10))
    elif layout == "single":
        frags[2:8, 2:8] = 42
    elif layout == "large_ids":
        frags[:5] = 4_000_000_000
        frags[5:] = 4_000_000_017
    else:
        raise ValueError(layout)
    return frags


@pytest.mark.parametrize(
    "layout",
    [
        "quadrants",
        "stripes",
        "no_background",
        "with_background_holes",
        "random",
        "random_dense",
        "single",
        "large_ids",
    ],
)
@pytest.mark.parametrize(
    "scores",
    [
        {"aff": [Coordinate(1, 0)]},
        {"aff": [Coordinate(1, 0), Coordinate(0, 1)]},
        {"z_aff": [Coordinate(1, 0)], "y_aff": [Coordinate(0, 1)]},
    ],
    ids=["one_offset", "pooled_offsets", "split_scores"],
)
def test_agglomerate_matches_bruteforce(
    frags_2d, sqlite_db_2d, tmp_path, layout, scores
):
    """The cantor-paired, boundary-only accumulation agrees with a per-voxel dict
    walk, across fragment layouts and score groupings."""
    frags_path, _ = frags_2d

    # Random affinities so the per-pair means are all distinct -constant affs
    # would let a mis-assigned pair pass unnoticed.
    affs = np.random.default_rng(4).random((2, 10, 10)).astype(np.float32)
    affs_path = write_affs(tmp_path / "test.zarr" / "affs", affs)

    frags = make_frags(layout)
    task = make_task(sqlite_db_2d, frags_path, affs_path, scores)

    edges = agglomerate(task, affs, frags)
    expected = reference_edges(task.affs_data.neighborhood, scores, affs, frags)

    # An edge exists iff some score found boundary voxels for it
    assert set(edges) == {pair for pairs in expected.values() for pair in pairs}

    for pair, data in edges.items():
        # ...and carries exactly the scores that did, with their weights. A score
        # whose offsets never straddle this pair is absent, not zero.
        assert set(data) == {
            score_name for score_name, pairs in expected.items() if pair in pairs
        }, pair
        for score_name, value in data.items():
            assert value == pytest.approx(expected[score_name][pair]), (
                f"{score_name} on {pair}"
            )


def test_agglomerate_does_not_mutate_fragments(frags_2d, sqlite_db_2d, tmp_path):
    """The relabel rebinds rather than writing into the caller's block, which
    `agglomerate_in_block` hands straight from `to_ndarray`."""
    frags_path, _ = frags_2d

    affs = np.full((2, 10, 10), 0.5, dtype=np.float32)
    affs_path = write_affs(tmp_path / "test.zarr" / "affs", affs)

    frags = make_frags("large_ids")
    original = frags.copy()

    task = make_task(sqlite_db_2d, frags_path, affs_path, {"aff": [Coordinate(1, 0)]})
    agglomerate(task, affs, frags)

    np.testing.assert_array_equal(frags, original)
