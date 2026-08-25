"""Tests for ExtractFrags.

The fragment filter, the fragment centroids and the affinity-shift construction
are all optimized - `np.bincount` over a `np.unique` inverse, and in-place
arithmetic on a single buffer - rather than the standard per-label scipy calls.
Speed is only worth having if the answer is unchanged, so each optimization has
a `reference_*` helper below with the previous implementation, and an
equivalence test asserting the two agree.
"""

import mwatershed as mws
import numpy as np
import pytest
from funlib.geometry import Coordinate
from funlib.persistence.arrays import prepare_ds
from scipy import ndimage

from volara.blockwise import ExtractFrags
from volara.datasets import Affs, Labels
from volara.dbs import SQLite
from volara.segment_utils import replace_values


def make_task(affs_path, tmp_path, **kwargs) -> ExtractFrags:
    """An ExtractFrags over `affs_path`, for exercising its methods directly."""
    return ExtractFrags(
        db=SQLite(path=tmp_path / "test.zarr" / "unit_db.sqlite", ndim=2),
        affs_data=Affs(store=affs_path),
        frags_data=Labels(store=tmp_path / "test.zarr" / "unit_frags"),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        bias=kwargs.pop("bias", [-0.5, -0.5]),
        **kwargs,
    )


def write_affs(path, data, neighborhood) -> None:
    arr = prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )
    arr[:] = data
    arr._source_data.attrs["neighborhood"] = neighborhood


# ---------------------------------------------------------------------------
# previous reference implementations
# ---------------------------------------------------------------------------


def reference_filter_avg_fragments(affs, fragments_data, filter_value, ndim):
    """`scipy.ndimage.mean` with an index array, which rescans the whole volume
    once per label. `ExtractFrags.filter_avg_fragments` replaces this."""
    average_affs = np.mean(affs[0:ndim], axis=0)
    fragment_ids = np.unique(fragments_data)
    filtered = [
        fragment
        for fragment, mean in zip(
            fragment_ids, ndimage.mean(average_affs, fragments_data, fragment_ids)
        )
        if mean < filter_value
    ]
    filtered = np.array(filtered, dtype=fragments_data.dtype)
    if filtered.size == 0:
        return fragments_data
    return replace_values(fragments_data, filtered, np.zeros_like(filtered))


def reference_fragment_centers(fragments_data, offset, voxel_size):
    """`scipy.ndimage.center_of_mass` over an explicit weight volume, alongside a
    separate `unique(return_counts=True)`. `ExtractFrags.fragment_centers`
    replaces this."""
    fragment_ids, counts = np.unique(fragments_data, return_counts=True)
    foreground = [(f, c) for f, c in zip(fragment_ids, counts) if f > 0]
    if not foreground:
        return {}
    fragment_ids, counts = zip(*foreground)
    centers = ndimage.center_of_mass(
        np.ones_like(fragments_data), fragments_data, fragment_ids
    )
    return {
        int(fragment_id): {
            "center": offset + voxel_size * Coordinate(center),
            "size": int(count),
        }
        for fragment_id, center, count in zip(fragment_ids, centers, counts)
    }


def reference_shift(affs_data, noise_eps, bias, rng):
    """A zeroed buffer plus a separately-allocated noise volume. The real code
    draws the noise straight into the buffer and folds the affinities in in
    place, holding two volumes where this holds four."""
    shift = np.zeros_like(affs_data)
    if noise_eps is not None:
        shift += rng.standard_normal(affs_data.shape, dtype=affs_data.dtype) * noise_eps
    shift += np.array([bias]).reshape((-1, *((1,) * (len(affs_data.shape) - 1))))
    return shift


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_extract_frags_init_and_drop(affs_2d, tmp_path):
    """init() creates both frags zarr and DB; drop_artifacts() removes them."""
    affs_path, _ = affs_2d
    frags_path = tmp_path / "test.zarr" / "frags"
    db = SQLite(path=tmp_path / "test.zarr" / "ef_db.sqlite", ndim=2)

    task = ExtractFrags(
        db=db,
        affs_data=Affs(store=affs_path),
        frags_data=Labels(store=frags_path),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        bias=[-0.5, -0.5],
    )
    task.init()
    assert frags_path.exists()
    assert db.path.exists()

    task.drop_artifacts()
    assert not frags_path.exists()
    assert not db.path.exists()


def test_extract_frags_basic(affs_2d, block_2d, tmp_path):
    """Affinities from 4-quadrant labels produce fragments and DB nodes.

    affs_2d is derived from labels_2d (4 regions in a 2x2 grid).
    Affinities are 0 at region boundaries and 1 within regions,
    so watershed should produce at least 2 distinct fragments.
    """
    affs_path, _ = affs_2d
    frags_path = tmp_path / "test.zarr" / "frags"
    db = SQLite(path=tmp_path / "test.zarr" / "ef_db.sqlite", ndim=2)

    task = ExtractFrags(
        db=db,
        affs_data=Affs(store=affs_path),
        frags_data=Labels(store=frags_path),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        bias=[-0.5, -0.5],
    )
    task.init()

    with task.process_block_func() as process_block:
        process_block(block_2d)

    frags = task.frags_data.array("r")[:]
    unique_ids = set(np.unique(frags)) - {0}
    assert len(unique_ids) >= 2, f"Expected >=2 fragments, got {unique_ids}"

    # Verify nodes were added to the DB with positions
    graph = db.open("r").read_graph()
    assert graph.number_of_nodes() >= 2
    for _, attrs in graph.nodes(data=True):
        assert "position" in attrs
        assert "size" in attrs


# ---------------------------------------------------------------------------
# bulk_write
#
# bulk_write=True is not just a faster INSERT. The block skips the
# `rag_provider[write_roi]` read and starts from a bare `nx.Graph()`, then writes
# via `bulk_write_graph` inside a `bulk_write_mode` context that drops and
# rebuilds the node position index. The speedup only shows at volumes far past
# unit-test scale, so what is pinned here is what must not drift: the rows that
# end up in the database.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block_size, context",
    [
        (Coordinate(20, 20), Coordinate(0, 0)),
        (Coordinate(10, 10), Coordinate(2, 2)),
        (Coordinate(5, 20), Coordinate(2, 0)),
    ],
    ids=["single_block", "four_blocks_with_context", "row_blocks"],
)
def test_extract_frags_bulk_matches_nonbulk(bulk, block_size, context):
    """Same fragments and the same node rows either way.

    Node attribution is unambiguous in both modes -- `bulk_write_nodes` still
    filters on the node position, and the centroids are computed inside write_roi
    -- so these must agree exactly, not merely as sets.
    """
    plain_db = bulk.db("plain")
    plain_frags = bulk.extract_frags(plain_db, "plain", False, block_size, context)

    bulk_db = bulk.db("bulk")
    bulk_frags = bulk.extract_frags(bulk_db, "bulk", True, block_size, context)

    np.testing.assert_array_equal(bulk_frags, plain_frags)

    plain_nodes = bulk.nodes(plain_db)
    assert len(plain_nodes) > 0, "no nodes written -- comparison would be vacuous"
    assert bulk.nodes(bulk_db) == plain_nodes


def test_extract_frags_bulk_writes_every_fragment(bulk):
    """Every non-zero fragment in the output volume gets a node row.

    Bulk mode starts from a bare graph instead of reading the RAG back, so a
    filter mistake would silently under-write nodes rather than raise.
    """
    db = bulk.db("coverage")
    frags = bulk.extract_frags(
        db, "coverage", True, Coordinate(10, 10), Coordinate(2, 2)
    )

    assert set(bulk.nodes(db)) == {int(i) for i in np.unique(frags) if i != 0}


def test_extract_frags_bulk_rebuilds_position_index(bulk):
    """Bulk mode drops the node position index and must put it back, or every
    later roi-restricted read degrades to a full table scan."""
    db = bulk.db("index")
    bulk.extract_frags(db, "index", True, Coordinate(10, 10), Coordinate(2, 2))

    indexes = {
        row[0]
        for row in db.open("r")
        .cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        .fetchall()
    }
    assert "pos_index" in indexes


# ---------------------------------------------------------------------------
# fragment_centers: bincount over a unique-inverse vs scipy center_of_mass
# ---------------------------------------------------------------------------


def test_fragment_centers_values(affs_2d, tmp_path):
    """Centroids and sizes of two hand-placed fragments, in world units."""
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path)

    fragments = np.zeros((10, 10), dtype=np.uint64)
    fragments[:5, :5] = 1  # centroid (2, 2), 25 voxels
    fragments[6:8, 2:4] = 7  # centroid (6.5, 2.5), 4 voxels

    centers = task.fragment_centers(fragments, Coordinate(0, 0), Coordinate(1, 1))

    assert set(centers) == {1, 7}
    assert centers[1] == {"center": Coordinate(2, 2), "size": 25}
    # Coordinate truncates, so the (6.5, 2.5) centroid lands on (6, 2).
    assert centers[7] == {"center": Coordinate(6, 2), "size": 4}


def test_fragment_centers_applies_offset_and_voxel_size(affs_2d, tmp_path):
    """Centroids are reported in world space, not voxel space."""
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path)

    fragments = np.zeros((10, 10), dtype=np.uint64)
    fragments[:5, :5] = 1  # voxel centroid (2, 2)

    centers = task.fragment_centers(fragments, Coordinate(100, 200), Coordinate(4, 8))

    assert centers[1]["center"] == Coordinate(100 + 4 * 2, 200 + 8 * 2)


def test_fragment_centers_excludes_background(affs_2d, tmp_path):
    """Label 0 is background and never gets a node, however much of the block
    it covers."""
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path)

    fragments = np.zeros((10, 10), dtype=np.uint64)
    fragments[0, 0] = 3

    centers = task.fragment_centers(fragments, Coordinate(0, 0), Coordinate(1, 1))

    assert set(centers) == {3}
    assert centers[3]["size"] == 1


def test_fragment_centers_empty(affs_2d, tmp_path):
    """An all-background block yields no centers rather than raising."""
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path)

    fragments = np.zeros((10, 10), dtype=np.uint64)

    assert task.fragment_centers(fragments, Coordinate(0, 0), Coordinate(1, 1)) == {}


@pytest.mark.parametrize(
    "layout",
    ["quadrants", "stripes", "sparse", "no_background", "random", "single"],
)
def test_fragment_centers_matches_scipy(affs_2d, tmp_path, layout):
    """The bincount centroids equal scipy's center_of_mass, layout for layout."""
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path)

    fragments = np.zeros((10, 10), dtype=np.uint64)
    if layout == "quadrants":
        fragments[:5, :5], fragments[:5, 5:] = 1, 2
        fragments[5:, :5], fragments[5:, 5:] = 3, 4
    elif layout == "stripes":
        fragments[:] = np.arange(1, 11, dtype=np.uint64)[:, None]
    elif layout == "sparse":
        fragments[1, 1], fragments[8, 3:6], fragments[4:9, 9] = 12, 40, 7
    elif layout == "no_background":
        fragments[:] = 5
        fragments[5:] = 7
    elif layout == "random":
        fragments[:] = np.random.default_rng(0).integers(0, 6, size=(10, 10))
    elif layout == "single":
        fragments[3:7, 3:7] = 99

    offset, voxel_size = Coordinate(30, 40), Coordinate(2, 3)
    assert task.fragment_centers(fragments, offset, voxel_size) == (
        reference_fragment_centers(fragments, offset, voxel_size)
    )


def test_fragment_centers_matches_scipy_3d(affs_2d, tmp_path):
    """The per-axis bincount loop has to hold in 3D, which is what the pipeline
    actually runs. Anisotropic voxels so a transposed axis would show up."""
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path)

    fragments = (
        np.random.default_rng(5).integers(0, 8, size=(6, 9, 12)).astype(np.uint64)
    )
    offset, voxel_size = Coordinate(10, 20, 30), Coordinate(40, 8, 8)

    assert task.fragment_centers(fragments, offset, voxel_size) == (
        reference_fragment_centers(fragments, offset, voxel_size)
    )


# ---------------------------------------------------------------------------
# filter_avg_fragments: bincount means vs scipy.ndimage.mean
# ---------------------------------------------------------------------------


def test_filter_avg_fragments_removes_low_affinity_fragments(affs_2d, tmp_path):
    """Fragments whose mean direct-neighbour affinity is below the threshold are
    zeroed; the rest are untouched."""
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path)

    fragments = np.zeros((10, 10), dtype=np.uint64)
    fragments[0:3], fragments[3:6], fragments[6:10] = 1, 2, 3
    affs = np.zeros((2, 10, 10), dtype=np.float32)
    affs[:, 0:3], affs[:, 3:6], affs[:, 6:10] = 0.1, 0.5, 0.9

    filtered = task.filter_avg_fragments(affs, fragments, 0.3)

    assert set(np.unique(filtered)) == {0, 2, 3}
    assert (filtered[0:3] == 0).all()
    assert (filtered[3:6] == 2).all()
    assert (filtered[6:10] == 3).all()


def test_filter_avg_fragments_no_op_leaves_block_unchanged(affs_2d, tmp_path):
    """With every mean above the threshold the block comes back as it went in -
    the early return skips a full-volume identity relabel."""
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path)

    fragments = np.zeros((10, 10), dtype=np.uint64)
    fragments[0:5], fragments[5:10] = 1, 2
    affs = np.full((2, 10, 10), 0.9, dtype=np.float32)

    filtered = task.filter_avg_fragments(affs, fragments, 0.5)

    np.testing.assert_array_equal(filtered, fragments)


def test_filter_avg_fragments_uses_only_direct_neighbour_offsets(tmp_path):
    """Only the first `ndim` channels - the direct-neighbour offsets are
    averaged. A high long-range channel must not rescue a low-affinity
    fragment."""
    affs_path = tmp_path / "test.zarr" / "affs_long_range"
    affs = np.zeros((3, 10, 10), dtype=np.float32)
    affs[0:2] = 0.1  # direct neighbours: mean 0.1
    affs[2] = 1.0  # long range: would drag the mean up to 0.4
    write_affs(affs_path, affs, [[1, 0], [0, 1], [4, 0]])

    task = make_task(affs_path, tmp_path, bias=[-0.5, -0.5, -0.5])
    fragments = np.ones((10, 10), dtype=np.uint64)

    filtered = task.filter_avg_fragments(affs, fragments, 0.25)

    assert (filtered == 0).all(), "long-range channel leaked into the mean"


@pytest.mark.parametrize("filter_value", [0.05, 0.15, 0.35, 0.55, 1.5])
def test_filter_avg_fragments_matches_scipy(affs_2d, tmp_path, filter_value):
    """The bincount means pick the same fragments as scipy.ndimage.mean.

    Affinities are keyed to the label value so each fragment's mean is exactly
    `label / 10`, keeping the comparison away from float ties at the threshold.
    """
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path)

    fragments = np.random.default_rng(1).integers(0, 6, size=(10, 10)).astype(np.uint64)
    affs = np.empty((2, 10, 10), dtype=np.float32)
    affs[:] = fragments / 10.0

    np.testing.assert_array_equal(
        task.filter_avg_fragments(affs, fragments, filter_value),
        reference_filter_avg_fragments(affs, fragments, filter_value, ndim=2),
    )


# ---------------------------------------------------------------------------
# compute_fragments: one in-place buffer vs separate noise/sum allocations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("noise_eps", [None, 0.01, 0.5])
def test_compute_fragments_matches_reference_shift(affs_2d, tmp_path, noise_eps):
    """Building the shift in place gives the same fragments as building it out of
    separate allocations, for the same noise draw."""
    affs_path, affs = affs_2d
    task = make_task(affs_path, tmp_path, noise_eps=noise_eps)

    fragments = task.compute_fragments(affs.copy(), rng=np.random.default_rng(7))

    shift = reference_shift(affs, noise_eps, task.bias, np.random.default_rng(7))
    expected = mws.agglom(
        (affs + shift).astype(np.float64),
        offsets=task.neighborhood,
        strides=None,
        seeds=None,
        randomized_strides=False,
    )

    np.testing.assert_array_equal(fragments, expected)


def test_compute_fragments_is_reproducible_for_a_seeded_rng(affs_2d, tmp_path):
    """Two calls with equally-seeded generators agree; `noise_eps` is the only
    source of randomness `compute_fragments` reaches."""
    affs_path, affs = affs_2d
    task = make_task(affs_path, tmp_path, noise_eps=0.5)

    first = task.compute_fragments(affs.copy(), rng=np.random.default_rng(0))
    second = task.compute_fragments(affs.copy(), rng=np.random.default_rng(0))
    other = task.compute_fragments(affs.copy(), rng=np.random.default_rng(1))

    np.testing.assert_array_equal(first, second)
    # A large noise_eps on binary affinities has to move at least one voxel,
    # otherwise the seed is not actually reaching the noise.
    assert not np.array_equal(first, other)


def test_compute_fragments_does_not_mutate_affs(affs_2d, tmp_path):
    """The shift is folded into the task's own buffer, not into the caller's
    affinities -- filter_avg_fragments still has to read them afterwards."""
    affs_path, affs = affs_2d
    task = make_task(affs_path, tmp_path, noise_eps=0.1)

    affs_data = affs.copy()
    task.compute_fragments(affs_data, rng=np.random.default_rng(3))

    np.testing.assert_array_equal(affs_data, affs)


# ---------------------------------------------------------------------------
# adaptive seeding
# ---------------------------------------------------------------------------


def touching_objects_affs(radii=(6, 3), separation=8):
    """Affinities over a big object plus two objects of different size whose
    shared boundary has collapsed - the case adaptive seeding exists for."""
    yy, xx = np.ogrid[:90, :90]

    def disk(cy, cx, r):
        return (yy - cy) ** 2 + (xx - cx) ** 2 < r * r

    big = disk(22, 25, 15)
    a = disk(65, 30, radii[0])
    b = disk(65, 30 + separation, radii[1])
    mask = big | a | b
    affs = np.stack([mask.astype(np.float64) * 0.9 + 0.05] * 4)
    return affs, big, a, b


def test_boundary_mask_offsets_selects_which_channels_are_averaged(affs_2d, tmp_path):
    """The mask follows `boundary_mask_offsets` and not the seeding rule, and a
    thin object survives only when the long range offsets are left out."""
    thin = np.zeros((40, 40), dtype=bool)
    thin[20:23, 5:35] = True  # thinner than the 8-voxel offsets
    affs = np.stack(
        [
            np.where(thin, 0.9, 0.05),
            np.where(thin, 0.9, 0.05),
            np.full(thin.shape, 0.05),
            np.full(thin.shape, 0.05),
        ]
    )
    affs_path = tmp_path / "test.zarr" / "lr_affs"
    write_affs(affs_path, affs.astype(np.float32), [[1, 0], [0, 1], [8, 0], [0, 8]])

    for kwargs in (dict(min_seed_distance=5), dict(min_seed_distance=5)):
        assert (
            make_task(affs_path, tmp_path, **kwargs).boundary_mask_offsets == "direct"
        )

        keeps = make_task(affs_path, tmp_path, **kwargs)
        drops = make_task(affs_path, tmp_path, boundary_mask_offsets="all", **kwargs)
        subset = make_task(affs_path, tmp_path, boundary_mask_offsets=[0, 1], **kwargs)

        assert keeps.boundary_mask(affs)[thin].all()
        assert not drops.boundary_mask(affs).any()
        assert subset.boundary_mask(affs)[thin].all()

    # no mask means no seed, so nothing forces a split anywhere in the thin object
    for task, seeded in ((keeps, True), (drops, False)):
        mask = task.boundary_mask(affs)
        seeds = np.asarray(task.get_seeds(ndimage.distance_transform_edt(mask), 5))
        seeds[~mask] = 0
        assert bool((seeds[thin] > 0).any()) is seeded


def test_boundary_mask_offsets_rejects_a_bad_selection(affs_2d, tmp_path):
    """An out-of-range or empty selection would otherwise surface as an IndexError
    or an all-False mask inside `compute_fragments`."""
    affs_path, _ = affs_2d
    for bad in ([0, 99], [], [-1]):
        with pytest.raises(ValueError, match="at least one offset"):
            make_task(affs_path, tmp_path, boundary_mask_offsets=bad)


def adaptive_task(affs_path, tmp_path, ratio):
    """A 2D task with a real context - the spacing cap is derived from it, so a
    zero context would collapse every spacing onto the noise floor."""
    return ExtractFrags(
        db=SQLite(path=tmp_path / "test.zarr" / "unit_db.sqlite", ndim=2),
        affs_data=Affs(store=affs_path),
        frags_data=Labels(store=tmp_path / "test.zarr" / "unit_frags"),
        block_size=Coordinate(60, 60),
        context=Coordinate(30, 30),
        bias=[-0.5, -0.5],
        adaptive_seed_spacing=ratio,
    )


def thin_and_thick():
    """A 3-voxel-wide bar beside a 25-voxel-wide blob, as a boundary distance."""
    mask = np.zeros((60, 60), dtype=bool)
    thin = np.zeros_like(mask)
    thin[8:11, 5:55] = True
    thick = np.zeros_like(mask)
    thick[25:50, 15:45] = True
    mask = thin | thick
    return ndimage.distance_transform_edt(mask), thin, thick


def test_adaptive_spacing_is_dense_in_thin_and_sparse_in_thick(affs_2d, tmp_path):
    """The whole point: one rule that seeds a thin process densely and a thick
    object sparsely, which a single footprint cannot do."""
    affs_path, _ = affs_2d
    dt, thin, thick = thin_and_thick()
    seeds = adaptive_task(affs_path, tmp_path, 1.5).get_adaptive_seeds(
        dt, Coordinate(1, 1)
    )
    density = lambda m: (seeds[m] > 0).sum() / m.sum()
    assert density(thin) > density(thick), (density(thin), density(thick))


def test_adaptive_spacing_ratio_is_the_density_knob(affs_2d, tmp_path):
    """Lower ratio -> finer everywhere. This is the tuning knob in adaptive mode."""
    affs_path, _ = affs_2d
    dt, _, _ = thin_and_thick()
    counts = [
        int(
            (
                adaptive_task(affs_path, tmp_path, r).get_adaptive_seeds(
                    dt, Coordinate(1, 1)
                )
                > 0
            ).sum()
        )
        for r in (1.0, 2.0, 4.0)
    ]
    assert counts[0] > counts[1] > counts[2], counts


def test_adaptive_spacing_seeds_a_process_along_its_length(affs_2d, tmp_path):
    """Candidates are ridge *voxels*, not one per connected component - along a
    smooth process the distance is flat, so a component would give one seed at
    any length."""
    affs_path, _ = affs_2d
    counts = []
    for length in (20, 120):
        mask = np.zeros((20, length + 10), dtype=bool)
        mask[8:11, 5 : 5 + length] = True
        seeds = adaptive_task(affs_path, tmp_path, 1.0).get_adaptive_seeds(
            ndimage.distance_transform_edt(mask), Coordinate(1, 1)
        )
        counts.append(int((seeds > 0).sum()))
    assert counts[1] > 3 * counts[0], counts


def test_adaptive_spacing_does_not_seed_below_resolution(affs_2d, tmp_path):
    """A structure thinner than a voxel gets no seed at all.

    Its boundary distance is a single voxel's worth and carries no thickness
    information, so clipping the spacing to the noise floor would pack seeds a
    few voxels apart down the wisp. Harmless while the affinities merged back
    across them, but `voronoi` enforces the seams, so the wisp shattered into
    fragments that `remove_debris` deleted - a gap where an object had been.
    """
    affs_path, _ = affs_2d
    task = adaptive_task(affs_path, tmp_path, 1.5)

    wisp = np.zeros((40, 60), dtype=bool)
    wisp[20, 5:55] = True  # one voxel thick
    seeds = task.get_adaptive_seeds(
        ndimage.distance_transform_edt(wisp), Coordinate(1, 1)
    )
    assert (seeds > 0).sum() == 0, int((seeds > 0).sum())

    # a resolvable process next to it must still be seeded along its length
    thick = np.zeros((40, 60), dtype=bool)
    thick[18:23, 5:55] = True
    seeds = task.get_adaptive_seeds(
        ndimage.distance_transform_edt(thick), Coordinate(1, 1)
    )
    assert (seeds > 0).sum() > 3, int((seeds > 0).sum())


def test_seeding_treats_the_volume_face_like_a_block_edge(affs_2d, tmp_path):
    """A read past the volume face is zero-filled, so the face reads as an
    object boundary: the distance transform distorts against it and the seeding
    (and the seam flood) misplace against the face rather than against tissue.
    With the mask continued across the clipped band, a process that truly
    continues past the face is seeded exactly as if the tissue were present."""
    affs_path, _ = affs_2d
    task = adaptive_task(affs_path, tmp_path, 1.5)

    band = 10
    mask = np.zeros((40, 80), dtype=bool)
    mask[14:25, :] = True  # a process crossing the whole read, band included
    clipped = mask.copy()
    clipped[:, :band] = False  # what a zero-filled read looks like

    fixed = task._continue_past_clipped_faces(
        clipped.copy(), (Coordinate(0, band), Coordinate(0, 0))
    )
    assert (fixed == mask).all()

    def seeds(m):
        return task.get_adaptive_seeds(
            ndimage.distance_transform_edt(m), Coordinate(1, 1)
        )

    # continuation restores ground-truth seeding; the zero-fill does not
    assert (seeds(fixed) == seeds(mask)).all()
    assert (seeds(clipped) != seeds(mask)).any()


def test_geodesic_flood_splits_a_flat_ridge_at_the_midpoint(affs_2d, tmp_path):
    """Along a smooth process the boundary distance is axially flat, so the
    flood has no preference and plain watershed breaks the tie by queue order:
    the first seed's front races the whole corridor and claims a medial tendril
    through the other's territory. The compactness tie-break settles it at the
    geodesic midpoint instead."""
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path, voronoi="geodesic")

    mask = np.zeros((20, 60), dtype=bool)
    mask[8:13, 2:58] = True  # a smooth process: flat ridge along axis 1
    bd = ndimage.distance_transform_edt(mask)
    seeds = np.zeros_like(mask, dtype=np.uint64)
    seeds[10, 5] = 1
    seeds[10, 54] = 2

    cells = task.seed_cells(seeds, bd, mask, Coordinate(1, 1))
    ridge = cells[10, 2:58]
    split = int(np.argmax(ridge == 2)) + 2  # first column owned by seed 2
    assert abs(split - 30) <= 3, split
    # and no tendril: each territory is one solid block of columns
    assert (ridge[: split - 2] == 1).all() and (ridge[split - 2 :] == 2).all()


def test_min_extent_removes_single_section_plates(affs_2d, tmp_path):
    """Low-certainty pockets come out of mws as stacks of fragments one section
    thick - confident in plane, so `filter_fragments` never sees them (the
    direct-offset average stays high). `min_extent` is the resolution statement
    that removes them: a fragment one voxel thick along an axis cannot
    represent a real structure."""
    affs_path, _ = affs_2d
    task = make_task(affs_path, tmp_path, min_extent=Coordinate(2, 1))

    frags = np.zeros((10, 12), dtype=np.uint64)
    frags[4, 2:10] = 7  # a plate: one section along axis 0, long along axis 1
    frags[6:9, 2:5] = 9  # genuine extent in both axes
    out = task.filter_min_extent(frags.copy())
    assert (out[frags == 7] == 0).all()
    assert (out[frags == 9] == 9).all()

    # one value per axis, validated
    with pytest.raises(ValueError, match="one value per axis"):
        make_task(affs_path, tmp_path, min_extent=Coordinate(2, 1, 1))


def test_seed_spacing_rules_are_mutually_exclusive(affs_2d, tmp_path):
    """They answer the same question, so the more restrictive one would just win."""
    affs_path, _ = affs_2d
    with pytest.raises(ValueError, match="one or the other"):
        make_task(affs_path, tmp_path, min_seed_distance=10, adaptive_seed_spacing=1.5)
