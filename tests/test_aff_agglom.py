import sqlite3

import numpy as np
import pytest
from funlib.geometry import Coordinate
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import AffAgglom
from volara.datasets import Affs, Labels


def test_aff_agglom_drop_edges(frags_2d, sqlite_db_2d, block_2d, tmp_path):
    """drop_artifacts() removes edges but preserves nodes."""
    frags_path, _ = frags_2d

    # Create 1-channel affs (needed for the task config, not for the drop test)
    affs_data = np.zeros((1, 10, 10), dtype=np.float32)
    affs_path = tmp_path / "test.zarr" / "affs"
    arr = prepare_ds(
        affs_path,
        shape=affs_data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=affs_data.dtype,
        mode="w",
    )
    arr[:] = affs_data
    arr._source_data.attrs["neighborhood"] = [[1, 0]]

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

    task = AffAgglom(
        db=sqlite_db_2d,
        frags_data=Labels(store=frags_path),
        affs_data=Affs(store=affs_path),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        scores={"y_aff": [Coordinate(1, 0)]},
    )
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
    affs_path = tmp_path / "test.zarr" / "affs"
    arr = prepare_ds(
        affs_path,
        shape=affs_data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=affs_data.dtype,
        mode="w",
    )
    arr[:] = affs_data
    arr._source_data.attrs["neighborhood"] = [[1, 0]]

    task = AffAgglom(
        db=sqlite_db_2d,
        frags_data=Labels(store=frags_path),
        affs_data=Affs(store=affs_path),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        scores={"y_aff": [Coordinate(1, 0)]},
    )

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
# bulk_write
#
# bulk_write=True changes three things about the block, not just the INSERT:
#   - it skips the `rag_provider[read_roi]` read and starts from a bare
#     `nx.Graph()`, so the graph has no node attributes at all;
#   - because of that, edges cannot be filtered on node position the way
#     `write_graph` does, so it substitutes its own filter: emit edge (u, v) if
#     min(u, v) has voxels in block.write_roi;
#   - writes go through `bulk_write_edges` inside a `bulk_write_mode` context.
#
# The tests run extract-frags first, both because aff-agglom needs the nodes it
# writes and because on SQLite its bulk context is what switches the file to WAL.
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
def test_aff_agglom_bulk_matches_nonbulk(bulk, block_size, context):
    """Same edges and the same weights either way.

    `ExtractFrags` relabels each block independently and bumps ids by block id, so
    every fragment lives in exactly one write_roi: the `min(u, v) in home_ids`
    filter is an exact partition and each edge is emitted by exactly one block.
    Weights must therefore match exactly, not just the edge set.
    """
    plain_db, _ = bulk.pipeline("plain", False, block_size, context)
    bulk_db, _ = bulk.pipeline("bulk", True, block_size, context)

    plain_edges, bulk_edges = bulk.edges(plain_db), bulk.edges(bulk_db)

    assert len(plain_edges) > 0, "no edges written - comparison would be empty"
    assert set(bulk_edges) == set(plain_edges)
    for pair, data in plain_edges.items():
        assert bulk_edges[pair]["aff"] == pytest.approx(data["aff"]), pair


@pytest.mark.parametrize(
    "extract_block, agglom_block",
    [(Coordinate(10, 10), Coordinate(5, 5)), (Coordinate(5, 5), Coordinate(10, 10))],
    ids=["agglom_blocks_smaller", "agglom_blocks_larger"],
)
def test_aff_agglom_bulk_matches_nonbulk_with_mismatched_block_sizes(
    bulk, extract_block, agglom_block
):
    """AffAgglom tiled differently from ExtractFrags.

    This is the one configuration where a fragment really can have voxels in
    several agglom blocks, so the same edge is emitted more than once and
    INSERT OR IGNORE picks whichever block committed first. The result still has
    to match the non-bulk run.
    """
    results = {}
    for tag, use_bulk in (("plain", False), ("bulk", True)):
        db = bulk.db(tag)
        bulk.extract_frags(db, tag, use_bulk, extract_block, Coordinate(2, 2))
        bulk.aff_agglom(db, tag, use_bulk, agglom_block, Coordinate(2, 2))
        results[tag] = bulk.edges(db)

    assert len(results["plain"]) > 0
    assert set(results["bulk"]) == set(results["plain"])
    for pair, data in results["plain"].items():
        assert results["bulk"][pair]["aff"] == pytest.approx(data["aff"]), pair


def test_aff_agglom_bulk_finds_every_adjacency(bulk):
    """The bulk filter keys on fragment id rather than node position. Check that
    nothing is dropped: every adjacent pair of fragments in the volume gets an
    edge."""
    db, frags = bulk.pipeline("adjacency", True, Coordinate(10, 10), Coordinate(2, 2))

    expected = set()
    for axis in (0, 1):
        base = np.take(frags, range(frags.shape[axis] - 1), axis=axis)
        shifted = np.take(frags, range(1, frags.shape[axis]), axis=axis)
        mask = (base != shifted) & (base > 0) & (shifted > 0)
        for u, v in zip(base[mask], shifted[mask]):
            expected.add((min(int(u), int(v)), max(int(u), int(v))))

    assert set(bulk.edges(db)) == expected


def test_aff_agglom_bulk_preserves_node_rows(bulk):
    """AffAgglom writes edges only - it enters `bulk_write_mode` with
    node_writes=False, so the node rows ExtractFrags wrote must survive intact."""
    db = bulk.db("nodes")
    bulk.extract_frags(db, "nodes", True, Coordinate(10, 10), Coordinate(2, 2))
    before = bulk.nodes(db)

    bulk.aff_agglom(db, "nodes", True, Coordinate(10, 10), Coordinate(2, 2))

    assert bulk.nodes(db) == before


@pytest.mark.xfail(
    strict=True,
    raises=sqlite3.OperationalError,
    reason=(
        "bulk_write on SQLite needs the file already in WAL mode. Running "
        "aff-agglom with bulk_write=True as the first bulk task against a DB that "
        "already holds rows fails: bulk_write_mode's rollback -> WAL "
        "switch cannot take its exclusive lock while another connection is open, "
        "and SQLite.open() never closes one. The normal pipeline is fine because a "
        "bulk extract-frags switches the file first, while it is still empty."
    ),
)
def test_aff_agglom_bulk_alone_on_a_non_wal_db(bulk):
    """Re-running only aff-agglom in bulk mode over a DB from a non-bulk run.

    Expected to fail today. Written strict so it turns into a failure the day it
    is fixed, rather than being rediscovered.
    """
    db = bulk.db("nonwal")
    bulk.extract_frags(db, "nonwal", False, Coordinate(10, 10), Coordinate(2, 2))
    assert (
        sqlite3.connect(db.path).execute("PRAGMA journal_mode").fetchone()[0]
        == "delete"
    )

    bulk.aff_agglom(db, "nonwal", True, Coordinate(10, 10), Coordinate(2, 2))
