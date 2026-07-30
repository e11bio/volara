import numpy as np
import pytest
from funlib.geometry import Coordinate

from volara.blockwise import ExtractFrags
from volara.datasets import Affs, Labels
from volara.dbs import SQLite


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
