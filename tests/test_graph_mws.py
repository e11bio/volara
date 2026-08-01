import sqlite3

import networkx as nx
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi

from volara.blockwise import GraphMWS, IterativeGraphMWS
from volara.dbs import SQLite
from volara.lut import LUT


def test_graph_mws_drop(sqlite_db_2d, tmp_path):
    """drop_artifacts() removes the saved LUT file."""
    lut = LUT(path=tmp_path / "lut.npz")
    lut.save(np.array([[1], [2]]))
    assert lut.file.exists()

    config = GraphMWS(
        roi=Roi((0, 0), (10, 10)),
        db=sqlite_db_2d,
        lut=lut,
        weights={"y_aff": (1, 0)},
    )
    config.drop_artifacts()
    assert not lut.file.exists()


@pytest.mark.parametrize("y_bias", [0.5, -0.5])
def test_graph_mws_merge_split(sqlite_db_2d, block_2d, tmp_path, y_bias):
    """Positive bias merges 2 nodes into 1 segment; negative bias keeps them separate."""
    # Seed DB with 2 nodes connected by an edge with y_aff=0
    db = sqlite_db_2d.open("r+")
    graph = db.read_graph()
    graph.add_node(1, position=(4, 2), size=600, raw_intensity=(0.1,))
    graph.add_node(2, position=(4, 7), size=400, raw_intensity=(0.1,))
    graph.add_edge(1, 2, y_aff=0)
    db.write_graph(graph)

    config = GraphMWS(
        roi=block_2d.read_roi,
        db=sqlite_db_2d,
        lut=LUT(path=tmp_path / "fragment_segment_lut.npz"),
        weights={"y_aff": (1, y_bias)},
    )

    with config.process_block_func() as process_block:
        process_block(block_2d)

    lut = config.lut.load()
    assert lut is not None
    fragments, segments = lut
    assert len(np.unique(fragments)) == 2
    # score = 1*0 + bias. Positive bias -> positive edge -> merge (1 seg).
    # Negative bias -> negative edge -> split (2 segs).
    assert len(np.unique(segments)) == 1 + (y_bias < 0)


# ---------------------------------------------------------------------------
# IterativeGraphMWS
#
# One round of a recursive coarsening: each block clusters its own region and
# writes the resulting super-fragments (plus size-weighted agglomerated edges)
# into `segments_db`, emitting a per-round LUT.
#
# Segment NODES are owned by the block whose write_roi holds their position, but
# segment EDGES are not: `write_edges` filters on min(u, v)'s position, so an
# edge between segments owned by two different blocks would be dropped by both.
# `write_graph(..., both_sides=True)` used to cover this and no longer exists;
# writing edges unfiltered is the replacement. `test_..._cross_block_edge` is
# the regression guard for that - it is the assertion that fails if the edge
# write is narrowed back to write_roi.
# ---------------------------------------------------------------------------


@pytest.fixture()
def chain_frags_db(tmp_path):
    """4 fragments in a line along z, spanning two 200-unit blocks.

    Strong affinity within each block, weak across the block boundary, so the
    expected outcome is two segments (1+2, 3+4) joined by one cross-block edge.
    """
    db = SQLite(
        path=tmp_path / "frags.sqlite",
        edge_attrs={"adj_weight": "float", "adj_weight__size": "float"},
    )
    provider = db.open("w")
    graph = nx.Graph()
    for node, z in [(1, 50), (2, 150), (3, 250), (4, 350)]:
        graph.add_node(node, position=(z, 50, 50), size=100, filtered=False)
    for u, v, weight in [(1, 2, 0.9), (2, 3, 0.2), (3, 4, 0.9)]:
        graph.add_edge(u, v, adj_weight=weight, adj_weight__size=10.0, distance=100.0)
    provider.write_graph(graph)
    return db


def _run_iterative(frags_db, tmp_path):
    seg_db = SQLite(
        path=tmp_path / "segs.sqlite",
        edge_attrs={"adj_weight": "float", "adj_weight__size": "float"},
    )
    task = IterativeGraphMWS(
        fragments_db=frags_db,
        segments_db=seg_db,
        lut=LUT(path=tmp_path / "lut.npz"),
        weights={"adj_weight": (1.0, -0.4)},
        roi=(Coordinate(0, 0, 0), Coordinate(400, 100, 100)),
        block_size=Coordinate(200, 100, 100),
    )
    task.init()
    task.run_blockwise(multiprocessing=False)
    return task, seg_db


def test_iterative_graph_mws_runs_and_writes_segments(chain_frags_db, tmp_path):
    """Every fragment lands in the LUT and super-fragment nodes are persisted."""
    task, seg_db = _run_iterative(chain_frags_db, tmp_path)

    lut = task.lut.load()
    assert lut is not None
    assert sorted(int(f) for f in lut[0]) == [1, 2, 3, 4]

    con = sqlite3.connect(seg_db.path)
    nodes = con.execute("SELECT id, size FROM nodes ORDER BY id").fetchall()
    con.close()
    # 1+2 and 3+4 merge; the 0.2 edge scores 0.2-0.4 < 0 and splits.
    assert len(nodes) == 2
    assert [size for _, size in nodes] == [200, 200]


def test_iterative_graph_mws_writes_cross_block_edge(chain_frags_db, tmp_path):
    """The edge between two differently-owned segments must survive the write.

    Regression guard for the removal of `write_graph(..., both_sides=True)`:
    filtering edge writes by write_roi drops this edge entirely.
    """
    _, seg_db = _run_iterative(chain_frags_db, tmp_path)

    con = sqlite3.connect(seg_db.path)
    edges = con.execute("SELECT u, v, adj_weight FROM edges").fetchall()
    segment_ids = {row[0] for row in con.execute("SELECT id FROM nodes")}
    con.close()

    # Deliberately not asserting *which* ids: scores here are 0.5, 0.5, -0.2, and
    # the tie between the two 0.5s means the cluster representative depends on
    # edge read order, which in turn depends on block execution order (the
    # neighbour-LUT reads make blocks order-dependent). What must hold is that
    # the edge survives the write at all - it is zero-length without the
    # unfiltered `write_edges`.
    assert edges, "cross-block segment edge was dropped by the write"
    for u, v, weight in edges:
        assert u in segment_ids and v in segment_ids
        assert u != v
        assert weight == pytest.approx(0.2)
