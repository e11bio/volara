import sqlite3

import daisy
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
# optimized flag
#
# The optimized path must produce the same segmentation, not merely a similar
# one. Note it cannot produce the same LUT *array*: mws.cluster names each
# segment after an arbitrary member of the cluster and that choice varies
# between calls, so these compare the induced partition instead.
# ---------------------------------------------------------------------------


def _partition(lut):
    """Fragment groupings, ignoring which member names each group."""
    groups = {}
    for frag, seg in zip(lut[0], lut[1]):
        groups.setdefault(int(seg), set()).add(int(frag))
    return frozenset(frozenset(g) for g in groups.values())


@pytest.fixture()
def tied_scores_db(tmp_path):
    """A graph where the order tied edges arrive in changes the segmentation.

    Skipping networkx is only safe if the optimized path reproduces networkx's
    edge order, so the guard needs a graph that can actually tell the two orders
    apart. Three nodes: 1-2 and 2-3 both score +0.5, and a stronger mutex
    between 1 and 3 (-1.0) is processed first, so whichever tied edge comes next
    merges and blocks the other. Reading rows in DB order yields {1},{2,3};
    networkx's order yields {1,2},{3}.

    Rows are inserted directly because the insertion order is the whole point,
    and `write_graph` would reorder them.
    """
    db = SQLite(path=tmp_path / "tied.sqlite", edge_attrs={"adj_weight": "float"})
    db.open("w")
    con = sqlite3.connect(tmp_path / "tied.sqlite")
    con.executemany(
        "INSERT INTO nodes (id,position_0,position_1,position_2,size,filtered) "
        "VALUES (?,?,?,?,?,?)",
        [(i, i * 10.0, 50.0, 50.0, 100, 0) for i in (1, 2, 3)],
    )
    con.executemany(
        "INSERT INTO edges (u,v,distance,adj_weight) VALUES (?,?,?,?)",
        [(2, 3, 1.0, 0.75), (1, 2, 1.0, 0.75), (1, 3, 1.0, 0.0)],
    )
    con.commit()
    con.close()
    return db


# score = 2*adj - 1, so 0.75 -> +0.5 (twice, a tie) and 0.0 -> -1.0 (the mutex)
TIED_WEIGHTS = {"adj_weight": (2.0, -1.0)}
TIED_ROI = Roi((0, 0, 0), (100, 100, 100))


@pytest.mark.parametrize("bounded_read", [True, False])
@pytest.mark.parametrize("edge_per_attr", [True, False])
def test_graph_mws_optimized_matches_original(
    tied_scores_db, tmp_path, bounded_read, edge_per_attr
):
    """optimized=True gives the same fragment groupings as optimized=False."""
    roi = TIED_ROI
    block = daisy.Block(total_roi=roi, read_roi=roi, write_roi=roi)

    partitions = {}
    for optimized in (False, True):
        task = GraphMWS(
            db=tied_scores_db,
            lut=LUT(path=tmp_path / f"lut_{optimized}.npz"),
            roi=roi,
            weights=TIED_WEIGHTS,
            bounded_read=bounded_read,
            edge_per_attr=edge_per_attr,
            optimized=optimized,
        )
        with task.process_block_func() as process_block:
            process_block(block)
        loaded = task.lut.load()
        assert loaded is not None
        partitions[optimized] = _partition(loaded)

    assert partitions[False] == partitions[True]
    # the fixture is only a guard if the two possible orders really do differ
    assert partitions[False] == frozenset({frozenset({1, 2}), frozenset({3})})


def test_graph_mws_optimized_drops_edges_from_lut(tied_scores_db, tmp_path):
    """The optimized save omits the edge list nothing reads back."""
    roi = TIED_ROI
    block = daisy.Block(total_roi=roi, read_roi=roi, write_roi=roi)

    keys = {}
    for optimized in (False, True):
        task = GraphMWS(
            db=tied_scores_db,
            lut=LUT(path=tmp_path / f"lut_{optimized}.npz"),
            roi=roi,
            weights=TIED_WEIGHTS,
            optimized=optimized,
        )
        with task.process_block_func() as process_block:
            process_block(block)
        keys[optimized] = set(np.load(task.lut.file).keys())

    assert "edges" in keys[False]
    assert keys[True] == {"fragment_segment_lut"}


def test_graph_mws_optimized_still_reads_nodes_for_starting_lut(
    tied_scores_db, tmp_path
):
    """starting_lut needs graph.nodes, so that path keeps the graph read."""
    roi = TIED_ROI
    block = daisy.Block(total_roi=roi, read_roi=roi, write_roi=roi)

    starting = LUT(path=tmp_path / "starting.npz")
    # force 1 and 3 together, which the -1.0 mutex between them would not
    starting.save(np.array([[1, 2, 3], [1, 2, 1]]))

    task = GraphMWS(
        db=tied_scores_db,
        lut=LUT(path=tmp_path / "lut.npz"),
        roi=roi,
        weights=TIED_WEIGHTS,
        starting_lut=starting,
        optimized=True,
    )
    with task.process_block_func() as process_block:
        process_block(block)

    loaded = task.lut.load()
    assert loaded is not None
    groups = _partition(loaded)
    assert any({1, 3} <= g for g in groups), "starting_lut merge was not preserved"


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
