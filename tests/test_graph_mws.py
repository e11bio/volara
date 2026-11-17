import numpy as np
import pytest
from funlib.geometry import Roi

from volara.blockwise import GraphMWS
from volara.blockwise.graph_mws import GraphMWSExtractFragments
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


def _three_fragment_db(tmp_path, edges) -> SQLite:
    """Fragments 1, 2 and 3 with the given ``(u, v, attrs)`` edges."""
    db = SQLite(
        path=tmp_path / "fragments.sqlite",
        node_attrs={"size": "int"},
        edge_attrs={"a": "float", "b": "float", "c": "float"},
        ndim=2,
    )
    db.init()
    provider = db.open("r+")
    graph = provider.read_graph()
    graph.add_node(1, position=(1, 1), size=1)
    graph.add_node(2, position=(1, 5), size=3)
    graph.add_node(3, position=(5, 5), size=4)
    for u, v, attrs in edges:
        graph.add_edge(u, v, **attrs)
    provider.write_graph(graph)
    return db


def _fragment_to_segment(
    task: GraphMWS | GraphMWSExtractFragments, block
) -> dict[int, int]:
    with task.process_block_func() as process_block:
        process_block(block)
    lut = task.lut.load()
    assert lut is not None
    fragments, segments = lut
    return dict(zip(fragments.tolist(), segments.tolist(), strict=True))


@pytest.mark.parametrize(
    "edge_per_attr, segments",
    [
        # one edge per attribute: the strongest, c = -0.6, splits first
        (True, {1: 1, 2: 2}),
        # one edge with the sum, 0.5 + 0.5 - 0.6 = 0.4: merge
        (False, {1: 1, 2: 1}),
    ],
)
def test_graph_mws_scores_each_attribute_as_an_edge_or_their_sum(
    block_2d, tmp_path, edge_per_attr, segments
):
    task = GraphMWS(
        roi=block_2d.read_roi,
        db=_three_fragment_db(tmp_path, [(1, 2, {"a": 0.5, "b": 0.5, "c": -0.6})]),
        lut=LUT(path=tmp_path / "lut.npz"),
        weights={"a": (1, 0), "b": (1, 0), "c": (1, 0)},
        edge_per_attr=edge_per_attr,
    )

    assert _fragment_to_segment(task, block_2d) == segments


def test_graph_mws_keeps_the_merges_of_the_starting_lut(block_2d, tmp_path):
    """Every edge says split; the starting LUT already merged fragments 1 and 2."""
    starting_lut = LUT(path=tmp_path / "starting.npz")
    starting_lut.save(np.array([[1, 2, 3], [1, 1, 3]]))
    task = GraphMWS(
        roi=block_2d.read_roi,
        db=_three_fragment_db(tmp_path, [(1, 2, {"a": -1.0}), (2, 3, {"a": -1.0})]),
        lut=LUT(path=tmp_path / "lut.npz"),
        starting_lut=starting_lut,
        weights={"a": (1, 0)},
    )

    assert _fragment_to_segment(task, block_2d) == {1: 1, 2: 1, 3: 3}


def test_graph_mws_extract_fragments_writes_a_size_weighted_node_per_segment(
    block_2d, tmp_path
):
    """Fragments 1 (size 1) and 2 (size 3) merge into segment 1; the 2-3 edge
    becomes the edge between segments 1 and 3."""
    segments_db = SQLite(
        path=tmp_path / "segments.sqlite",
        node_attrs={"size": "int"},
        edge_attrs={"a": "float"},
        ndim=2,
    )
    task = GraphMWSExtractFragments(
        fragments_db=_three_fragment_db(
            tmp_path, [(1, 2, {"a": 1.0}), (2, 3, {"a": -1.0})]
        ),
        segments_db=segments_db,
        lut=LUT(path=tmp_path / "lut.npz"),
        weights={"a": (1, 0)},
        roi=((0, 0), (10, 10)),
        block_size=(10, 10),
    )

    assert _fragment_to_segment(task, block_2d) == {1: 1, 2: 1, 3: 3}
    segments = segments_db.open("r").read_graph()
    assert {
        node: (attrs["position"], attrs["size"])
        for node, attrs in segments.nodes(data=True)
    } == {1: ((1.0, 4.0), 4), 3: ((5.0, 5.0), 4)}
    assert {(u, v): attrs["a"] for u, v, attrs in segments.edges(data=True)} == {
        (1, 3): -1.0
    }
