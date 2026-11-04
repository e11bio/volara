from funlib.geometry import Coordinate

from volara.blockwise import DistanceAgglom
from volara.datasets import Labels
from volara.dbs import SQLite


def _make_distance_agglom(labels_path, tmp_path):
    """Helper: DistanceAgglom task with a 4-node DB (one per labels_2d quadrant)."""
    db = SQLite(
        path=tmp_path / "test.zarr" / "da_db.sqlite",
        node_attrs={"embedding": 3},
        edge_attrs={"distance": "float", "embedding_similarity": "float"},
        ndim=2,
    )
    db.init()

    # One node per quadrant, each with a distinct 3D embedding
    graph_provider = db.open("r+")
    g = graph_provider.read_graph()
    g.add_node(1, position=(2, 2), size=25, embedding=(1.0, 0.0, 0.0))
    g.add_node(2, position=(2, 7), size=25, embedding=(0.9, 0.1, 0.0))
    g.add_node(3, position=(7, 2), size=25, embedding=(0.0, 1.0, 0.0))
    g.add_node(4, position=(7, 7), size=25, embedding=(0.0, 0.0, 1.0))
    graph_provider.write_graph(g)

    task = DistanceAgglom(
        storage=db,
        frags_data=Labels(store=labels_path),
        distance_keys=["embedding"],
        block_size=Coordinate(10, 10),
        context=Coordinate(2, 2),
    )
    return task, db


def test_distance_agglom_drop(labels_2d, block_2d, tmp_path):
    """drop_artifacts() removes edges but preserves nodes."""
    labels_path, _ = labels_2d
    task, db = _make_distance_agglom(labels_path, tmp_path)

    # Create edges first
    with task.process_block_func() as process_block:
        process_block(block_2d)

    g = db.open("r").read_graph()
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() >= 1

    task.drop_artifacts()

    g = db.open("r").read_graph()
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 0


def test_distance_agglom_basic(labels_2d, block_2d, tmp_path):
    """4 quadrant fragments produce edges with distance and embedding_similarity."""
    labels_path, _ = labels_2d
    task, db = _make_distance_agglom(labels_path, tmp_path)

    with task.process_block_func() as process_block:
        process_block(block_2d)

    g = db.open("r").read_graph()
    assert g.number_of_edges() >= 1
    for _u, _v, data in g.edges(data=True):
        assert "distance" in data
        assert "embedding_similarity" in data
        assert data["distance"] >= 0


def test_distance_agglom_similar_embeddings(labels_2d, block_2d, tmp_path):
    """Fragments with similar embeddings have higher cosine similarity than dissimilar ones."""
    labels_path, _ = labels_2d
    db = SQLite(
        path=tmp_path / "test.zarr" / "da_db2.sqlite",
        node_attrs={"embedding": 3},
        edge_attrs={"distance": "float", "embedding_similarity": "float"},
        ndim=2,
    )
    db.init()

    # Nodes 1 and 2 have nearly identical embeddings; 3 and 4 are orthogonal
    graph_provider = db.open("r+")
    g = graph_provider.read_graph()
    g.add_node(1, position=(2, 2), size=25, embedding=(1.0, 0.0, 0.0))
    g.add_node(2, position=(2, 7), size=25, embedding=(0.99, 0.01, 0.0))
    g.add_node(3, position=(7, 2), size=25, embedding=(0.0, 1.0, 0.0))
    g.add_node(4, position=(7, 7), size=25, embedding=(0.0, 0.0, 1.0))
    graph_provider.write_graph(g)

    task = DistanceAgglom(
        storage=db,
        frags_data=Labels(store=labels_path),
        distance_keys=["embedding"],
        block_size=Coordinate(10, 10),
        context=Coordinate(2, 2),
    )

    with task.process_block_func() as process_block:
        process_block(block_2d)

    g = db.open("r").read_graph()
    similarities = {}
    for u, v, data in g.edges(data=True):
        similarities[(u, v)] = data["embedding_similarity"]

    # 1-2 should have high similarity (nearly parallel embeddings)
    if (1, 2) in similarities:
        assert similarities[(1, 2)] > 0.9
    elif (2, 1) in similarities:
        assert similarities[(2, 1)] > 0.9

    # 3-4 should have low similarity (orthogonal embeddings)
    if (3, 4) in similarities:
        assert similarities[(3, 4)] < 0.1
    elif (4, 3) in similarities:
        assert similarities[(4, 3)] < 0.1
