import daisy
import numpy as np
import pytest
from funlib.geometry import Roi

from volara.blockwise import GraphMWS
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
# A database that reads as arrays (CSR) is clustered as arrays. It has to
# give the partition the networkx path gives on the same graph in SQLite.
# ---------------------------------------------------------------------------

MWS_ROI = Roi((0, 0, 0), (100, 100, 100))


def random_graph(seed=0, n=300, m=1500):
    """Nodes 1..n at random positions; m edges with two scores, each present
    on about two thirds of the edges. Every score is distinct in magnitude,
    so the order of the edges is the same whatever order they are read in."""
    import networkx as nx

    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    for node in range(1, n + 1):
        position = tuple(float(p) for p in rng.uniform(0, 100, size=3))
        graph.add_node(node, position=position, size=int(rng.integers(1, 50)))
    magnitudes = iter(rng.permutation(np.linspace(0.01, 1.0, 2 * m)))
    while graph.number_of_edges() < m:
        u, v = (int(x) for x in rng.integers(1, n + 1, size=2))
        if u == v or graph.has_edge(u, v):
            continue
        attrs = {}
        for name in ("adj", "lr"):
            if rng.random() < 0.67:
                attrs[name] = float(next(magnitudes) * rng.choice([-1, 1]))
        graph.add_edge(u, v, **attrs)
    return graph


def both_dbs(tmp_path, graph):
    from volara.dbs import CSR, SQLite

    attrs = {"adj": "float", "lr": "float"}
    sqlite = SQLite(path=tmp_path / "rag.sqlite", edge_attrs=attrs)
    sqlite.init()
    sqlite.open("r+").write_graph(graph)
    csr = CSR(path=tmp_path / "rag.csr", edge_attrs=attrs, cell_size=20.0)
    csr.open("w").write_graph(graph)
    return sqlite, csr


def partition(task) -> set[frozenset[int]]:
    with task.process_block_func() as process_block:
        process_block(daisy.Block(MWS_ROI, MWS_ROI, MWS_ROI))
    lut = task.lut.load()
    assert lut is not None
    groups: dict[int, set[int]] = {}
    for fragment, segment in zip(*lut.tolist(), strict=True):
        groups.setdefault(segment, set()).add(fragment)
    return {frozenset(group) for group in groups.values()}


@pytest.mark.parametrize("edge_per_attr", [True, False])
@pytest.mark.parametrize("bounded_read", [True, False])
def test_graph_mws_arrays_match_networkx(tmp_path, edge_per_attr, bounded_read):
    graph = random_graph()
    sqlite, csr = both_dbs(tmp_path, graph)
    partitions = [
        partition(
            GraphMWS(
                roi=MWS_ROI,
                db=db,
                lut=LUT(path=tmp_path / f"{db.db_type}.npz"),
                weights={"adj": (1.0, -0.3), "lr": (2.0, -0.9)},
                edge_per_attr=edge_per_attr,
                bounded_read=bounded_read,
            )
        )
        for db in (sqlite, csr)
    ]
    assert partitions[0] == partitions[1]
    # a real clustering: neither one segment nor every fragment alone
    assert 1 < len(partitions[1]) < graph.number_of_nodes()


def test_graph_mws_arrays_keep_a_starting_lut(tmp_path):
    graph = random_graph(seed=1)
    sqlite, csr = both_dbs(tmp_path, graph)
    nodes = np.arange(1, graph.number_of_nodes() + 1)
    starting = LUT(path=tmp_path / "starting.npz")
    # shuffled, so the LUT is not in id order; groups of three are pre-merged
    order = np.random.default_rng(0).permutation(len(nodes))
    starting.save(np.array([nodes[order], ((nodes - 1) // 3 + 1)[order]]))
    partitions = [
        partition(
            GraphMWS(
                roi=MWS_ROI,
                db=db,
                lut=LUT(path=tmp_path / f"{db.db_type}.npz"),
                starting_lut=starting,
                weights={"adj": (1.0, -0.3), "lr": (2.0, -0.9)},
            )
        )
        for db in (sqlite, csr)
    ]
    assert partitions[0] == partitions[1]
    for group in partitions[1]:
        for node in group:
            triple = {3 * ((node - 1) // 3) + k for k in (1, 2, 3)}
            assert triple <= group


def test_graph_mws_arrays_need_every_node_in_the_starting_lut(tmp_path):
    graph = random_graph(seed=2, n=30, m=60)
    _, csr = both_dbs(tmp_path, graph)
    starting = LUT(path=tmp_path / "starting.npz")
    starting.save(np.array([[1, 2], [1, 1]]))
    task = GraphMWS(
        roi=MWS_ROI,
        db=csr,
        lut=LUT(path=tmp_path / "lut.npz"),
        starting_lut=starting,
        weights={"adj": (1.0, 0.0)},
    )
    with pytest.raises(KeyError, match="starting LUT"):
        partition(task)


def test_graph_mws_arrays_on_a_graph_without_edges(tmp_path):
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(1, position=(1.0, 1.0, 1.0), size=1)
    _, csr = both_dbs(tmp_path, graph)
    task = GraphMWS(
        roi=MWS_ROI,
        db=csr,
        lut=LUT(path=tmp_path / "lut.npz"),
        weights={"adj": (1.0, 0.0)},
    )
    assert partition(task) == set()
