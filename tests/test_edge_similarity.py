"""Tests for EdgeSimilarity.

The graph is a 6x6x6 lattice of nodes with shuffled ids, cut into eight
planted clusters (the octants). Every node of a cluster has the same vector
and the vectors of different clusters are orthogonal, so the cosine of an
edge is 1 inside a cluster and 0 across two. Lattice edges carry an
``adj_weight``; a set of longer edges carries an ``lr_weight``, and some edges
carry both. The blocks (20 world units) do not line up with the clusters (30).
"""

import networkx as nx
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence.graphs import CSRGraphDataBase
from funlib.persistence.types import Vec

from volara.blockwise import EdgeSimilarity, GraphMWS, get_task
from volara.blockwise.edge_similarity import similarity
from volara.dbs import CSR, SQLite
from volara.lut import LUT
from volara.node_arrays import NodeArrays

SIDE = 6
ROI = Roi((0, 0, 0), (60, 60, 60))
SCORES = {"adj_similarity": "adj_weight", "lr_similarity": "lr_weight"}


class Lattice:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        grid = np.stack(
            np.meshgrid(*(np.arange(SIDE),) * 3, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        self.ids = rng.permutation(np.arange(1, len(grid) + 1)).astype(np.uint64)
        self.positions = grid * 10.0 + 5.0
        self.sizes = rng.integers(1, 100, size=len(grid))
        self.cluster = (grid // 3) @ np.array([4, 2, 1])
        self.vectors = np.eye(8, dtype=np.float32)[self.cluster] * 3.0

        index = {tuple(g): i for i, g in enumerate(grid)}
        adjacent = set()
        for g in grid:
            for axis in range(3):
                h = g.copy()
                h[axis] += 1
                if tuple(h) in index:
                    adjacent.add((index[tuple(g)], index[tuple(h)]))
        long_range = set()
        while len(long_range) < 80:
            a, b = rng.integers(0, len(grid), size=2)
            if a != b:
                long_range.add((min(a, b), max(a, b)))
        both = set(list(adjacent)[::9])
        long_range |= both
        self.edges = {}
        self.cosine: dict[tuple[int, int], float] = {}
        for a, b in sorted(adjacent | long_range):
            u, v = sorted((int(self.ids[a]), int(self.ids[b])))
            self.edges[(u, v)] = {
                "adj_weight": float(rng.random()) if (a, b) in adjacent else None,
                "lr_weight": float(rng.random()) if (a, b) in long_range else None,
            }
            self.cosine[(u, v)] = float(self.cluster[a] == self.cluster[b])

    def expected(self, transform=None):
        """{(u, v): {out_attr: score}} from the whole graph at once."""
        out = {}
        for pair, data in self.edges.items():
            score = self.cosine[pair]
            if transform == "atanh":
                score = float(np.arctanh(min(score, 1 - 1e-6)))
            out[pair] = {
                out_attr: score
                for out_attr, presence in SCORES.items()
                if data[presence] is not None
            }
        return out

    def write_bare_csr(self, path):
        """A store with no attribute but the ones the graph has: none of
        volara's defaults."""
        db = CSRGraphDataBase(
            path,
            mode="w",
            node_attrs={"position": Vec(float, 3), "size": int},
            edge_attrs={"adj_weight": "float32", "lr_weight": "float32"},
            cell_size=15.0,
        )
        u, v = (np.array(x, dtype=np.uint64) for x in zip(*self.edges, strict=True))
        with db.bulk_write_mode():
            db.write_node_arrays(self.ids, position=self.positions, size=self.sizes)
            db.write_edge_arrays(
                u,
                v,
                **{
                    name: np.array(
                        [
                            np.nan if d[name] is None else d[name]
                            for d in self.edges.values()
                        ]
                    )
                    for name in ("adj_weight", "lr_weight")
                },
            )
        return CSR(path=path)

    def write_sqlite(self, path):
        db = SQLite(path=path, edge_attrs={"adj_weight": "float", "lr_weight": "float"})
        db.init()
        graph = nx.Graph()
        for node, position, size in zip(
            self.ids, self.positions, self.sizes, strict=True
        ):
            graph.add_node(int(node), position=tuple(position), size=int(size))
        for (u, v), data in self.edges.items():
            graph.add_edge(
                u,
                v,
                **{
                    name: data[name]
                    for name in ("adj_weight", "lr_weight")
                    if data[name] is not None
                },
            )
        db.open("r+").write_graph(graph)
        return db

    def write_vectors(self, path, drop=()):
        keep = ~np.isin(self.ids, np.array(drop, dtype=np.uint64))
        nodes = NodeArrays(path=path)
        nodes.save(self.ids[keep], {"embedding": self.vectors[keep]})
        return nodes


@pytest.fixture()
def lattice():
    return Lattice()


def out_db(tmp_path, kind):
    attrs = {name: "float" for name in SCORES}
    if kind == "csr":
        return CSR(path=tmp_path / "scored.csr", edge_attrs=attrs, cell_size=15.0)
    return SQLite(path=tmp_path / "scored.sqlite", edge_attrs=attrs)


def run(task):
    state = task.run_blockwise(multiprocessing=False)[task.task_name]
    assert state.failed_count == 0
    assert state.orphaned_count == 0
    assert state.completed_count > 0
    return state


def stored_edges(db):
    graph = db.open("r").read_graph()
    return {
        (min(int(u), int(v)), max(int(u), int(v))): {
            name: data[name] for name in SCORES if data.get(name) is not None
        }
        for u, v, data in graph.edges(data=True)
    }


def assert_scores(got, want):
    assert set(got) == set(want)
    for pair, scores in want.items():
        assert set(got[pair]) == set(scores), pair
        for name, score in scores.items():
            assert got[pair][name] == pytest.approx(score, abs=1e-6), (pair, name)


@pytest.mark.parametrize("source", ["csr", "sqlite"])
@pytest.mark.parametrize("out", ["csr", "sqlite"])
def test_blocks_give_the_scores_of_the_whole_graph(tmp_path, lattice, source, out):
    db = (
        lattice.write_bare_csr(tmp_path / "rag.csr")
        if source == "csr"
        else lattice.write_sqlite(tmp_path / "rag.sqlite")
    )
    task = EdgeSimilarity(
        db=db,
        out_db=out_db(tmp_path, out),
        vectors=lattice.write_vectors(tmp_path / "vectors"),
        scores=SCORES,
        block_size=Coordinate(20, 20, 20),
        roi=ROI,
    )
    state = run(task)
    assert state.completed_count == 27
    assert_scores(stored_edges(task.out_db), lattice.expected())


def test_scored_graph_keeps_the_nodes(tmp_path, lattice):
    task = EdgeSimilarity(
        db=lattice.write_bare_csr(tmp_path / "rag.csr"),
        out_db=out_db(tmp_path, "csr"),
        vectors=lattice.write_vectors(tmp_path / "vectors"),
        scores=SCORES,
        block_size=Coordinate(20, 20, 20),
    )
    run(task)
    provider = task.out_db.open("r")
    assert isinstance(provider, CSRGraphDataBase)
    nodes, _ = provider.read_arrays(node_attrs=["position", "size"])
    order = np.argsort(nodes["id"])
    want = np.argsort(lattice.ids)
    np.testing.assert_array_equal(nodes["id"][order], lattice.ids[want])
    np.testing.assert_array_equal(nodes["position"][order], lattice.positions[want])
    np.testing.assert_array_equal(nodes["size"][order], lattice.sizes[want])


def test_roi_defaults_to_the_extent_of_a_csr_graph(tmp_path, lattice):
    task = EdgeSimilarity(
        db=lattice.write_bare_csr(tmp_path / "rag.csr"),
        out_db=out_db(tmp_path, "csr"),
        vectors=lattice.write_vectors(tmp_path / "vectors"),
        scores=SCORES,
        block_size=Coordinate(20, 20, 20),
    )
    assert task.write_roi == Roi((5, 5, 5), (51, 51, 51))


def test_roi_is_required_without_a_csr_graph(tmp_path, lattice):
    task = EdgeSimilarity(
        db=lattice.write_sqlite(tmp_path / "rag.sqlite"),
        out_db=out_db(tmp_path, "csr"),
        vectors=lattice.write_vectors(tmp_path / "vectors"),
        scores=SCORES,
        block_size=Coordinate(20, 20, 20),
    )
    with pytest.raises(ValueError, match="needs a roi"):
        _ = task.write_roi


@pytest.mark.parametrize("source", ["csr", "sqlite"])
def test_every_edge_belongs_to_exactly_one_block(tmp_path, lattice, source):
    db = (
        lattice.write_bare_csr(tmp_path / "rag.csr")
        if source == "csr"
        else lattice.write_sqlite(tmp_path / "rag.sqlite")
    )
    task = EdgeSimilarity(
        db=db,
        out_db=out_db(tmp_path, "csr"),
        vectors=lattice.write_vectors(tmp_path / "vectors"),
        scores=SCORES,
        block_size=Coordinate(20, 20, 20),
        roi=ROI,
    )
    provider = db.open("r")
    pairs, nodes, crossing = [], [], 0
    for z in range(0, 60, 20):
        for y in range(0, 60, 20):
            for x in range(0, 60, 20):
                block_nodes, edges = task.read_block(
                    provider, Roi((z, y, x), (20, 20, 20))
                )
                nodes.extend(block_nodes["id"].tolist())
                assert np.isin(edges["u"], block_nodes["id"]).all()
                crossing += int((~np.isin(edges["v"], block_nodes["id"])).sum())
                pairs.extend(zip(edges["u"].tolist(), edges["v"].tolist(), strict=True))
    assert sorted(nodes) == sorted(lattice.ids.tolist())
    assert crossing > 0
    assert len(pairs) == len(lattice.edges)
    assert set(pairs) == set(lattice.edges)


def test_similarity_then_graph_mws_recovers_the_planted_clusters(tmp_path, lattice):
    scored = out_db(tmp_path, "csr")
    run(
        EdgeSimilarity(
            db=lattice.write_bare_csr(tmp_path / "rag.csr"),
            out_db=scored,
            vectors=lattice.write_vectors(tmp_path / "vectors"),
            scores=SCORES,
            block_size=Coordinate(20, 20, 20),
            roi=ROI,
        )
    )
    mws = GraphMWS(
        db=scored,
        lut=LUT(path=tmp_path / "lut.npz"),
        weights={"adj_similarity": (1.0, -0.5), "lr_similarity": (1.0, -0.5)},
        roi=ROI,
    )
    run(mws)
    lut = mws.lut.load()
    assert lut is not None
    fragments, segments = lut
    assert sorted(fragments.tolist()) == sorted(lattice.ids.tolist())
    segment_of = dict(zip(fragments.tolist(), segments.tolist(), strict=True))
    found = {}
    for node, cluster in zip(
        lattice.ids.tolist(), lattice.cluster.tolist(), strict=True
    ):
        found.setdefault(segment_of[node], set()).add(cluster)
    assert len(found) == 8
    assert all(len(clusters) == 1 for clusters in found.values())


def test_an_edge_whose_end_has_no_vector_gets_no_score(tmp_path, lattice, caplog):
    dropped = int(lattice.ids[0])
    task = EdgeSimilarity(
        db=lattice.write_bare_csr(tmp_path / "rag.csr"),
        out_db=out_db(tmp_path, "csr"),
        vectors=lattice.write_vectors(tmp_path / "vectors", drop=[dropped]),
        scores=SCORES,
        block_size=Coordinate(60, 60, 60),
        roi=ROI,
    )
    with caplog.at_level("INFO"):
        run(task)
    want = lattice.expected()
    touching = [pair for pair in want if dropped in pair]
    assert len(touching) > 0
    for pair in touching:
        want[pair] = {}
    assert_scores(stored_edges(task.out_db), want)
    assert f"{len(touching)} without a score" in caplog.text


def test_atanh_spreads_the_cosines(tmp_path, lattice):
    task = EdgeSimilarity(
        db=lattice.write_bare_csr(tmp_path / "rag.csr"),
        out_db=out_db(tmp_path, "csr"),
        vectors=lattice.write_vectors(tmp_path / "vectors"),
        scores=SCORES,
        transform="atanh",
        block_size=Coordinate(20, 20, 20),
        roi=ROI,
    )
    run(task)
    assert_scores(stored_edges(task.out_db), lattice.expected(transform="atanh"))


def test_none_scores_every_edge(tmp_path, lattice):
    task = EdgeSimilarity(
        db=lattice.write_bare_csr(tmp_path / "rag.csr"),
        out_db=CSR(
            path=tmp_path / "scored.csr",
            edge_attrs={"similarity": "float"},
            cell_size=15.0,
        ),
        vectors=lattice.write_vectors(tmp_path / "vectors"),
        scores={"similarity": None},
        block_size=Coordinate(20, 20, 20),
        roi=ROI,
    )
    run(task)
    graph = task.out_db.open("r").read_graph()
    got = {
        (min(int(u), int(v)), max(int(u), int(v))): data["similarity"]
        for u, v, data in graph.edges(data=True)
    }
    assert got == lattice.cosine


def test_similarity_values():
    a = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    b = np.array([[3.0, 0.0], [0.0, 5.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(similarity(a, b), [1.0, 0.0, -1.0, np.nan, np.sqrt(0.5)])
    spread = similarity(a, b, transform="atanh", clip=1e-3)
    np.testing.assert_allclose(
        spread,
        [np.arctanh(0.999), 0.0, -np.arctanh(0.999), np.nan, np.arctanh(np.sqrt(0.5))],
    )


def test_scores_must_be_edge_attributes_of_the_output(tmp_path, lattice):
    with pytest.raises(ValueError, match="does not declare \\['lr_similarity'\\]"):
        EdgeSimilarity(
            db=CSR(path=tmp_path / "rag.csr"),
            out_db=CSR(
                path=tmp_path / "scored.csr", edge_attrs={"adj_similarity": "float"}
            ),
            vectors=NodeArrays(path=tmp_path / "vectors"),
            scores=SCORES,
            block_size=Coordinate(20, 20, 20),
        )


def test_drop_removes_the_scored_graph_and_keeps_the_source(tmp_path, lattice):
    task = EdgeSimilarity(
        db=lattice.write_bare_csr(tmp_path / "rag.csr"),
        out_db=out_db(tmp_path, "csr"),
        vectors=lattice.write_vectors(tmp_path / "vectors"),
        scores=SCORES,
        block_size=Coordinate(20, 20, 20),
        roi=ROI,
    )
    run(task)
    task.drop()
    assert isinstance(task.out_db, CSR) and isinstance(task.db, CSR)
    assert not task.out_db.local.exists()
    assert task.db.local.exists()
    assert task.vectors.exists()


def test_task_is_found_by_its_type():
    from volara.blockwise import discover_tasks

    discover_tasks()
    assert get_task("edge-similarity") is EdgeSimilarity


def test_a_block_of_empty_space_inside_the_graph_is_skipped(tmp_path):
    """Two nodes at the far corners: most blocks lie inside the extent of the
    graph and hold nothing."""
    source = CSRGraphDataBase(
        tmp_path / "rag.csr",
        mode="w",
        node_attrs={"position": Vec(float, 3)},
        edge_attrs={"adj_weight": "float32"},
        cell_size=10.0,
    )
    with source.bulk_write_mode():
        source.write_node_arrays(
            np.array([1, 2], dtype=np.uint64),
            position=np.array([[5.0, 5.0, 5.0], [95.0, 95.0, 95.0]]),
        )
        source.write_edge_arrays(np.array([1]), np.array([2]), adj_weight=np.array([1]))
    vectors = NodeArrays(path=tmp_path / "vectors")
    vectors.save(np.array([1, 2]), {"embedding": np.array([[1.0, 0.0], [1.0, 1.0]])})
    task = EdgeSimilarity(
        db=CSR(path=tmp_path / "rag.csr"),
        out_db=CSR(
            path=tmp_path / "scored.csr",
            edge_attrs={"adj_similarity": "float"},
            cell_size=10.0,
        ),
        vectors=vectors,
        scores={"adj_similarity": "adj_weight"},
        block_size=Coordinate(20, 20, 20),
    )
    state = run(task)
    assert state.completed_count == 125
    graph = task.out_db.open("r").read_graph()
    assert sorted(graph.nodes) == [1, 2]
    assert graph.edges[1, 2]["adj_similarity"] == pytest.approx(np.sqrt(0.5))
