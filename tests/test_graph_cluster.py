"""End-to-end GraphCluster tests on synthetic SQLite RAGs: legacy
compatibility at num_levels=1, hierarchy invariants, determinism, restart,
roi policy, adversarial cross-block graphs, and multiprocessing."""

import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi

from volara.blockwise import GraphCluster, GraphMWS
from volara.blockwise.graph_cluster import supports_hierarchy
from volara.lut import LUT

from .graph_cluster_helpers import make_db, partition, random_rag

ROI = Roi((0, 0, 0), (80, 80, 80))
W = {"aff": (1.0, 0.0)}


def run(db, roi, levels, tmp_path, name=None, **kw):
    if levels > 1 and not supports_hierarchy(db.open("r")):
        pytest.skip("num_levels > 1 needs funlib.persistence with read_nodes(ids=)")
    name = name or f"levels_{levels}"
    task = GraphCluster(
        db=db,
        roi=roi,
        weights=kw.pop("weights", W),
        lut=tmp_path / f"{name}.npz",
        num_levels=levels,
        **kw,
    )
    task.drop()
    task.run_blockwise(multiprocessing=False)
    lut = task.lut.load()
    assert lut is not None
    return task, lut


def legacy(db, roi, tmp_path, name="legacy", **kw):
    task = GraphMWS(
        db=db,
        roi=roi,
        lut=LUT(path=tmp_path / f"{name}.npz"),
        weights=kw.pop("weights", W),
        **kw,
    )
    task.drop()
    task.run_blockwise(multiprocessing=False)
    lut = task.lut.load()
    assert lut is not None
    return lut


# ---------------------------------------------------------------------------
# num_levels=1 == legacy GraphMWS
# ---------------------------------------------------------------------------


def test_levels_one_matches_legacy_partition_with_ties(tmp_path):
    nodes, edges = random_rag(ROI, 300, 900, seed=0, ties=True)
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    _, lut = run(db, ROI, 1, tmp_path)
    assert partition(lut) == partition(
        legacy(db, ROI, tmp_path, canonical_order=True), nodes
    )
    # tie-free graph: the legacy default order gives the same partition too
    nodes, edges = random_rag(ROI, 300, 900, seed=1, ties=False)
    db2 = make_db(tmp_path / "rag2.sqlite", nodes, edges)
    _, lut2 = run(db2, ROI, 1, tmp_path, name="tiefree")
    assert partition(lut2) == partition(
        legacy(db2, ROI, tmp_path, name="legacy2"), nodes
    )


def test_levels_one_edge_per_attr_false_and_two_attributes(tmp_path):
    rng = np.random.default_rng(3)
    nodes, edges = random_rag(ROI, 200, 600, seed=2, ties=True)
    for _, _, a in edges:
        if rng.random() < 0.5:
            a["lr"] = round(float(rng.uniform(-1, 1)), 1)
        if rng.random() < 0.3:
            del a["aff"]
    db = make_db(tmp_path / "rag.sqlite", nodes, edges, edge_attrs=("aff", "lr"))
    weights = {"aff": (1.0, -0.2), "lr": (0.5, -0.3)}
    for edge_per_attr in (True, False):
        _, lut = run(
            db,
            ROI,
            1,
            tmp_path,
            name=f"epa{edge_per_attr}",
            weights=weights,
            edge_per_attr=edge_per_attr,
        )
        ref = legacy(
            db,
            ROI,
            tmp_path,
            name=f"legacy{edge_per_attr}",
            weights=weights,
            edge_per_attr=edge_per_attr,
            canonical_order=True,
        )
        assert partition(lut) == partition(ref, nodes)


def test_bounded_read_false_matches_legacy(tmp_path):
    nodes, edges = random_rag(ROI, 200, 600, seed=4)
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    _, lut = run(db, ROI, 1, tmp_path, bounded_read=False)
    assert partition(lut) == partition(
        legacy(db, ROI, tmp_path, bounded_read=False, canonical_order=True), nodes
    )
    with pytest.raises(ValueError, match="bounded_read"):
        GraphCluster(
            db=db,
            roi=ROI,
            weights=W,
            lut=tmp_path / "x.npz",
            num_levels=2,
            bounded_read=False,
        )


# ---------------------------------------------------------------------------
# hierarchy invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("levels", [2, 3, 4])
def test_hierarchy_invariants(tmp_path, levels):
    nodes, edges = random_rag(ROI, 400, 1400, seed=5, long_range=0.4)
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    task, lut = run(db, ROI, levels, tmp_path)

    # 1. every node exactly once, ids are member ids (min of the component)
    assert sorted(lut[0].tolist()) == sorted(nodes)
    for frag, seg in zip(lut[0], lut[1]):
        assert seg <= frag
    seg_of = dict(zip(lut[0].tolist(), lut[1].tolist()))
    for seg in set(seg_of.values()):
        assert seg_of[seg] == seg

    # 2. every signed edge introduced exactly once across the hierarchy, or
    #    dropped as provably irrelevant (repulsive, touching a final component)
    summary = task.summary()
    introduced = sum(l["n_edges_local"] + l["n_edges_new"] for l in summary["levels"])
    dropped_final = sum(l["n_edges_dropped_final"] for l in summary["levels"])
    assert introduced + dropped_final == len(edges)
    assert dropped_final > 0  # pruning is actually exercised on this graph
    assert summary["levels"][0]["n_blocks"] == task.octree.num_blocks(0)
    assert summary["levels"][-1]["n_blocks"] == 1
    assert sum(l["n_edges_dropped_outside"] for l in summary["levels"]) == 0

    # 3. each level's partition is a coarsening of the level below
    store = task.store
    pruned_total = 0
    for level in range(levels):
        in_ids, out_ids = store.level_lut(level)
        assert len(np.unique(in_ids)) == len(in_ids)
        comp_of = dict(zip(in_ids.tolist(), out_ids.tolist()))
        if level == 0:
            assert sorted(in_ids.tolist()) == sorted(nodes)
            final_of = {n: seg_of[n] for n in in_ids.tolist()}
        else:
            # inputs of this level are the outputs of the level below that can
            # still merge (components with a pending attractive edge); the rest
            # are final and keep their id through the composition
            prev_in, prev_out = store.level_lut(level - 1)
            assert set(in_ids.tolist()) <= set(prev_out.tolist())
            pruned_total += len(set(prev_out.tolist())) - len(in_ids)
            final_of = {c: seg_of[c] for c in in_ids.tolist()}  # comps are member ids
        by_comp = {}
        for n, c in comp_of.items():
            by_comp.setdefault(c, set()).add(final_of[n])
        assert all(
            len(s) == 1 for s in by_comp.values()
        ), "a parent split a child component"
    assert pruned_total > 0  # final components were left behind somewhere


def test_deterministic_and_restartable(tmp_path):
    nodes, edges = random_rag(ROI, 300, 1000, seed=6)
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    task, lut_a = run(db, ROI, 3, tmp_path)
    task, lut_b = run(db, ROI, 3, tmp_path)
    np.testing.assert_array_equal(lut_a, lut_b)

    # restart 1: all block-done caches gone, intermediate state intact -> every
    # block is skipped via its done marker and the result is identical
    task.drop(drop_outputs=False)
    task.lut.drop()
    task.run_blockwise(multiprocessing=False)
    np.testing.assert_array_equal(task.lut.load(), lut_a)

    # restart 2: the root block and one level-1 block lost their outputs
    store = task.store
    store.clear_block(2, (0, 0, 0))
    store.clear_block(1, (0, 0, 0))
    task.drop(drop_outputs=False)
    task.lut.drop()
    task.run_blockwise(multiprocessing=False)
    np.testing.assert_array_equal(task.lut.load(), lut_a)
    assert store.is_done(1, (0, 0, 0)) and store.is_done(2, (0, 0, 0))

    # a different configuration must not reuse the work dir silently
    other = GraphCluster(
        db=db, roi=ROI, weights={"aff": (1.0, 0.1)}, lut=task.lut, num_levels=3
    )
    with pytest.raises(RuntimeError, match="different configuration"):
        other.run_blockwise(multiprocessing=False)


def test_edges_to_nodes_outside_roi_are_excluded(tmp_path):
    inner = Roi((0, 0, 0), (40, 40, 40))
    nodes = {
        1: (5, 5, 5),
        2: (15, 5, 5),
        3: (35, 35, 35),
        99: (50, 50, 50),
        98: (60, 60, 60),
    }
    edges = [
        (1, 2, {"aff": 0.5}),
        (1, 99, {"aff": 1.0}),  # strong merge to an outside node: ignored
        (3, 98, {"aff": 1.0}),
        (98, 99, {"aff": 1.0}),
        (2, 3, {"aff": -0.9}),
    ]
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    for levels in (1, 2):
        task, lut = run(db, inner, levels, tmp_path)
        assert sorted(lut[0].tolist()) == [1, 2, 3]
        assert partition(lut) == frozenset({frozenset({1, 2}), frozenset({3})})
        dropped = sum(l["n_edges_dropped_outside"] for l in task.summary()["levels"])
        assert dropped == 2


def test_empty_leaves_isolated_nodes_and_large_ids(tmp_path):
    big = 2**40
    nodes = {
        big + 1: (1, 1, 1),
        big + 2: (2, 2, 2),
        big + 3: (3, 3, 3),
    }  # all in one corner leaf
    edges = [(big + 1, big + 2, {"aff": 0.7})]
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    task, lut = run(db, ROI, 3, tmp_path)
    assert partition(lut) == frozenset(
        {frozenset({big + 1, big + 2}), frozenset({big + 3})}
    )
    assert task.summary()["levels"][0]["n_blocks"] == 64  # empty leaves still complete


def test_starting_lut_only_with_one_level(tmp_path):
    nodes = {1: (5, 5, 5), 2: (6, 6, 6), 3: (7, 7, 7)}
    edges = [(1, 2, {"aff": -0.9}), (2, 3, {"aff": 0.5})]
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    starting = LUT(path=tmp_path / "starting.npz")
    starting.save(np.array([[1, 2, 3], [1, 1, 3]]))
    task, lut = run(db, ROI, 1, tmp_path, starting_lut=starting)
    assert partition(lut) == frozenset({frozenset({1, 2, 3})})
    with pytest.raises(ValueError, match="starting_lut"):
        GraphCluster(
            db=db,
            roi=ROI,
            weights=W,
            lut=tmp_path / "x.npz",
            num_levels=2,
            starting_lut=starting,
        )


# ---------------------------------------------------------------------------
# adversarial cross-block graphs
# ---------------------------------------------------------------------------

TWO = Roi(
    (0, 0, 0), (20, 10, 10)
)  # levels=2 -> leaf A: x in [0,10), leaf B: x in [10,20)
A = (2, 5, 5)
B = (12, 5, 5)


def test_a_strong_cross_block_attractive_edge_changes_a_merge(tmp_path):
    """Leaf A commits a 1-2 mutex (-0.5); stronger cross edges 1-3 (+0.9) and
    2-3 (+0.8) merge all three in the global run. Soft carry reproduces that;
    commit_child_mutexes keeps the child's mutex and blocks 2-3."""
    nodes = {1: A, 2: (3, 5, 5), 3: B}
    edges = [(1, 2, {"aff": -0.5}), (1, 3, {"aff": 0.9}), (2, 3, {"aff": 0.8})]
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    _, g = run(db, TWO, 1, tmp_path)
    assert partition(g) == frozenset({frozenset({1, 2, 3})})
    _, soft = run(db, TWO, 2, tmp_path, name="soft")
    assert partition(soft) == partition(g)
    _, hard = run(db, TWO, 2, tmp_path, name="hard", commit_child_mutexes=True)
    assert partition(hard) == frozenset({frozenset({1, 3}), frozenset({2})})


def test_b_cross_block_repulsive_edge_blocks_a_merge(tmp_path):
    """Node 1 in A; 2, 3 in B. Global: 1-2 mutex (-0.9), 1-3 merge (+0.8),
    2-3 blocked -> {1,3},{2}. Leaf B merges 2-3 (+0.7) before it can see the
    mutex, so the hierarchy ends with {1},{2,3}: the cross-block repulsive
    edge is honoured (1 never joins), and the early merge is the documented
    irreversible approximation."""
    nodes = {1: A, 2: B, 3: (14, 5, 5)}
    edges = [(1, 2, {"aff": -0.9}), (1, 3, {"aff": 0.8}), (2, 3, {"aff": 0.7})]
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    _, g = run(db, TWO, 1, tmp_path)
    assert partition(g) == frozenset({frozenset({1, 3}), frozenset({2})})
    _, h = run(db, TWO, 2, tmp_path, name="h")
    assert partition(h) == frozenset({frozenset({1}), frozenset({2, 3})})


def test_c_parallel_edges_collapse_onto_one_component_pair(tmp_path):
    nodes = {1: A, 2: (3, 5, 5), 3: B, 4: (14, 5, 5)}
    edges = [
        (1, 2, {"aff": 0.9}),
        (3, 4, {"aff": 0.9}),
        (1, 3, {"aff": -0.3}),
        (2, 4, {"aff": -0.6}),
        (1, 4, {"aff": 0.5}),
    ]
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    expected = frozenset({frozenset({1, 2}), frozenset({3, 4})})
    for levels in (1, 2):
        _, lut = run(db, TWO, levels, tmp_path)
        assert partition(lut) == expected
    # and with the strongest cross edge attractive they merge
    edges[3] = (2, 4, {"aff": -0.4})
    db = make_db(tmp_path / "rag2.sqlite", nodes, edges)
    for levels in (1, 2):
        _, lut = run(db, TWO, levels, tmp_path, name=f"m{levels}")
        assert partition(lut) == frozenset({frozenset({1, 2, 3, 4})})


def test_d_child_mutex_survives_two_contractions(tmp_path):
    """Three levels. Leaf A holds 1,2 with a -0.9 mutex. Node 3 sits in A's
    sibling (LCA level 1), node 4 far away (LCA level 2). 1-3 (+0.5) merges at
    level 1, so at level 2 the carried 1-2 mutex must be between components
    {1,3} and {2}; 3-4 (+0.7) then merges 4 in and 2-4 (+0.6) must be blocked.
    If the carried mutex were lost, 2 would join."""
    roi = Roi((0, 0, 0), (40, 10, 10))  # leaf 10 along x
    nodes = {1: (2, 5, 5), 2: (3, 5, 5), 3: (12, 5, 5), 4: (32, 5, 5)}
    edges = [
        (1, 2, {"aff": -0.9}),
        (1, 3, {"aff": 0.5}),
        (3, 4, {"aff": 0.7}),
        (2, 4, {"aff": 0.6}),
    ]
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    task, lut = run(db, roi, 3, tmp_path)
    assert partition(lut) == frozenset({frozenset({1, 3, 4}), frozenset({2})})
    # the level-1 block containing 1,2,3 carried exactly one edge: the mutex,
    # remapped onto components (1, 2)
    _, carried = task.store.read_graph(1, (0, 0, 0))
    assert carried.score.tolist() == [-0.9]
    assert sorted([int(carried.u[0]), int(carried.v[0])]) == [1, 2]


# ---------------------------------------------------------------------------
# multiprocessing
# ---------------------------------------------------------------------------


def test_multiprocessing_matches_serial(tmp_path):
    nodes, edges = random_rag(ROI, 300, 1000, seed=7)
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    _, serial = run(db, ROI, 3, tmp_path)
    task = GraphCluster(
        db=db, roi=ROI, weights=W, lut=tmp_path / "mp.npz", num_levels=3, num_workers=2
    )
    task.drop()
    task.run_blockwise(multiprocessing=True)
    np.testing.assert_array_equal(task.lut.load(), serial)


def test_intermediate_luts_are_successive_coarsenings(tmp_path):
    nodes, edges = random_rag(ROI, 300, 1000, seed=8)
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    task, lut = run(db, ROI, 3, tmp_path)
    np.testing.assert_array_equal(task.intermediate_lut(2), lut)
    prev = None
    for level in range(3):
        inter = task.intermediate_lut(level)
        assert sorted(inter[0].tolist()) == sorted(nodes)
        parts = partition(inter)
        if prev is not None:
            # every earlier component lies inside one later component
            later_of = {}
            for group in parts:
                for n in group:
                    later_of[n] = group
            assert all(len({later_of[n] for n in g}) == 1 for g in prev)
            assert len(parts) <= len(prev)
        prev = parts
    with pytest.raises(ValueError):
        task.intermediate_lut(3)


# ---------------------------------------------------------------------------
# cluster execution: worker configs per level, derived depth, measured memory
# ---------------------------------------------------------------------------


def test_local_worker_subprocesses_match_serial(tmp_path):
    """Dispatch through the worker CLI (the path Slurm/LSF workers use: the
    level task is serialised to JSON and rebuilt by `volara-cli
    blockwise-worker`), with a different worker config per level."""
    from volara.workers import LocalWorker

    nodes, edges = random_rag(ROI, 300, 1000, seed=9)
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    _, serial = run(db, ROI, 3, tmp_path)
    task = GraphCluster(
        db=db,
        roi=ROI,
        weights=W,
        lut=tmp_path / "workers.npz",
        num_levels=3,
        num_workers=[2, 2, 1],
        worker_config=[LocalWorker(), LocalWorker(), None],
    )
    task.drop()
    states = task.run_blockwise(multiprocessing=True)
    assert sum(s.failed_count for s in states.values()) == 0
    np.testing.assert_array_equal(task.lut.load(), serial)

    summary = task.summary()
    measured = [lvl["block_peak_rss_mb"] for lvl in summary["levels"]]
    assert all(m is not None and m["max"] > 0 for m in measured)
    assert measured[0]["n_blocks"] == task.octree.num_blocks(0)


def test_per_level_settings_must_match_depth(tmp_path):
    nodes, edges = random_rag(ROI, 50, 100, seed=10)
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    with pytest.raises(ValueError, match="num_workers has 2 entries"):
        GraphCluster(
            db=db,
            roi=ROI,
            weights=W,
            lut=tmp_path / "x.npz",
            num_levels=3,
            num_workers=[1, 2],
        )
    with pytest.raises(ValueError, match="worker_config has 1 entries"):
        GraphCluster(
            db=db,
            roi=ROI,
            weights=W,
            lut=tmp_path / "x.npz",
            num_levels=2,
            worker_config=[None],
        )


def test_num_levels_derived_from_block_shape(tmp_path):
    from volara.blockwise.graph_cluster import Octree

    nodes, edges = random_rag(ROI, 50, 100, seed=11)
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    task = GraphCluster(
        db=db,
        roi=ROI,
        weights=W,
        lut=tmp_path / "x.npz",
        num_levels=None,
        block_shape=Coordinate(20, 20, 20),
    )
    assert task.num_levels == 3  # 4 leaves per axis -> 2 -> 1
    assert Octree.levels_for(ROI, Coordinate(80, 80, 80)) == 1
    assert Octree.levels_for(ROI, Coordinate(10, 80, 80)) == 4
    assert Octree.levels_for(Roi((0, 0, 0), (100, 90, 70)), Coordinate(25, 23, 18)) == 3
    with pytest.raises(ValueError, match="block_shape"):
        GraphCluster(db=db, roi=ROI, weights=W, lut=tmp_path / "x.npz", num_levels=None)


def test_final_components_are_not_carried_upward(tmp_path):
    """Nodes 1,2 (leaf A) merge and have no attractive edge leaving A: they
    are final after the leaf and must not appear in the parent graph. Node 3
    (A) has an attractive edge to 4 (B) and is carried. A repulsive deferred
    edge 2-4 touches a final component and is dropped as irrelevant."""
    nodes = {1: A, 2: (3, 5, 5), 3: (4, 5, 5), 4: B}
    edges = [
        (1, 2, {"aff": 0.9}),
        (2, 3, {"aff": -0.5}),
        (3, 4, {"aff": 0.6}),
        (2, 4, {"aff": -0.8}),
    ]
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    task, lut = run(db, TWO, 2, tmp_path)
    assert partition(lut) == frozenset({frozenset({1, 2}), frozenset({3, 4})})
    _, g = run(db, TWO, 1, tmp_path, name="g")
    assert partition(g) == partition(lut)
    leaf_a = tuple(int(i) for i in task.octree.leaf_index(np.array([A]))[0])
    carried_nodes, carried_edges = task.store.read_graph(0, leaf_a)
    assert carried_nodes.tolist() == [3]
    assert len(carried_edges) == 0  # the 2-3 mutex touches final {1,2}: dropped
    root = task.store.read_stats(1, (0, 0, 0))
    assert root["n_edges_dropped_final"] == 1
    assert root["n_nodes"] == 2


def test_unimplemented_methods_fail_at_construction(tmp_path):
    nodes, edges = random_rag(ROI, 20, 30, seed=12)
    db = make_db(tmp_path / "rag.sqlite", nodes, edges)
    with pytest.raises(NotImplementedError, match="gasp"):
        GraphCluster(db=db, roi=ROI, weights=W, lut=tmp_path / "x.npz", method="gasp")
    with pytest.raises(NotImplementedError, match="hierarchical"):
        GraphCluster(
            db=db, roi=ROI, weights=W, lut=tmp_path / "x.npz", method="hierarchical"
        )
    with pytest.raises(ValueError):
        GraphCluster(
            db=db,
            roi=ROI,
            weights=W,
            lut=tmp_path / "x.npz",
            method="nope",  # ty: ignore[invalid-argument-type]
        )
