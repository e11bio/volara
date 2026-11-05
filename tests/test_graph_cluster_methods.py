"""Method-level tests: score preparation, canonical order, min-id labels,
contraction, and the derived-mutex-state argument checked against a
brute-force mutex watershed."""

import mwatershed as mws
import numpy as np
import pytest

from volara.blockwise.graph_cluster import MutexWatershed, SignedEdges

from .graph_cluster_helpers import groups_of, reference_mws


def test_backend_matches_reference_mws():
    """The Rust backend uses the cluster-level mutex semantics the design relies on."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 40
        edges = SignedEdges(
            rng.uniform(-1, 1, 200).round(1),
            rng.integers(1, n + 1, 200),
            rng.integers(1, n + 1, 200),
            np.zeros(200, dtype=np.int64),
        )
        edges = edges.take(edges.u != edges.v).sorted()
        pairs = dict(mws.cluster(edges.to_mws()))
        ref = reference_mws(edges.to_mws())
        assert groups_of(pairs) == groups_of(ref)


def test_prepare_applies_weights_once_per_attribute():
    m = MutexWatershed({"a": (2.0, -1.0), "b": (1.0, 0.5)})
    attrs = np.array([[0.75, np.nan], [np.nan, 0.0], [0.0, -1.0]])
    signed, src = m.prepare([1, 2, 3], [2, 3, 1], attrs)
    assert src.tolist() == [0, 1, 2, 2]
    assert signed.score.tolist() == [0.5, 0.5, -1.0, -0.5]
    assert signed.attr.tolist() == [0, 1, 0, 1]

    summed = MutexWatershed({"a": (2.0, -1.0), "b": (1.0, 0.5)}, edge_per_attr=False)
    signed, src = summed.prepare(
        [1, 2, 3, 4], [2, 3, 1, 1], np.vstack([attrs, [[np.nan, np.nan]]])
    )
    assert signed.score.tolist() == [0.5, 0.5, -1.5, 0.0]  # no attrs -> 0, like legacy


def test_canonical_order_is_independent_of_input_order():
    rng = np.random.default_rng(1)
    e = SignedEdges(
        rng.uniform(-1, 1, 50).round(1),
        rng.integers(0, 9, 50),
        rng.integers(0, 9, 50),
        rng.integers(0, 2, 50),
    )
    a = e.sorted()
    b = e.take(rng.permutation(50)).sorted()
    assert a.score.tolist() == b.score.tolist()
    assert np.minimum(a.u, a.v).tolist() == np.minimum(b.u, b.v).tolist()
    assert a.attr.tolist() == b.attr.tolist()
    mags = np.abs(a.score)
    assert np.all(mags[1:] <= mags[:-1])


def test_cluster_labels_every_node_including_isolated():
    m = MutexWatershed({"aff": (1.0, 0.0)})
    nodes = np.array([1, 2, 3, 10, 20])
    edges = SignedEdges([0.9, -0.5], [2, 1], [3, 3], [0, 0])
    result = m.cluster(nodes, edges)
    assert result.labels.tolist() == [1, 2, 2, 10, 20]
    assert result.stats["n_components"] == 4


def test_cluster_rejects_unknown_endpoint():
    m = MutexWatershed({"aff": (1.0, 0.0)})
    with pytest.raises(ValueError, match="endpoint"):
        m.cluster(np.array([1, 2]), SignedEdges([0.5], [1], [3], [0]))


def test_contract_keeps_strongest_repulsive_edge_per_pair_only():
    m = MutexWatershed({"aff": (1.0, 0.0)})
    nodes = np.array([1, 2, 3, 4])
    labels = np.array([1, 1, 3, 3])
    edges = SignedEdges(
        [0.9, 0.9, -0.3, -0.6, 0.5, -0.1],
        [1, 3, 1, 2, 1, 1],
        [2, 4, 3, 4, 4, 2],
        [0, 0, 0, 1, 0, 0],
    )
    c = m.contract(nodes, labels, edges)
    assert len(c) == 1
    assert (c.score[0], c.u[0], c.v[0], c.attr[0]) == (-0.6, 1, 3, 1)


def test_derived_mutex_state_reproduces_child_decisions():
    """Re-clustering the contracted graph (components + carried repulsive
    edges) with no new edges must change nothing, and carried edges must block
    exactly the merges the child blocked: random graphs, soft and hard carry."""
    rng = np.random.default_rng(2)
    m_soft = MutexWatershed({"aff": (1.0, 0.0)})
    m_hard = MutexWatershed({"aff": (1.0, 0.0)}, commit_child_mutexes=True)
    for _ in range(20):
        n = 30
        nodes = np.arange(1, n + 1)
        edges = SignedEdges(
            rng.uniform(-1, 1, 120).round(1),
            rng.integers(1, n + 1, 120),
            rng.integers(1, n + 1, 120),
            np.zeros(120, dtype=np.int64),
        )
        edges = edges.take(edges.u != edges.v)
        child = m_soft.cluster(nodes, edges)
        comps = np.unique(child.labels)
        carried = m_soft.contract(nodes, child.labels, edges)
        for m in (m_soft, m_hard):
            again = m.cluster(comps, SignedEdges.empty(), carried=carried)
            assert again.labels.tolist() == comps.tolist()
        # every carried edge really was a committed mutex: an attractive edge
        # between the pair, weaker than it, must still be blocked
        if len(carried):
            weaker = SignedEdges(
                np.abs(carried.score) * 0.5, carried.u, carried.v, carried.attr
            )
            for m in (m_soft, m_hard):
                blocked = m.cluster(comps, weaker, carried=carried)
                assert blocked.labels.tolist() == comps.tolist()


def test_soft_vs_hard_carry_on_a_stronger_cross_edge():
    """A new attractive edge stronger than the carried mutex merges the pair
    under the default soft carry (as the global run would) and is refused
    under commit_child_mutexes=True."""
    comps = np.array([1, 2])
    carried = SignedEdges([-0.5], [1], [2], [0])
    new = SignedEdges([0.9], [1], [2], [0])
    soft = MutexWatershed({"aff": (1.0, 0.0)}).cluster(comps, new, carried=carried)
    assert soft.labels.tolist() == [1, 1]
    hard = MutexWatershed({"aff": (1.0, 0.0)}, commit_child_mutexes=True).cluster(
        comps, new, carried=carried
    )
    assert hard.labels.tolist() == [1, 2]


def test_cluster_matches_reference_with_prefix_and_carried():
    """``MutexWatershed.cluster`` (backend ``cluster_arrays``) equals a
    brute-force mutex watershed over the same processing order: prefix
    decisions first, then carried and new edges in canonical order (soft
    carry) or carried mutexes as prefix (hard carry). Random tie-heavy graphs
    with gaps in the node ids."""
    rng = np.random.default_rng(3)
    weights = {"aff": (1.0, 0.0)}
    for trial in range(20):
        nodes = np.unique(rng.integers(1, 10_000, 60))
        edges = SignedEdges(
            rng.uniform(-1, 1, 250).round(1),
            rng.choice(nodes, 250),
            rng.choice(nodes, 250),
            np.zeros(250, dtype=np.int64),
        )
        edges = edges.take(edges.u != edges.v)
        carried = SignedEdges([-0.5, -0.2], nodes[:2], nodes[2:4], [0, 0])
        prefix = [
            (True, int(nodes[5]), int(nodes[6])),
            (False, int(nodes[7]), int(nodes[8])),
        ]
        for commit in (False, True):
            m = MutexWatershed(weights, commit_child_mutexes=commit)
            labels = m.cluster(nodes, edges, carried=carried, prefix=prefix).labels
            if commit:
                order = prefix + carried.sorted().to_mws() + edges.sorted().to_mws()
            else:
                order = prefix + SignedEdges.concat([edges, carried]).sorted().to_mws()
            ref = reference_mws(order)
            expected = groups_of({int(n): ref.get(int(n), int(n)) for n in nodes})
            assert (
                groups_of(dict(zip(nodes.tolist(), labels.tolist(), strict=False)))
                == expected
            ), (
                trial,
                commit,
            )
            # labels are the smallest member of each group
            for n, lab in zip(nodes.tolist(), labels.tolist(), strict=False):
                assert lab <= n
