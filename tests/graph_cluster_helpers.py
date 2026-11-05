"""Shared helpers for the GraphCluster tests: synthetic RAGs in SQLite, a
brute-force mutex watershed, and partition comparison."""

from __future__ import annotations

import networkx as nx
import numpy as np
from funlib.geometry import Roi

from volara.dbs import SQLite


def make_db(path, nodes: dict, edges: list, edge_attrs=("aff",)) -> SQLite:
    """``nodes``: id -> position; ``edges``: ``(u, v, {attr: value})``."""
    db = SQLite(path=path, edge_attrs={a: "float" for a in edge_attrs})
    provider = db.open("w")
    graph = nx.Graph()
    for node, pos in nodes.items():
        graph.add_node(
            int(node), position=tuple(float(p) for p in pos), size=1, filtered=False
        )
    for u, v, attrs in edges:
        graph.add_edge(int(u), int(v), **attrs)
    provider.write_graph(graph)
    return db


def random_rag(
    roi: Roi,
    n_nodes: int,
    n_edges: int,
    seed: int,
    ties: bool = True,
    long_range: float = 0.3,
    id_offset: int = 1,
):
    """Random positions in ``roi``; mostly short edges, some long ones; scores
    rounded to one decimal when ``ties`` so many magnitudes collide."""
    rng = np.random.default_rng(seed)
    begin = np.asarray(roi.begin, dtype=float)
    shape = np.asarray(roi.shape, dtype=float)
    pos = begin + rng.random((n_nodes, roi.dims)) * shape
    ids = np.arange(n_nodes) + id_offset
    nodes = {int(i): tuple(p) for i, p in zip(ids, pos, strict=False)}
    edges = {}
    tries = 0
    while len(edges) < n_edges and tries < 50 * n_edges:
        tries += 1
        u = int(rng.integers(n_nodes))
        if rng.random() < long_range:
            v = int(rng.integers(n_nodes))
        else:
            d = np.linalg.norm(pos - pos[u], axis=1)
            d[u] = np.inf
            cand = np.argsort(d)[:6]
            v = int(rng.choice(cand))
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in edges:
            continue
        aff = float(rng.uniform(-1, 1))
        if ties:
            aff = round(aff, 1)
        edges[key] = {"aff": aff}
    return nodes, [(int(ids[u]), int(ids[v]), a) for (u, v), a in edges.items()]


def partition(lut: np.ndarray, nodes=None) -> frozenset:
    """Fragment groupings of a LUT, ignoring which member names each group.
    ``nodes`` adds singletons for nodes missing from the LUT: the legacy
    ``GraphMWS`` LUT omits fragments without edges, ``GraphCluster`` lists them."""
    groups: dict[int, set[int]] = {}
    seen = set()
    for frag, seg in zip(lut[0].tolist(), lut[1].tolist(), strict=False):
        groups.setdefault(int(seg), set()).add(int(frag))
        seen.add(int(frag))
    parts = [frozenset(g) for g in groups.values()]
    if nodes is not None:
        parts += [frozenset({int(n)}) for n in nodes if int(n) not in seen]
    return frozenset(parts)


def reference_mws(edges):
    """Brute-force mutex watershed over sorted ``(positive, u, v)`` edges.
    Returns node -> representative."""
    parent: dict[int, int] = {}
    mutex: dict[int, set[int]] = {}

    def find(x):
        parent.setdefault(x, x)
        mutex.setdefault(x, set())
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for positive, u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if rv in mutex[ru]:
            continue
        if positive:
            a, b = (ru, rv) if ru < rv else (rv, ru)
            parent[b] = a
            for m in mutex.pop(b):
                mutex[m].discard(b)
                mutex[m].add(a)
                mutex[a].add(m)
        else:
            mutex[ru].add(rv)
            mutex[rv].add(ru)
    return {x: find(x) for x in list(parent)}


def groups_of(mapping: dict) -> frozenset:
    groups: dict[int, set[int]] = {}
    for k, v in mapping.items():
        groups.setdefault(v, set()).add(k)
    return frozenset(frozenset(g) for g in groups.values())
