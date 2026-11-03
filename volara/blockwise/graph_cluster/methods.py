"""Clustering methods for :class:`GraphCluster`.

A method owns three things and nothing else: how raw edge attributes become
signed scores (applied exactly once, at the leaf), how a block's graph is
clustered, and how the clustered graph is contracted into what the parent
block sees. The octree, scheduler, store and LUT composition never look
inside, which is what lets other methods (GASP linkages, exact hierarchical
agglomeration with boundary freezing) plug in without touching the driver.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import mwatershed as mws
import numpy as np
from numpy.typing import ArrayLike

MwsEdge = tuple[bool, int, int]


class SignedEdges:
    """Edges with a prepared signed score: positive attracts, negative repels,
    magnitude is priority. ``attr`` is the index of the weight attribute the
    score came from (for deterministic tie-breaking). Endpoints are node ids
    of whatever level the table belongs to. Any array-likes are accepted and
    stored as 1-D numpy arrays."""

    score: np.ndarray
    u: np.ndarray
    v: np.ndarray
    attr: np.ndarray

    def __init__(self, score: ArrayLike, u: ArrayLike, v: ArrayLike, attr: ArrayLike):
        self.score = np.asarray(score, dtype=np.float64).reshape(-1)
        self.u = np.asarray(u, dtype=np.int64).reshape(-1)
        self.v = np.asarray(v, dtype=np.int64).reshape(-1)
        self.attr = np.asarray(attr, dtype=np.int64).reshape(-1)
        n = len(self.score)
        assert len(self.u) == len(self.v) == len(self.attr) == n, "ragged SignedEdges"

    def __repr__(self) -> str:
        return f"SignedEdges(n={len(self)})"

    @classmethod
    def empty(cls) -> "SignedEdges":
        z = np.zeros(0, dtype=np.int64)
        return cls(np.zeros(0, dtype=np.float64), z, z, z)

    @classmethod
    def concat(cls, parts: list["SignedEdges"]) -> "SignedEdges":
        parts = [p for p in parts if p is not None]
        if not parts:
            return cls.empty()
        return cls(
            np.concatenate([p.score for p in parts]),
            np.concatenate([p.u for p in parts]),
            np.concatenate([p.v for p in parts]),
            np.concatenate([p.attr for p in parts]),
        )

    def __len__(self) -> int:
        return len(self.score)

    def take(self, idx) -> "SignedEdges":
        return SignedEdges(self.score[idx], self.u[idx], self.v[idx], self.attr[idx])

    def canonical_order(self) -> np.ndarray:
        """Permutation sorting by ``(-|score|, min(u,v), max(u,v), attr)``.

        Magnitude first (what MWS needs), then endpoints and attribute index so
        ties are broken independently of database row order, worker count or
        task order. Ties are common on real data, so without this the
        partition would depend on the database backend.
        """
        lo = np.minimum(self.u, self.v)
        hi = np.maximum(self.u, self.v)
        return np.lexsort((self.attr, hi, lo, -np.abs(self.score)))

    def sorted(self) -> "SignedEdges":
        return self.take(self.canonical_order())

    def to_mws(self) -> list[MwsEdge]:
        """``(positive, u, v)`` tuples in the current order. Zero scores are
        repulsive, matching the legacy ``aff > 0`` rule."""
        return list(zip((self.score > 0).tolist(), self.u.tolist(), self.v.tolist()))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"score": self.score, "u": self.u, "v": self.v, "attr": self.attr}

    @classmethod
    def from_arrays(cls, arrays) -> "SignedEdges":
        return cls(arrays["score"], arrays["u"], arrays["v"], arrays["attr"])


@dataclass
class ClusterResult:
    labels: np.ndarray
    """Component id per input node, aligned with the ``nodes`` array handed to
    ``cluster``. A component is named after its smallest member id, so ids stay
    globally unique and deterministic across blocks and levels."""
    stats: dict = field(default_factory=dict)


class ClusterMethod(Protocol):
    name: str
    attr_names: list[str]

    def prepare(
        self, u: ArrayLike, v: ArrayLike, attrs: ArrayLike
    ) -> tuple[SignedEdges, np.ndarray]: ...

    def cluster(
        self,
        nodes: np.ndarray,
        edges: SignedEdges,
        carried: SignedEdges | None = None,
        prefix: list[MwsEdge] | None = None,
    ) -> ClusterResult: ...

    def contract(
        self, nodes: np.ndarray, labels: np.ndarray, edges: SignedEdges
    ) -> SignedEdges: ...


def _lookup_sorted(keys_sorted: np.ndarray, values: np.ndarray, query: np.ndarray):
    """``values[i]`` for each ``query`` element equal to ``keys_sorted[i]``;
    raises if any query key is missing."""
    pos = np.searchsorted(keys_sorted, query)
    pos_clipped = np.minimum(pos, len(keys_sorted) - 1) if len(keys_sorted) else pos
    found = (
        (pos < len(keys_sorted)) & (keys_sorted[pos_clipped] == query)
        if len(keys_sorted)
        else np.zeros(len(query), dtype=bool)
    )
    if not np.all(found):
        missing = np.unique(query[~found])
        raise KeyError(
            f"{len(missing)} ids not found (first few: {missing[:10].tolist()})"
        )
    return values[pos_clipped]


def min_id_labels(nodes: np.ndarray, reps: np.ndarray) -> np.ndarray:
    """Relabel each group of equal ``reps`` to the smallest ``nodes`` id in it."""
    nodes = np.asarray(nodes, dtype=np.int64)
    reps = np.asarray(reps, dtype=np.int64)
    if len(nodes) == 0:
        return nodes.copy()
    order = np.argsort(reps, kind="stable")
    r = reps[order]
    n = nodes[order]
    new_group = np.r_[True, r[1:] != r[:-1]]
    starts = np.flatnonzero(new_group)
    mins = np.minimum.reduceat(n, starts)
    sizes = np.diff(np.r_[starts, len(r)])
    labels = np.empty_like(nodes)
    labels[order] = np.repeat(mins, sizes)
    return labels


class MutexWatershed:
    """Mutex watershed on a signed edge list via the ``mwatershed`` backend.

    Carried state for the parent is the contracted graph itself. Merges are
    monotone and a merge is refused only by an existing mutex, so (a) every
    repulsive edge whose endpoints end in different components was processed
    while they were separate and therefore committed a mutex, and (b) every
    attractive edge between different final components was blocked by a
    mutex that itself came from a repulsive edge between the same two
    components. Hence the committed mutex constraints between a block's
    components are exactly its surviving inter-component repulsive edges,
    and surviving inter-component attractive edges are dead. ``contract``
    keeps the strongest repulsive edge per component pair (lossless for MWS:
    once the first repulsive edge between two clusters is processed the
    mutex exists and later ones are no-ops). Verified against a brute-force
    implementation in the tests.
    """

    name = "mws"

    def __init__(
        self,
        weights: dict[str, tuple[float, float]],
        edge_per_attr: bool = True,
        commit_child_mutexes: bool = False,
    ):
        if not weights:
            raise ValueError("weights must name at least one edge attribute")
        self.weights = dict(weights)
        self.attr_names = list(self.weights.keys())
        self.edge_per_attr = edge_per_attr
        self.commit_child_mutexes = commit_child_mutexes

    # -- score preparation (applied exactly once, at the leaf) --------------

    def prepare(
        self, u: ArrayLike, v: ArrayLike, attrs: ArrayLike
    ) -> tuple[SignedEdges, np.ndarray]:
        """Turn raw edge attributes (``(n, n_attrs)``, NaN = missing) into
        signed scores ``w * x + b``. Returns the signed edges and, per signed
        edge, the index of the raw edge it came from.

        With ``edge_per_attr`` every present attribute yields its own signed
        edge; otherwise the present attributes are summed into one edge (an
        edge with no attributes scores 0, as in the legacy implementation).
        """
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        attrs = np.asarray(attrs, dtype=np.float64).reshape(
            len(u), len(self.attr_names)
        )
        w = np.array([self.weights[a][0] for a in self.attr_names])
        b = np.array([self.weights[a][1] for a in self.attr_names])
        scores = w * attrs + b
        present = ~np.isnan(attrs)
        if self.edge_per_attr:
            src, attr = np.nonzero(present)
            return (
                SignedEdges(scores[src, attr], u[src], v[src], attr),
                src,
            )
        total = np.where(present, scores, 0.0).sum(axis=1)
        src = np.arange(len(u))
        return SignedEdges(total, u, v, np.zeros(len(u), dtype=np.int64)), src

    # -- clustering ---------------------------------------------------------

    def cluster(
        self,
        nodes: np.ndarray,
        edges: SignedEdges,
        carried: SignedEdges | None = None,
        prefix: list[MwsEdge] | None = None,
    ) -> ClusterResult:
        """Cluster ``nodes`` (sorted, unique int64 ids). ``edges`` are this
        block's edges, ``carried`` the surviving edges of child blocks (the
        child mutex state), ``prefix`` optional hard decisions processed first
        (e.g. a starting LUT's merges)."""
        t0 = time.perf_counter()
        nodes = np.asarray(nodes, dtype=np.int64)
        assert np.all(nodes[1:] > nodes[:-1]), "nodes must be sorted and unique"
        carried = carried if carried is not None else SignedEdges.empty()
        head: list[MwsEdge] = list(prefix) if prefix else []

        if self.commit_child_mutexes:
            # child mutexes become hard constraints, processed before any new edge
            head += carried.sorted().to_mws()
            ordered = edges.sorted()
        else:
            ordered = SignedEdges.concat([edges, carried]).sorted()

        self._check_endpoints(nodes, ordered)
        pairs = mws.cluster(head + ordered.to_mws())

        reps = nodes.copy()
        if pairs:
            in_ids = np.fromiter(
                (p[0] for p in pairs), dtype=np.int64, count=len(pairs)
            )
            out_ids = np.fromiter(
                (p[1] for p in pairs), dtype=np.int64, count=len(pairs)
            )
            order = np.argsort(in_ids)
            in_sorted, out_sorted = in_ids[order], out_ids[order]
            pos = np.searchsorted(in_sorted, nodes)
            pos = np.minimum(pos, len(in_sorted) - 1)
            hit = in_sorted[pos] == nodes
            reps[hit] = out_sorted[pos[hit]]
        labels = min_id_labels(nodes, reps)

        return ClusterResult(
            labels=labels,
            stats={
                "n_nodes": int(len(nodes)),
                "n_edges": int(len(edges)),
                "n_carried": int(len(carried)),
                "n_prefix": int(len(head)),
                "n_components": int(len(np.unique(labels))),
                "cluster_seconds": time.perf_counter() - t0,
            },
        )

    @staticmethod
    def _check_endpoints(nodes: np.ndarray, edges: SignedEdges) -> None:
        if len(edges) == 0:
            return
        for name, ends in (("u", edges.u), ("v", edges.v)):
            try:
                _lookup_sorted(nodes, nodes, ends)
            except KeyError as e:
                raise ValueError(
                    f"edge endpoint {name} not among block nodes: {e}"
                ) from e

    # -- contraction --------------------------------------------------------

    def contract(
        self, nodes: np.ndarray, labels: np.ndarray, edges: SignedEdges
    ) -> SignedEdges:
        """Project ``edges`` through ``labels`` and keep what the parent must
        see: the strongest repulsive edge per distinct component pair.
        Internal edges and inter-component attractive edges are dropped (see
        class docstring)."""
        if len(edges) == 0:
            return SignedEdges.empty()
        lu = _lookup_sorted(nodes, labels, edges.u)
        lv = _lookup_sorted(nodes, labels, edges.v)
        keep = (lu != lv) & (edges.score <= 0)
        projected = SignedEdges(edges.score[keep], lu[keep], lv[keep], edges.attr[keep])
        if len(projected) == 0:
            return projected
        # canonical order puts the strongest edge of every pair first; a
        # stable regroup by (lo, hi) then makes "first of each group" the
        # strongest repulsive edge between that component pair
        projected = projected.sorted()
        lo = np.minimum(projected.u, projected.v)
        hi = np.maximum(projected.u, projected.v)
        order = np.lexsort((np.arange(len(lo)), hi, lo))
        lo_s, hi_s = lo[order], hi[order]
        first = np.r_[True, (lo_s[1:] != lo_s[:-1]) | (hi_s[1:] != hi_s[:-1])]
        kept = np.sort(order[first])
        return SignedEdges(
            projected.score[kept], lo[kept], hi[kept], projected.attr[kept]
        )


class GASP:
    """Placeholder for GASP (Bailoni et al. 2022): generalised agglomeration
    with ``sum`` / ``mean`` / ``max`` / ``min`` linkage via the backend's
    ``cluster_weighted``. Not implemented yet.

    What it needs beyond :class:`MutexWatershed`: ``cluster`` passes the
    signed float scores and a linkage name instead of booleans, and
    ``contract`` must keep **every** surviving inter-component edge
    (attractive ones too) so the parent can recompute the linkage statistic
    exactly from individual edges; only the greedy processing history is
    approximated. The derived-mutex argument of :class:`MutexWatershed` does
    not transfer to cannot-link constraints under other linkages.
    """

    name = "gasp"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "method='gasp' is not implemented yet; see the GASP class docstring"
        )


class HierarchicalAgglomeration:
    """Placeholder for exact distributed hierarchical agglomeration with
    boundary freezing (Lu, Zlateski & Seung 2021, arXiv:2106.10795). Not
    implemented yet.

    Unlike MWS this can be made **exact**: with a reducible linkage, a block
    commits only merges that cannot be affected by edges it has not seen
    (those not touching a frozen boundary region) and carries the residual
    graph upward. ``contract`` therefore owns the "which merges to commit"
    decision and the carried state is the residual graph, not just surviving
    repulsive edges. The driver, octree and store need no change.
    """

    name = "hierarchical"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "method='hierarchical' is not implemented yet; see the "
            "HierarchicalAgglomeration class docstring"
        )


METHODS: dict[str, type] = {
    "mws": MutexWatershed,
    "gasp": GASP,
    "hierarchical": HierarchicalAgglomeration,
}


def get_method(name: str, **kwargs) -> ClusterMethod:
    try:
        cls = METHODS[name]
    except KeyError:
        raise ValueError(
            f"unknown clustering method {name!r}; known: {sorted(METHODS)}"
        )
    return cls(**kwargs)
