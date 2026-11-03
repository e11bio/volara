"""``GraphCluster``: hierarchical (octree) graph clustering with persisted
intermediate state, and the per-level ``BlockwiseTask`` it is built from.

Level 0 blocks read their own nodes and edges from the RAG, cluster, and
contract; every higher level unions its children's contracted graphs, adds
the edges first owned by it (those crossing its children), clusters again,
and contracts; the root composes all block LUTs into one fragment -> segment
LUT. With ``num_levels=1`` there is a single block over the whole roi and
the result is the plain in-memory algorithm.

Hierarchical mutex watershed is an approximation of the global run: a
block commits every decision it can make from the edges it sees, parents
may merge child components but never split them, and the only irreversible
errors are leaf decisions a cross-boundary edge would have prevented. The
error therefore scales with the fraction of edges crossing block
boundaries, so leaves should be as large as a worker can hold.
"""

from __future__ import annotations

import logging
import resource
import time
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import daisy
import numpy as np
from funlib.geometry import Coordinate, Roi
from pydantic import Field, field_validator, model_validator

from volara.lut import LUT
from volara.segment_utils import replace_values

from ...dbs import PostgreSQL, SQLite
from ...utils import PydanticCoordinate, PydanticRoi, StrictBaseModel
from ...workers import Worker
from ..blockwise import BlockwiseTask
from .methods import ClusterMethod, MwsEdge, SignedEdges, _lookup_sorted, get_method
from .octree import Index, Octree
from .store import HierarchyStore

if TYPE_CHECKING:
    from ..pipeline import Pipeline

logger = logging.getLogger(__name__)

DB = Annotated[
    PostgreSQL | SQLite,
    Field(discriminator="db_type"),
]


def _max_per_key(
    keys_in: np.ndarray, values: np.ndarray, *, keys: np.ndarray | None = None
):
    """Max of ``values`` per distinct key in ``keys_in``. With ``keys`` given
    (sorted unique), the result is aligned to them, ``-1`` for keys without
    values."""
    k = np.asarray(np.unique(keys_in) if keys is None else keys, dtype=np.int64)
    out = np.full(len(k), -1, dtype=np.int64)
    if len(values):
        pos = np.searchsorted(k, keys_in)
        np.maximum.at(out, pos, np.asarray(values, dtype=np.int64))
    return k, out


def _lookup_optional(keys_sorted: np.ndarray, values: np.ndarray, query: np.ndarray):
    """``values`` for each ``query`` key, plus a found mask (missing -> 0)."""
    out = np.zeros(len(query), dtype=values.dtype)
    if len(keys_sorted) == 0 or len(query) == 0:
        return out, np.zeros(len(query), dtype=bool)
    pos = np.minimum(np.searchsorted(keys_sorted, query), len(keys_sorted) - 1)
    found = keys_sorted[pos] == query
    out[found] = values[pos[found]]
    return out, found


def _isin_sorted(query: np.ndarray, sorted_unique: np.ndarray) -> np.ndarray:
    if len(sorted_unique) == 0:
        return np.zeros(len(query), dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_unique, query), len(sorted_unique) - 1)
    return sorted_unique[pos] == query


def _edge_arrays(rows, name_u: str, name_v: str, attr_names: list[str]):
    """Edge rows (dicts from ``read_edges``) as ``u``, ``v`` and an
    ``(n, n_attrs)`` float array with NaN for a missing attribute."""
    n = len(rows)
    u = np.fromiter((r[name_u] for r in rows), dtype=np.int64, count=n)
    v = np.fromiter((r[name_v] for r in rows), dtype=np.int64, count=n)
    attrs = np.full((n, len(attr_names)), np.nan)
    for j, name in enumerate(attr_names):
        attrs[:, j] = np.fromiter(
            (np.nan if r.get(name) is None else float(r[name]) for r in rows),
            dtype=np.float64,
            count=n,
        )
    return u, v, attrs


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class GraphCluster(StrictBaseModel):
    """
    Hierarchical graph clustering of a RAG into a fragment -> segment LUT.

    The roi is split into an octree of ``num_levels`` levels; each level is a
    blockwise task and the levels run in series as a pipeline. ``num_levels=1``
    reproduces the in-memory :class:`GraphMWS` partition for the same graph
    and weights (under canonical edge ordering, see ``GraphMWS.canonical_order``).
    Deeper hierarchies are an approximation of the global run: parents may
    merge child components but never split them.
    """

    db: DB
    roi: PydanticRoi
    """The total region to cluster; nodes outside it, and edges to them, are ignored."""
    weights: dict[str, tuple[float, float]]
    """
    Edge attribute -> ``(weight, bias)``. Each present attribute scores
    ``weight * value + bias``: positive merges, negative splits, larger
    magnitude is processed first. Same semantics as :class:`GraphMWS`.
    """
    lut: LUT
    """Where the final fragment -> segment LUT is written. A path string is accepted."""
    method: Literal["mws", "gasp", "hierarchical"] = "mws"
    """Clustering method. Only ``"mws"`` is implemented; the others raise ``NotImplementedError``."""
    num_levels: int | None = 1
    """
    Octree depth. 1 = one in-memory block over the whole roi. ``None`` derives
    the smallest depth whose root is a single block from ``block_shape``.
    """
    num_workers: int | list[int] = 1
    """Workers per level: one value for every level, or a list with one per level."""
    num_cache_workers: int | None = None
    """Threads per worker running blocks (see :class:`BlockwiseTask`)."""
    worker_config: Worker | list[Worker | None] | None = None
    """
    How workers are started (``SlurmWorker``, ``LSFWorker``, ``LocalWorker``,
    or ``None`` for local processes): one config for every level, or a list
    with one per level, e.g. many small cluster nodes for the leaves and one
    large-memory node for the root, whose single block holds the whole
    contracted graph. Each level is its own blockwise task, so levels may run
    on different queues.
    """
    work_dir: Path | None = None
    """Intermediate per-block state (restartable). Defaults to a directory next to ``lut``."""
    block_shape: PydanticCoordinate | None = None
    """Leaf block shape in world units. Defaults to ``ceil(roi.shape / 2**(num_levels-1))``."""
    edge_per_attr: bool = True
    """One signed edge per present attribute (default) or one summed edge per RAG edge."""
    commit_child_mutexes: bool = False
    """
    If True, repulsive decisions a child block committed are hard constraints
    in the parent (never overridden). Default False carries them as ordinary
    edges with their original priority, which is closer to the global run.
    """
    starting_lut: LUT | None = None
    """Merges that must be preserved. Only supported with ``num_levels=1``."""
    bounded_read: bool = True
    """
    With ``num_levels=1`` only: ``False`` reads the whole database without a
    spatial query, as :class:`GraphMWS` does, which is much faster on large
    graphs when the roi covers the whole graph anyway. Nodes outside the roi
    are then included, exactly like the legacy task.
    """
    block_timeout: float = 24 * 60 * 60
    """Per-block daisy timeout; the root block of a large graph runs long."""

    @field_validator("lut", "starting_lut", mode="before")
    @classmethod
    def _coerce_lut(cls, value):
        if isinstance(value, (str, Path)):
            return LUT(path=value)
        return value

    @model_validator(mode="after")
    def _validate(self):
        if self.num_levels is None:
            if self.block_shape is None:
                raise ValueError(
                    "num_levels=None needs block_shape to derive the depth"
                )
            self.num_levels = Octree.levels_for(self.roi, self.block_shape)
        if self.num_levels < 1:
            raise ValueError("num_levels must be >= 1")
        depth = self.num_levels
        for name in ("num_workers", "worker_config"):
            value = getattr(self, name)
            if isinstance(value, list) and len(value) != depth:
                raise ValueError(
                    f"{name} has {len(value)} entries but there are {depth} levels"
                )
        if not self.weights:
            raise ValueError("weights must name at least one edge attribute")
        if self.starting_lut is not None and self.num_levels != 1:
            raise ValueError("starting_lut is only supported with num_levels=1")
        if not self.bounded_read and self.num_levels != 1:
            raise ValueError("bounded_read=False is only supported with num_levels=1")
        self.octree  # validates roi vs. levels / block_shape, raises if unusable
        self.make_method()  # unimplemented methods fail here, not in a worker
        return self

    # -- derived objects -----------------------------------------------------

    @property
    def depth(self) -> int:
        """``num_levels`` after validation resolved it (never ``None``)."""
        assert self.num_levels is not None
        return self.num_levels

    @property
    def octree(self) -> Octree:
        return Octree(self.roi, self.depth, self.block_shape)

    @property
    def store_dir(self) -> Path:
        if self.work_dir is not None:
            return Path(self.work_dir)
        return Path(str(self.lut.file.parent)) / f"{self.lut.name}-graph-cluster"

    @property
    def store(self) -> HierarchyStore:
        return HierarchyStore(self.store_dir)

    def make_method(self) -> ClusterMethod:
        return get_method(
            self.method,
            weights=self.weights,
            edge_per_attr=self.edge_per_attr,
            commit_child_mutexes=self.commit_child_mutexes,
        )

    def manifest(self) -> dict:
        tree = self.octree
        return {
            "version": 1,
            "db": self.db.id,
            "roi": [list(self.roi.begin), list(self.roi.shape)],
            "num_levels": self.depth,
            "leaf_shape": [int(x) for x in tree.leaf_shape],
            "weights": {k: list(v) for k, v in self.weights.items()},
            "method": self.method,
            "edge_per_attr": self.edge_per_attr,
            "commit_child_mutexes": self.commit_child_mutexes,
            "starting_lut": None
            if self.starting_lut is None
            else self.starting_lut.uri,
            "bounded_read": self.bounded_read,
        }

    # -- tasks ----------------------------------------------------------------

    def _per_level(self, value, level: int):
        return value[level] if isinstance(value, list) else value

    def level_task(self, level: int) -> "GraphClusterLevel":
        return GraphClusterLevel(
            config=self,
            level=level,
            roi=self.roi,
            num_workers=self._per_level(self.num_workers, level),
            num_cache_workers=self.num_cache_workers,
            worker_config=self._per_level(self.worker_config, level),
            block_timeout=self.block_timeout,
        )

    @property
    def tasks(self) -> list["GraphClusterLevel"]:
        return [self.level_task(level) for level in range(self.depth)]

    def pipeline(self) -> "Pipeline":
        from ..pipeline import Pipeline

        return reduce(lambda a, b: a + b, (Pipeline(t) for t in self.tasks))

    def run_blockwise(self, multiprocessing: bool = True):
        """Run every level in order. Returns daisy's ``{task_id: TaskState}``."""
        logger.info("GraphCluster octree: %s", self.octree.describe())
        return self.pipeline().run_blockwise(multiprocessing=multiprocessing)

    def drop(self, drop_outputs: bool = True) -> None:
        for task in self.tasks:
            task.drop(drop_outputs=False)
        if drop_outputs:
            self.store.drop()
            self.lut.drop()

    def __add__(self, other):
        return self.pipeline() + other

    def __or__(self, other):
        return self.pipeline() | other

    def intermediate_lut(self, level: int) -> np.ndarray:
        """The partition after ``level`` has run, as a full fragment -> component
        LUT (shape ``(2, n)``), composed from the per-block LUTs of levels
        ``0..level``. ``level = num_levels - 1`` equals the final LUT. Every
        level is a complete, if less merged, partition of all fragments."""
        if not 0 <= level < self.depth:
            raise ValueError(f"level must be in [0, {self.depth}), got {level}")
        store = self.store
        for j in range(level + 1):
            if not all(store.is_done(j, index) for index in self.octree.blocks(j)):
                raise RuntimeError(f"level {j} is not complete in {store.work_dir}")
        in_ids, out_ids = store.level_lut(0)
        for j in range(1, level + 1):
            lj_in, lj_out = store.level_lut(j)
            out_ids = replace_values(out_ids, lj_in, lj_out)
        return np.array([in_ids, out_ids])

    def measured_block_memory(self, level: int) -> dict | None:
        """Peak RSS per block of ``level`` as daisy measured it in the last run
        (``resource_tracking``), in MB: ``{"max", "mean", "n_blocks"}``. This
        is the number to size a worker's memory request from; ``None`` if no
        report exists (e.g. tracking off)."""
        try:
            from daisy._report import read_last_run
        except ImportError:  # pragma: no cover - older daisy
            return None
        report = read_last_run(self.level_task(level).tracking_path / "runs.jsonl")
        if report is None:
            return None
        peak = report.blocks.peak_rss  # daisy Stat over the blocks measured this run
        if peak.n == 0:
            return None
        return {
            "max": peak.max / 2**20,
            "mean": peak.mean / 2**20,
            "n_blocks": int(peak.n),
            "wall_seconds_max": report.blocks.wall.max,
        }

    def summary(self) -> dict:
        """Octree description plus per-level aggregates of the block stats,
        including the measured per-block peak RSS where daisy recorded it."""
        store = self.store
        levels = []
        for level in range(self.depth):
            stats = store.level_stats(level)
            measured = self.measured_block_memory(level)
            levels.append(
                {
                    "level": level,
                    "n_blocks": len(stats),
                    "n_nodes": sum(s.get("n_nodes", 0) for s in stats),
                    "n_edges_local": sum(s.get("n_edges_local", 0) for s in stats),
                    "n_edges_carried": sum(s.get("n_edges_carried", 0) for s in stats),
                    "n_edges_new": sum(s.get("n_edges_new", 0) for s in stats),
                    "n_edges_deferred": sum(
                        s.get("n_edges_deferred", 0) for s in stats
                    ),
                    "n_edges_dropped_outside": sum(
                        s.get("n_edges_dropped_outside", 0) for s in stats
                    ),
                    "n_edges_dropped_final": sum(
                        s.get("n_edges_dropped_final", 0) for s in stats
                    ),
                    "n_components_carried": sum(
                        s.get("n_components_carried", 0) for s in stats
                    ),
                    "n_components": sum(s.get("n_components", 0) for s in stats),
                    "seconds": sum(s.get("seconds", 0.0) for s in stats),
                    "max_rss_mb": max(
                        (s.get("max_rss_mb", 0.0) for s in stats), default=0.0
                    ),
                    "block_peak_rss_mb": measured,
                }
            )
        return {"octree": self.octree.describe(), "levels": levels}


class GraphClusterLevel(BlockwiseTask):
    """One octree level of a :class:`GraphCluster` as a blockwise task.
    Blocks are the level's octree blocks; dependencies on the level below are
    spatial, so a parent runs as soon as its children are done."""

    task_type: Literal["graph-cluster-level"] = "graph-cluster-level"

    config: GraphCluster
    level: int

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.config.lut.name}-graph-cluster-L{self.level}"

    @property
    def write_roi(self) -> Roi:
        return self.config.roi

    @property
    def write_size(self) -> Coordinate:
        return Coordinate(*self.config.octree.block_shape(self.level).tolist())

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.config.roi.dims)

    @property
    def is_root(self) -> bool:
        return self.config.octree.is_root(self.level)

    def init(self):
        self.config.store.ensure_manifest(self.config.manifest())

    def drop_artifacts(self):
        self.config.store.drop_level(self.level)
        if self.is_root:
            self.config.lut.drop()

    @contextmanager
    def process_block_func(self):
        config = self.config
        db = config.db.open("r")
        if self.level == 0:
            self._warn_if_v_index_missing(db)
        octree = config.octree
        store = config.store
        method = config.make_method()
        starting = None
        if config.starting_lut is not None:
            starting = config.starting_lut.load()
            assert starting is not None, f"unable to load {config.starting_lut}"

        def process_block(block: daisy.Block):
            self._process_block(block, db, octree, store, method, starting)

        yield process_block

    @staticmethod
    def _warn_if_v_index_missing(db) -> None:
        """Leaf reads fetch edges by either endpoint. Without an index on the
        second endpoint column every leaf scans the whole edge table (measured
        ~300x slower per leaf). funlib.persistence creates the index for new
        databases; older ones may lack it."""
        con = getattr(db, "con", None)
        if con is None or not hasattr(con, "execute"):
            return  # not SQLite; PostgreSQL databases get the index at creation
        try:
            v = db.endpoint_names[1]
            rows = con.execute(
                f"SELECT name FROM sqlite_master WHERE type='index' "
                f"AND tbl_name='{db.edges_table_name}'"
            ).fetchall()
            indexed = set()
            for (name,) in rows:
                cols = [
                    c[2] for c in con.execute(f"PRAGMA index_info({name})").fetchall()
                ]
                if cols:
                    indexed.add(cols[0])
            if v not in indexed:
                logger.warning(
                    "edges table has no index on '%s': every leaf block will scan "
                    "the whole edge table. Create it once with "
                    "`CREATE INDEX %s_%s_index ON %s(%s)` for fast leaf reads.",
                    v,
                    db.edges_table_name,
                    v,
                    db.edges_table_name,
                    v,
                )
        except Exception as e:  # diagnostics only, never fail a run over it
            logger.debug("could not inspect edge indexes: %s", e)

    # -- block processing ------------------------------------------------------

    def _process_block(self, block, db, octree, store, method, starting):
        level = self.level
        index = octree.index_of_roi(level, block.write_roi)
        if store.is_done(level, index):
            if octree.is_root(level) and not self.config.lut.exists():
                # the block's own state is intact but its output, the final
                # LUT, is gone: recompose it rather than recluster
                logger.info(
                    "%s complete but final LUT missing; recomposing",
                    Octree.key(level, index),
                )
                nodes, labels = store.read_lut(level, index)
                self._finalize(nodes, labels, level, store)
                return
            logger.info("%s already complete, skipping", Octree.key(level, index))
            return
        t0 = time.perf_counter()
        store.clear_block(level, index)

        if level == 0:
            nodes, edges, carried, boundary, extra = self._read_leaf(
                block.write_roi, index, db, octree, store, method
            )
        else:
            nodes, edges, carried, boundary, extra = self._load_parent(
                index, octree, store
            )

        prefix = (
            self._starting_prefix(nodes, starting) if starting is not None else None
        )
        result = method.cluster(nodes, edges, carried=carried, prefix=prefix)
        labels = result.labels
        store.write_lut(level, index, nodes, labels)

        if not octree.is_root(level):
            orig, comp_in, max_level, max_attractive = boundary
            comp_out = _lookup_sorted(nodes, labels, comp_in)
            # a component can still merge iff one of its members has an
            # attractive edge pending above this level; everything else is
            # final here and its boundary rows and edges are dropped
            active = np.unique(comp_out[max_attractive > level])
            keep = (max_level > level) & _isin_sorted(comp_out, active)
            store.write_boundary(
                level,
                index,
                orig[keep],
                comp_out[keep],
                max_level[keep],
                max_attractive[keep],
            )
            contracted = method.contract(
                nodes, labels, SignedEdges.concat([edges, carried])
            )
            both_active = _isin_sorted(contracted.u, active) & _isin_sorted(
                contracted.v, active
            )
            contracted = contracted.take(both_active)
            store.write_graph(level, index, active, contracted)
            extra["n_edges_contracted"] = int(len(contracted))
            extra["n_components_carried"] = int(len(active))
        else:
            self._finalize(nodes, labels, level, store)

        stats = {
            "level": level,
            "index": list(index),
            "roi": [list(block.write_roi.begin), list(block.write_roi.shape)],
            "n_edges_local": int(len(edges)) if level == 0 else 0,
            "n_edges_new": int(len(edges)) if level > 0 else 0,
            "n_edges_carried": int(len(carried)),
            "seconds": time.perf_counter() - t0,
            "max_rss_mb": _max_rss_mb(),
            **extra,
            **result.stats,
        }
        store.write_stats(level, index, stats)
        store.mark_done(level, index)
        logger.info(
            "%s: %d nodes, %d edges -> %d components in %.2fs",
            Octree.key(level, index),
            stats["n_nodes"],
            stats["n_edges_local"] + stats["n_edges_new"] + stats["n_edges_carried"],
            stats["n_components"],
            stats["seconds"],
        )

    def _read_leaf(self, roi: Roi, index: Index, db, octree: Octree, store, method):
        """Read the nodes owned by this leaf and every edge touching them.
        Same-leaf edges are clustered here; cross-leaf edges are written to
        the block that first contains both endpoints (their LCA) by the leaf
        owning ``u``; endpoints outside the roi drop the edge."""
        attr_names = method.attr_names
        read_roi = roi if self.config.bounded_read else None
        name_u, name_v = db.endpoint_names
        if hasattr(db, "read_edges_arrays"):
            # columnar reads: ~10-20x less peak memory than one dict per row
            nodes = np.unique(db.read_nodes_arrays(read_roi, read_attrs=[])["id"])
            cols = db.read_edges_arrays(
                roi=read_roi, read_attrs=attr_names, fetch_on_v=True
            )
            u, v = cols[name_u], cols[name_v]
            attrs = (
                np.stack([cols[a] for a in attr_names], axis=1)
                if len(u)
                else np.zeros((0, len(attr_names)))
            )
            del cols
        else:
            node_rows = db.read_nodes(read_roi, read_attrs=[])
            nodes = np.unique(
                np.fromiter(
                    (r["id"] for r in node_rows), dtype=np.int64, count=len(node_rows)
                )
            )
            rows = db.read_edges(roi=read_roi, read_attrs=attr_names, fetch_on_v=True)
            u, v, attrs = _edge_arrays(rows, name_u, name_v, attr_names)
            del rows
        if len(nodes) and nodes[0] == 0:
            logger.warning(
                "%s: ignoring node id 0 (reserved background)", Octree.key(0, index)
            )
            nodes = nodes[1:]
        ok = (u != v) & (u != 0) & (v != 0)
        u, v, attrs = u[ok], v[ok], attrs[ok]

        in_u = _isin_sorted(u, nodes)
        in_v = _isin_sorted(v, nodes)
        local = in_u & in_v
        cross = in_u ^ in_v
        n_dropped = int((~(in_u | in_v)).sum())

        # position lookup for the foreign endpoint of every cross edge
        other_id = np.where(in_u, v, u)
        foreign = np.unique(other_id[cross])
        foreign_leaf = octree.leaf_index(self._positions(db, foreign, octree.dims))
        group = np.full(
            len(u), -1, dtype=np.int64
        )  # LCA level per raw edge, -1 = not deferred
        if len(foreign):
            cross_idx = np.flatnonzero(cross)
            other_leaf = foreign_leaf[np.searchsorted(foreign, other_id[cross_idx])]
            valid = np.all(other_leaf >= 0, axis=1)
            n_dropped += int((~valid).sum())
            cross_idx = cross_idx[valid]
            my_leaf = np.broadcast_to(np.asarray(index), (len(cross_idx), octree.dims))
            lca_level, _ = octree.lca(my_leaf, other_leaf[valid])
            group[cross_idx] = lca_level

        signed, src = method.prepare(u, v, attrs)
        local_edges = signed.take(local[src])

        n_deferred = 0
        for lvl in np.unique(group[group > 0]):
            mask = (group[src] == lvl) & in_u[src]  # the leaf owning u writes the edge
            if not mask.any():
                continue
            target = octree.ancestor(0, index, int(lvl))
            store.write_deferred(int(lvl), target, 0, index, signed.take(mask))
            n_deferred += int(mask.sum())

        # Nodes of this leaf with an edge deferred to some ancestor: the
        # highest level any such edge is introduced at (needed to map the
        # edge there) and the highest level of an *attractive* one. A
        # component none of whose members has an attractive edge pending
        # above can never merge again: it is final here and is not carried up
        # (see ``GraphClusterLevel._process_block``); repulsive deferred edges touching
        # it are dropped at the ancestor since they could only block merges
        # that cannot happen.
        lca = group[src]
        pending = lca > 0
        mine_all = np.where(in_u, u, v)[src]
        uniq, max_level = _max_per_key(mine_all[pending], lca[pending])
        _, max_attractive = _max_per_key(
            mine_all[pending & (signed.score > 0)],
            lca[pending & (signed.score > 0)],
            keys=uniq,
        )
        boundary = (uniq, uniq, max_level, max_attractive)

        extra = {
            "n_edges_raw": int(len(u)),
            "n_edges_deferred": n_deferred,
            "n_edges_dropped_outside": n_dropped,
        }
        return nodes, local_edges, SignedEdges.empty(), boundary, extra

    @staticmethod
    def _positions(db, ids: np.ndarray, dims: int) -> np.ndarray:
        """Positions of ``ids`` (sorted unique), NaN rows for unknown ids."""
        pos = np.full((len(ids), dims), np.nan)
        if len(ids) == 0:
            return pos
        if hasattr(db, "read_nodes_arrays"):
            cols = db.read_nodes_arrays(
                ids=ids.tolist(), read_attrs=[db.position_attribute]
            )
            pos[np.searchsorted(ids, cols["id"])] = cols[db.position_attribute]
            return pos
        rows = db.read_nodes(ids=ids.tolist(), read_attrs=[db.position_attribute])
        for r in rows:
            p = r.get(db.position_attribute)
            if p is None:
                continue
            i = np.searchsorted(ids, r["id"])
            pos[i] = p
        return pos

    def _load_parent(self, index: Index, octree: Octree, store: HierarchyStore):
        """Union the children's contracted graphs and introduce the edges
        deferred to this block, projected onto child component ids."""
        level = self.level
        node_parts, edge_parts, orig_parts, comp_parts, lvl_parts, attr_parts = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for child in octree.children(level, index):
            if not store.is_done(level - 1, child):
                raise RuntimeError(
                    f"child block {Octree.key(level - 1, child)} of "
                    f"{Octree.key(level, index)} is not complete"
                )
            n, e = store.read_graph(level - 1, child)
            node_parts.append(n)
            edge_parts.append(e)
            o, c, m, a = store.read_boundary(level - 1, child)
            orig_parts.append(o)
            comp_parts.append(c)
            lvl_parts.append(m)
            attr_parts.append(a)

        nodes = (
            np.sort(np.concatenate(node_parts)) if node_parts else np.zeros(0, np.int64)
        )
        if len(nodes) > 1 and np.any(nodes[1:] == nodes[:-1]):
            raise RuntimeError(
                f"{Octree.key(level, index)}: duplicate component ids across children"
            )
        carried = SignedEdges.concat(edge_parts)
        orig = np.concatenate(orig_parts) if orig_parts else np.zeros(0, np.int64)
        comp = np.concatenate(comp_parts) if comp_parts else np.zeros(0, np.int64)
        max_level = np.concatenate(lvl_parts) if lvl_parts else np.zeros(0, np.int64)
        max_attractive = (
            np.concatenate(attr_parts) if attr_parts else np.zeros(0, np.int64)
        )

        deferred = store.read_deferred(level, index)
        n_dropped_final = 0
        if len(deferred):
            order = np.argsort(orig)
            new_u, found_u = _lookup_optional(orig[order], comp[order], deferred.u)
            new_v, found_v = _lookup_optional(orig[order], comp[order], deferred.v)
            found = found_u & found_v
            # an attractive edge always has both endpoints tracked (each leaf
            # records its own endpoint); a repulsive one may touch a component
            # that is already final, in which case it cannot matter
            if np.any(~found & (deferred.score > 0)):
                bad = deferred.take(~found & (deferred.score > 0))
                raise RuntimeError(
                    f"{Octree.key(level, index)}: {len(bad)} attractive deferred "
                    f"edges have an endpoint with no boundary mapping from the "
                    f"children (first: {int(bad.u[0])}-{int(bad.v[0])}); this is a "
                    "bug in edge ownership"
                )
            n_dropped_final = int((~found).sum())
            new_edges = SignedEdges(
                deferred.score[found], new_u[found], new_v[found], deferred.attr[found]
            )
        else:
            new_edges = SignedEdges.empty()

        extra = {
            "n_children": len(node_parts),
            "n_edges_dropped_final": n_dropped_final,
        }
        return nodes, new_edges, carried, (orig, comp, max_level, max_attractive), extra

    @staticmethod
    def _starting_prefix(nodes: np.ndarray, starting: np.ndarray) -> list[MwsEdge]:
        """Chain the members of every starting-LUT group with hard merges."""
        in_ids = np.asarray(starting[0], dtype=np.int64)
        out_ids = np.asarray(starting[1], dtype=np.int64)
        order = np.argsort(in_ids)
        seg = nodes.copy()
        pos = np.minimum(np.searchsorted(in_ids[order], nodes), max(len(order) - 1, 0))
        hit = (len(order) > 0) & (in_ids[order][pos] == nodes)
        seg[hit] = out_ids[order][pos[hit]]
        by_seg = np.argsort(seg, kind="stable")
        s, n = seg[by_seg], nodes[by_seg]
        same = s[1:] == s[:-1]
        return [(True, int(a), int(b)) for a, b in zip(n[:-1][same], n[1:][same])]

    def _finalize(self, nodes, labels, level, store: HierarchyStore) -> None:
        """Compose every level's block LUTs into the final fragment -> segment LUT."""
        if level == 0:
            in_ids, out_ids = nodes, labels
        else:
            in_ids, out_ids = store.level_lut(0)
            for j in range(1, level):
                lj_in, lj_out = store.level_lut(j)
                out_ids = replace_values(out_ids, lj_in, lj_out)
            out_ids = replace_values(out_ids, nodes, labels)
        self.config.lut.save(np.array([in_ids, out_ids]))
        logger.info(
            "wrote %s: %d fragments -> %d segments",
            self.config.lut.uri,
            len(in_ids),
            len(np.unique(out_ids)),
        )
