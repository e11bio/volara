import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from typing import Annotated, Literal

import daisy
import networkx as nx
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence.graphs import CSRGraphDataBase
from pydantic import Field, model_validator

from ..dbs import CSR, GraphProvider, PostgreSQL, SQLite
from ..node_arrays import NodeArrays, NodeArraysReader
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

logger = logging.getLogger(__file__)

DB = Annotated[
    PostgreSQL | SQLite | CSR,
    Field(discriminator="db_type"),
]


def similarity(
    a: np.ndarray,
    b: np.ndarray,
    transform: Literal["atanh"] | None = None,
    clip: float = 1e-6,
) -> np.ndarray:
    """
    The cosine similarity of the rows of ``a`` and ``b``, as float64. A row
    that is all zero or not finite gives NaN.

    With ``transform="atanh"`` the cosine is clipped to
    ``[-(1 - clip), 1 - clip]`` and passed through ``atanh``, which spreads
    out the similarities near 1 and -1.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.einsum("ij,ij->i", a, b) / norms
    cosine[~np.isfinite(cosine)] = np.nan
    cosine = np.clip(cosine, -1.0, 1.0)
    if transform == "atanh":
        cosine = np.arctanh(np.clip(cosine, -(1.0 - clip), 1.0 - clip))
    return cosine


class EdgeSimilarity(BlockwiseTask):
    """
    Score the edges of a graph by the similarity of the vectors of their two
    nodes, reading the graph alone: no voxel array is opened.

    The vectors are a node attribute kept beside the graph in a
    :class:`~volara.node_arrays.NodeArrays` store, e.g. the embedding a graph
    model wrote for every fragment. Every block reads its nodes and the edges
    that touch them from ``db``, computes one similarity per edge, and writes
    the nodes and the scored edges to ``out_db``. ``db`` is never written.
    :class:`GraphMWS` on ``out_db`` then turns the scores into a look up
    table, as it does for the scores of :class:`AffAgglom`.

    An edge belongs to the block that holds the end with the smaller id, so
    every edge is scored exactly once however the blocks are cut. An edge
    whose smaller end lies outside the task's ``roi`` is not scored.
    """

    task_type: Literal["edge-similarity"] = "edge-similarity"

    db: DB
    """
    The graph to score: its nodes, their positions and its edges. It is only
    read.
    """
    out_db: DB
    """
    The graph the scored edges are written to, with the nodes of ``db`` (id,
    position and ``copy_node_attrs``). Every attribute named in ``scores``
    must be one of its ``edge_attrs``.
    """
    vectors: NodeArrays
    """
    The node arrays holding one vector per node of ``db``.
    """
    attribute: str = "embedding"
    """
    The attribute of ``vectors`` to compare.
    """
    scores: dict[str, str | None]
    """
    The edge attributes to write, each with the edge attribute of ``db`` that
    says which edges get it: ``{out_attr: presence_attr}``. An edge gets its
    similarity as ``out_attr`` where its ``presence_attr`` in ``db`` is
    present, and no ``out_attr`` otherwise. ``None`` in place of a presence
    attribute scores every edge.

        `scores={"adj_similarity": "adj_weight", "lr_similarity": "lr_weight"}`

    keeps the similarities of adjacent and of long range edges apart, so
    :class:`GraphMWS` can give each its own bias.
    """
    metric: Literal["cosine"] = "cosine"
    """
    How two vectors are compared.
    """
    transform: Literal["atanh"] | None = None
    """
    An optional transform of the similarity. ``"atanh"`` spreads out the
    cosines near 1 and -1 (see :func:`similarity`).
    """
    clip: float = 1e-6
    """
    How far from 1 and -1 a cosine is clipped before ``atanh``.
    """
    copy_node_attrs: list[str] = ["size"]
    """
    Node attributes of ``db`` to copy to ``out_db``. One that ``db`` does not
    hold is left out.
    """
    block_size: PydanticCoordinate
    """
    The size of a block in world units, the units of the node positions.
    """
    bulk_write: bool = False
    """
    Whether to bulk-write to ``out_db`` (false by default). A database that
    is only written in bulk, such as :class:`~volara.dbs.CSR`, is bulk-written
    whatever this says.
    """

    fit: Literal["shrink"] = "shrink"
    """
    The boundary behavior for our daisy task.
    """
    read_write_conflict: Literal[False] = False
    """
    Blocks read ``db`` and write ``out_db``, so no block waits for another.
    """

    @model_validator(mode="after")
    def _scores_are_declared(self):
        if not self.scores:
            raise ValueError("scores names no edge attribute to write")
        declared = self.out_db.edge_attrs or {}
        missing = sorted(set(self.scores) - set(declared))
        if missing:
            raise ValueError(
                f"out_db.edge_attrs does not declare {missing}; "
                f"it declares {sorted(declared)}"
            )
        return self

    @property
    def task_name(self) -> str:
        return f"{self.out_db.id}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        if self.roi is not None:
            return self.roi
        if isinstance(self.db, CSR):
            bounds = self.db.open("r").meta.get("bounds")
            if bounds is not None:
                begin = Coordinate(int(np.floor(b)) for b in bounds[0])
                end = Coordinate(int(np.floor(b)) + 1 for b in bounds[1])
                return Roi(begin, end - begin)
        raise ValueError(
            "EdgeSimilarity needs a roi: only a CSR database with nodes in it "
            "knows the extent of its graph"
        )

    @property
    def write_size(self) -> Coordinate:
        return self.block_size

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.block_size.dims)

    def drop_artifacts(self):
        self.out_db.drop()

    def init(self) -> None:
        self.out_db.init()

    @property
    def _bulk(self) -> bool:
        """Whether blocks write through the bulk path: asked for, or the only
        way the database can be written."""
        return self.bulk_write or self.out_db.bulk_only

    def _task_context(self, worker, provider=None):
        """The bulk-write context of the task. A database that is only
        written in bulk counts the writes of the handle that entered the
        context, so a worker passes the ``provider`` its blocks write with."""
        if self._bulk:
            if provider is None or not self.out_db.bulk_only:
                provider = self.out_db.open("r+")
            return provider.bulk_write_mode(
                worker=worker, node_writes=True, edge_writes=True
            )
        return nullcontext()

    def _presence_attrs(self) -> list[str]:
        return sorted({attr for attr in self.scores.values() if attr is not None})

    def read_block(
        self, provider: GraphProvider, roi: Roi
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        The nodes of ``db`` in ``roi`` and the edges whose smaller end is one
        of them, as arrays.

        Returns ``(nodes, edges)``: ``nodes`` has ``id``, ``position`` and the
        ``copy_node_attrs`` the graph holds; ``edges`` has ``u``, ``v`` (ids,
        ``u < v``) and one float array per presence attribute, NaN where the
        edge does not have it.
        """
        presence = self._presence_attrs()
        position = provider.position_attribute
        copied = [a for a in self.copy_node_attrs if a in provider.node_attrs]
        if isinstance(provider, CSRGraphDataBase):
            read, edges = provider.read_arrays(
                roi, num_steps=1, node_attrs=[position, *copied], edge_attrs=presence
            )
            own = read["step"] == 0
            nodes = {"id": read["id"][own], "position": read[position][own]}
            for name in copied:
                nodes[name] = read[name][own]
            mine = np.isin(edges["u"], nodes["id"])
            edges = {name: values[mine] for name, values in edges.items()}
        else:
            rows = provider.read_nodes(roi, read_attrs=[position, *copied])
            nodes = {
                "id": np.array([row["id"] for row in rows], dtype=np.uint64),
                "position": np.array(
                    [row[position] for row in rows], dtype=np.float64
                ).reshape(len(rows), -1),
            }
            for name in copied:
                nodes[name] = np.array([row[name] for row in rows])
            name_u, name_v = provider.endpoint_names
            found = provider.read_edges(roi, read_attrs=presence, fetch_on_v=True)
            ends = np.array(
                [(row[name_u], row[name_v]) for row in found], dtype=np.uint64
            ).reshape(len(found), 2)
            u, v = ends.min(axis=1), ends.max(axis=1)
            mine = np.isin(u, nodes["id"])
            edges = {"u": u[mine], "v": v[mine]}
            for name in presence:
                values = np.array(
                    [np.nan if row[name] is None else row[name] for row in found],
                    dtype=np.float64,
                )
                edges[name] = values[mine]
        for name in presence:
            edges[name] = np.asarray(edges[name], dtype=np.float64)
        return nodes, edges

    def score_edges(
        self, edges: dict[str, np.ndarray], vectors: NodeArraysReader
    ) -> tuple[dict[str, np.ndarray], int]:
        """
        One array per attribute of ``scores`` for ``edges``, NaN where an
        edge does not get that score, and the number of edges that could not
        be scored because an end has no vector.
        """
        ends, inverse = np.unique(
            np.concatenate([edges["u"], edges["v"]]), return_inverse=True
        )
        found, values = vectors.read(ends, [self.attribute])
        stored = values[self.attribute]
        if stored.ndim != 2:
            raise ValueError(
                f"attribute {self.attribute!r} of {vectors.path} holds one value "
                "per node, not a vector"
            )
        at_u, at_v = inverse[: len(edges["u"])], inverse[len(edges["u"]) :]
        scored = similarity(
            stored[at_u], stored[at_v], transform=self.transform, clip=self.clip
        )
        scored[~(found[at_u] & found[at_v])] = np.nan
        unscored = int(np.isnan(scored).sum())
        out = {}
        for out_attr, presence in self.scores.items():
            if presence is None:
                out[out_attr] = scored
            else:
                out[out_attr] = np.where(np.isnan(edges[presence]), np.nan, scored)
        return out, unscored

    def write_block(
        self,
        provider: GraphProvider,
        nodes: dict[str, np.ndarray],
        edges: dict[str, np.ndarray],
        scores: dict[str, np.ndarray],
    ) -> None:
        """Write the nodes of a block and its scored edges to ``out_db``."""
        attrs = {
            name: values
            for name, values in nodes.items()
            if name != "id" and (name == "position" or name in provider.node_attrs)
        }
        if len(nodes["id"]) == 0:
            return
        if isinstance(provider, CSRGraphDataBase):
            provider.write_node_arrays(nodes["id"], **attrs)
            if len(edges["u"]) > 0:
                provider.write_edge_arrays(edges["u"], edges["v"], **scores)
            return
        # the nodes alone, so that an edge's far end is not written as a node
        own = nx.Graph()
        for i, node in enumerate(nodes["id"]):
            own.add_node(
                int(node),
                **{name: values[i].tolist() for name, values in attrs.items()},
            )
        scored = own.copy()
        for i, (u, v) in enumerate(zip(edges["u"], edges["v"], strict=True)):
            scored.add_edge(
                int(u),
                int(v),
                **{
                    name: float(values[i])
                    for name, values in scores.items()
                    if not np.isnan(values[i])
                },
            )
        node_attrs, edge_attrs = list(attrs), list(scores)
        if self._bulk:
            provider.bulk_write_nodes(own.nodes, attributes=node_attrs)
            provider.bulk_write_edges(scored.nodes, scored.edges, attributes=edge_attrs)
        else:
            provider.write_nodes(own.nodes, attributes=node_attrs)
            provider.write_edges(scored.nodes, scored.edges, attributes=edge_attrs)

    def score_block(
        self,
        block: daisy.Block,
        provider: GraphProvider,
        out_provider: GraphProvider,
        vectors: NodeArraysReader,
    ) -> None:
        nodes, edges = self.read_block(provider, block.write_roi)
        scores, unscored = self.score_edges(edges, vectors)
        self.write_block(out_provider, nodes, edges, scores)
        logger.info(
            "block %s: %d nodes, %d edges, %d without a score (an end has no vector)",
            block.write_roi,
            len(nodes["id"]),
            len(edges["u"]),
            unscored,
        )

    @contextmanager
    def process_block_func(self) -> Generator[Callable, None, None]:
        provider = self.db.open("r")
        out_provider = self.out_db.open("r+")
        vectors = self.vectors.open()

        def process_block(block) -> None:
            self.score_block(block, provider, out_provider, vectors)

        with self._task_context(worker=True, provider=out_provider):
            yield process_block

    @contextmanager
    def task(
        self,
        upstream_tasks: daisy.Task | list[daisy.Task] | None = None,
        multiprocessing: bool = True,
    ) -> Generator[daisy.Task, None, None]:
        # bulk_write_mode changes the database, so it has to exist first
        self.init()

        with self._task_context(worker=False):
            with super().task(upstream_tasks, multiprocessing) as task:
                yield task
