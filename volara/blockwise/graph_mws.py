from contextlib import contextmanager
from typing import Annotated, Literal

import daisy
import mwatershed as mws
import numpy as np
from funlib.geometry import Coordinate, Roi
from pydantic import Field

from volara.lut import LUT

from ..dbs import PostgreSQL, SQLite
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

DB = Annotated[
    PostgreSQL | SQLite,
    Field(discriminator="db_type"),
]


class GraphMWS(BlockwiseTask):
    """
    Graph based execution of the MWS algorithm.
    Currently only supports executing in memory. The full graph for the given ROI
    is read into memory and then we run the mutex watershed algorithm on the full
    graph to get a globally optimal look up table.
    """

    task_type: Literal["graph-mws"] = "graph-mws"

    db: DB
    lut: LUT
    """
    The Look Up Table that will be saved on completion of this task.
    """
    starting_lut: LUT | None = None
    """
    An optional Look Up Table that provides a set of merged fragments that must
    be preserved in the final Look Up Table.
    """
    weights: dict[str, tuple[float, float]]
    """
    A dictionary of edge attributes and their weight and bias. These will be used
    to compute the edge weights for the mutex watershed algorithm. Positive edges
    will result in fragments merging, negative edges will result in splitting and
    edges will be processed in order of high to low magnitude.
    Each attribute will have a final score of `w * edge_data[attr] + b` for every
    `attr, (w, b) in weights.items()`
    If an attribute is not present in the edge data it will be skipped.
    """
    roi: tuple[PydanticCoordinate, PydanticCoordinate]
    """
    The roi to process. This is the roi of the full graph to process.
    """
    edge_per_attr: bool = True
    """
    Whether or not to create a separate edge for each attribute in the weights. If
    False, the sum of all the weighted attributes will be used as the only edge weight.
    """
    mean_attrs: dict[str, str] | None = None
    """
    A dictionary of attributes to compute the mean of for each segment. Given
    `mean_attrs = {"attr1": "out_attr1"}` and nodes `n_i` in a segment `s` we will
    set `s.out_attr1 = sum(n_i.attr1 * n_i.size) / sum(n_i.size)`.
    """
    out_db: DB | None = None
    """
    The db in which to store segment nodes and their attributes. Must not be None
    if `mean_attrs` is not None.
    """
    bounded_read: bool = True
    """
    Reading from the db can be made more efficient by not doing a spatial query
    and assuming we want all nodes and edges. If you don't want to process a
    sub volume of the graph setting this to false will speed up the read.
    """

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.lut.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        return Roi(*self.roi)

    @property
    def write_size(self) -> Coordinate:
        return self.write_roi.shape

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.write_size.dims)

    @property
    def num_voxels_in_block(self) -> int:
        # We currently can't process in blocks
        return 1

    def drop_artifacts(self):
        self.lut.drop()
        if self.out_db is not None:
            self.out_db.drop()

    @contextmanager
    def process_block_func(self):
        rag_provider = self.db.open("r+")

        if self.out_db is not None:
            out_rag_provider = self.out_db.open("w")

        if self.starting_lut is not None:
            starting_frags, starting_segs = self.starting_lut.load()
            starting_map = {
                in_frag: out_frag
                for in_frag, out_frag in zip(starting_frags, starting_segs)
            }
        else:
            starting_map = None

        def process_block(block: daisy.Block):
            read_roi = block.write_roi if self.bounded_read else None
            node_attrs = (
                ["size"] + list(self.mean_attrs.keys())
                if self.mean_attrs is not None
                else []
            )
            graph = rag_provider.read_graph(
                read_roi, node_attrs=node_attrs, edge_attrs=list(self.weights.keys())
            )

            edges = []

            for u, v, edge_attrs in graph.edges(data=True):
                scores = [
                    w * edge_attrs.get(attr, None) + b
                    for attr, (w, b) in self.weights.items()
                    if edge_attrs.get(attr, None) is not None
                ]
                if self.edge_per_attr:
                    for score in scores:
                        edges.append((score, u, v))
                else:
                    edges.append((sum(scores), u, v))

            prefix_edges = []
            if starting_map is not None:
                groups: dict[int, set[int]] = {}
                for node in graph.nodes:
                    groups.setdefault(starting_map[node], set()).add(node)
                for group in groups.values():
                    pre_merged_ids = list(group)
                    for u, v in zip(pre_merged_ids, pre_merged_ids[1:]):
                        prefix_edges.append((True, u, v))

            edges = sorted(
                edges,
                key=lambda edge: abs(edge[0]),
                reverse=True,
            )
            edges = [(bool(aff > 0), u, v) for aff, u, v in edges]

            # generate the look up table via mutex watershed clustering
            mws_lut: list[tuple[int, int]] = mws.cluster(prefix_edges + edges)
            inputs: list[int]
            outputs: list[int]
            if len(mws_lut) > 0:
                inputs, outputs = [list(x) for x in zip(*mws_lut)]
            else:
                inputs, outputs = [], []

            lut = np.array([inputs, outputs])
            self.lut.save(lut, edges=edges)

            if self.mean_attrs is not None:
                assert self.out_db is not None, self.out_db
                out_graph = out_rag_provider.read_graph(block.write_roi)
                assert out_graph.number_of_nodes() == 0, out_graph.number_of_nodes
                mapping: dict[int, set[int]] = {}
                for in_frag, out_frag in zip(inputs, outputs):
                    in_group = mapping.setdefault(out_frag, set())
                    in_group.add(in_frag)

                for out_frag, in_group in mapping.items():
                    computed_codes = {
                        "seg_size": sum(
                            [graph.nodes[in_frag]["size"] for in_frag in in_group]
                        )
                    }
                    for in_code, out_code in self.mean_attrs.items():
                        out_codes = [
                            np.array(graph.nodes[in_frag][in_code])
                            * graph.nodes[in_frag]["size"]
                            for in_frag in in_group
                        ]
                        out_data = (
                            np.sum(np.array(out_codes), axis=0)
                            / computed_codes["seg_size"]
                        )
                        computed_codes[out_code] = out_data
                    for in_frag in in_group:
                        frag_attrs = graph.nodes[in_frag]
                        frag_attrs.update(computed_codes)
                        out_graph.add_node(in_frag, **frag_attrs)

                out_rag_provider.write_graph(out_graph, block.write_roi)

        yield process_block
