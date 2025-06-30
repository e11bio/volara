from contextlib import contextmanager
from typing import Annotated, Literal
import tempfile
import itertools
import functools
from pathlib import Path
import time

import daisy
import mwatershed as mws
import numpy as np
import networkx as nx
from funlib.geometry import Coordinate, Roi
from pydantic import Field

from volara.lut import LUT, LUTS

from ..dbs import PostgreSQL, SQLite
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask
from ..datasets import Labels

DB = Annotated[
    PostgreSQL | SQLite,
    Field(discriminator="db_type"),
]


class IterativeGraphMWS(BlockwiseTask):
    """
    Graph based execution of the MWS algorithm.
    Currently only supports executing in memory. The full graph for the given ROI
    is read into memory and then we run the mutex watershed algorithm on the full
    graph to get a globally optimal look up table.
    """

    task_type: Literal["graph-mws"] = "graph-mws"

    fragments_db: DB
    segments_db: DB
    """
    The db in which to store segment nodes and their attributes.
    """
    lut: LUT
    """
    The Look Up Table that will be saved on completion of this task.
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
    block_size: PydanticCoordinate
    edge_per_attr: bool = True
    """
    Whether or not to create a separate edge for each attribute in the weights. If
    False, the sum of all the weighted attributes will be used as the only edge weight.
    """

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.segments_db.id}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        return Roi(*self.roi)

    @property
    def write_size(self) -> Coordinate:
        return self.block_size

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.write_size.dims)

    @property
    def num_voxels_in_block(self) -> int:
        # We currently can't process in blocks
        return 1

    def drop_artifacts(self):
        self.lut.drop()
        self.segments_db.drop()

    @contextmanager
    def process_block_func(self):
        rag_provider = self.fragments_db.open("r+")
        out_rag_provider = self.segments_db.open("w")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)

            def process_block(block: daisy.Block):
                # mutex watershed inside write roi only to get super fragments
                graph = rag_provider.read_graph(
                    block.write_roi,
                    node_attrs=["size", "position"],
                    edge_attrs=list(self.weights.keys()),
                    both_sides=True,
                )

                t1 = time.time()
                edges = []
                for u, v, edge_attrs in graph.edges(data=True):
                    if (
                        graph.nodes[u].get("size", None) is None
                        or graph.nodes[v].get("size", None) is None
                    ):
                        # out of bounds nodes
                        continue
                    scores = [
                        w * edge_attrs[attr] + b
                        for attr, (w, b) in self.weights.items()
                        if edge_attrs.get(attr, None) is not None
                    ]
                    if self.edge_per_attr:
                        for score in scores:
                            edges.append((score, u, v))
                    else:
                        edges.append((sum(scores), u, v))

                # generate the look up table via mutex watershed clustering
                mws_lut: list[tuple[int, int]] = mws.cluster_edges(edges)
                inputs: list[int]
                outputs: list[int]
                if len(mws_lut) > 0:
                    inputs, outputs = [list(x) for x in zip(*mws_lut)]
                else:
                    inputs, outputs = [], []

                t1 = time.time()
                # save the lut to a temporary file for this block
                block_lut = LUT(
                    path=f"{tmp_path}/{'-'.join([str(o) for o in block.write_roi.offset])}-lut"
                )
                lut = np.array([inputs, outputs])
                block_lut.save(lut, edges=edges)

                t1 = time.time()
                # read luts and existing super fragments in neighboring blocks
                existing_luts = [
                    LUT(
                        path=f"{tmp_path}/{'-'.join([str(o) for o in block.write_roi.offset + block.write_roi.shape * Coordinate(*neighbor)])}-lut"
                    )
                    for neighbor in itertools.product(
                        *[range(-1, 2) for _ in range(block.write_roi.dims)]
                    )
                    if neighbor != (0,) * block.write_roi.dims
                ]

                # build a mapping of all fragments -> super fragments in this roi and neighboring blocks
                # build a mapping of edges (super_fragment - super_fragment) -> list[(fragment, fragment)]
                total_lut = functools.reduce(
                    lambda x, y: x + y,
                    [block_lut]
                    + [lut for lut in existing_luts if lut is not None]
                    + [self.lut],
                ).load()

                t1 = time.time()
                frag_seg_mapping: dict[int, int] = {
                    int(k): int(v) for k, v in total_lut.T
                }
                seg_frag_mapping: dict[int, set[int]] = {}
                for in_frag, out_frag in frag_seg_mapping.items():
                    in_group = seg_frag_mapping.setdefault(out_frag, set())
                    in_group.add(in_frag)

                t1 = time.time()
                out_graph = out_rag_provider.read_graph(block.read_roi)
                assert out_graph.number_of_nodes() == 0, out_graph.number_of_nodes

                t1 = time.time()
                for out_frag in np.unique(outputs):
                    if out_frag is not None and out_frag not in out_graph.nodes:
                        in_group = seg_frag_mapping[out_frag]

                        agglomeraged_attrs = {
                            "size": sum(
                                [graph.nodes[in_frag]["size"] for in_frag in in_group]
                            )
                        }
                        agglomeraged_attrs["position"] = (
                            sum(
                                [
                                    np.array(
                                        graph.nodes[in_frag]["position"], dtype=float
                                    )
                                    * graph.nodes[in_frag]["size"]
                                    for in_frag in in_group
                                    if in_frag in graph.nodes
                                ],
                                start=np.array((0,) * block.read_roi.dims, dtype=float),
                            )
                            / agglomeraged_attrs["size"]
                        )

                        out_graph.add_node(int(out_frag), **agglomeraged_attrs)

                t1 = time.time()
                edges_to_agglomerate = {}
                for u, v in graph.edges():
                    if u in frag_seg_mapping and v in frag_seg_mapping:
                        # all edges between super fragments associated with u and v
                        seg_u, seg_v = (
                            frag_seg_mapping[u],
                            frag_seg_mapping[v],
                        )
                        if seg_u != seg_v:
                            edges_to_agglomerate.setdefault((seg_u, seg_v), []).append(
                                (u, v)
                            )
                for (seg_u, seg_v), edges in edges_to_agglomerate.items():
                    out_graph.add_edge(
                        seg_u,
                        seg_v,
                        **{
                            weight_attr: sum(
                                graph.edges[edge].get(weight_attr, 0) for edge in edges
                            )
                            / len(edges)
                            for weight_attr in self.weights.keys()
                        },
                    )

                out_rag_provider.write_graph(out_graph, block.write_roi, both_sides=True)

            yield process_block

            block_luts = [LUT(path=block_lut) for block_lut in tmp_path.iterdir()]
            self.lut.save(LUTS(luts=block_luts).load())


class GraphMWSExtractFragments(BlockwiseTask):
    """
    Graph based execution of the MWS algorithm.
    Executes the mutex watershed algorithm on a graph to get a mapping of fragments
    to segments. Only stores the segments with their node attributes, does not
    create edges.
    """

    task_type: Literal["graph-mws"] = "graph-mws"

    fragments_db: DB
    segments_db: DB
    """
    The db in which to store segment nodes and their attributes.
    """
    lut: LUT
    """
    The Look Up Table that will be saved on completion of this task.
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
    block_size: PydanticCoordinate
    edge_per_attr: bool = True
    """
    Whether or not to create a separate edge for each attribute in the weights. If
    False, the sum of all the weighted attributes will be used as the only edge weight.
    """

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.segments_db.id}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        return Roi(*self.roi)

    @property
    def write_size(self) -> Coordinate:
        return self.block_size

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.write_size.dims)

    @property
    def num_voxels_in_block(self) -> int:
        # We currently can't process in blocks
        return 1

    def drop_artifacts(self):
        self.lut.drop()
        self.segments_db.drop()

    @contextmanager
    def process_block_func(self):
        rag_provider = self.fragments_db.open("r+")
        out_rag_provider = self.segments_db.open("w")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = Path(tmpdirname)

            def process_block(block: daisy.Block):
                # mutex watershed inside write roi only to get super fragments
                graph = rag_provider.read_graph(
                    block.write_roi,
                    node_attrs=["size", "position"],
                    edge_attrs=list(self.weights.keys()),
                )

                t1 = time.time()
                edges = []
                for u, v, edge_attrs in graph.edges(data=True):
                    if (
                        graph.nodes[u].get("size", None) is None
                        or graph.nodes[v].get("size", None) is None
                    ):
                        # out of bounds nodes
                        continue
                    scores = [
                        w * edge_attrs[attr] + b
                        for attr, (w, b) in self.weights.items()
                        if edge_attrs.get(attr, None) is not None
                    ]
                    if self.edge_per_attr:
                        for score in scores:
                            edges.append((score, u, v))
                    else:
                        edges.append((sum(scores), u, v))

                # generate the look up table via mutex watershed clustering
                mws_lut: list[tuple[int, int]] = mws.cluster_edges(edges)
                inputs: list[int]
                outputs: list[int]
                if len(mws_lut) > 0:
                    inputs, outputs = [list(x) for x in zip(*mws_lut)]
                else:
                    inputs, outputs = [], []

                t1 = time.time()
                # save the lut to a temporary file for this block
                block_lut = LUT(
                    path=f"{tmp_path}/{'-'.join([str(o) for o in block.write_roi.offset])}-lut"
                )
                lut = np.array([inputs, outputs])
                block_lut.save(lut, edges=edges)

                t1 = time.time()
                # read luts and existing super fragments in neighboring blocks
                existing_luts = [
                    LUT(
                        path=f"{tmp_path}/{'-'.join([str(o) for o in block.write_roi.offset + block.write_roi.shape * Coordinate(*neighbor)])}-lut"
                    )
                    for neighbor in itertools.product(
                        *[range(-1, 2) for _ in range(block.write_roi.dims)]
                    )
                    if neighbor != (0,) * block.write_roi.dims
                ]

                # build a mapping of all fragments -> super fragments in this roi and neighboring blocks
                # build a mapping of edges (super_fragment - super_fragment) -> list[(fragment, fragment)]
                total_lut = functools.reduce(
                    lambda x, y: x + y,
                    [block_lut]
                    + [lut for lut in existing_luts if lut is not None]
                    + [self.lut],
                ).load()

                t1 = time.time()
                frag_seg_mapping: dict[int, int] = {
                    int(k): int(v) for k, v in total_lut.T
                }
                seg_frag_mapping: dict[int, set[int]] = {}
                for in_frag, out_frag in frag_seg_mapping.items():
                    in_group = seg_frag_mapping.setdefault(out_frag, set())
                    in_group.add(in_frag)

                t1 = time.time()
                out_graph = out_rag_provider.read_graph(block.read_roi)
                assert out_graph.number_of_nodes() == 0, out_graph.number_of_nodes

                t1 = time.time()
                for out_frag in np.unique(outputs):
                    if out_frag is not None and out_frag not in out_graph.nodes:
                        in_group = seg_frag_mapping[out_frag]

                        agglomeraged_attrs = {
                            "size": sum(
                                [graph.nodes[in_frag]["size"] for in_frag in in_group]
                            )
                        }
                        agglomeraged_attrs["position"] = (
                            sum(
                                [
                                    np.array(
                                        graph.nodes[in_frag]["position"], dtype=float
                                    )
                                    * graph.nodes[in_frag]["size"]
                                    for in_frag in in_group
                                    if in_frag in graph.nodes
                                ],
                                start=np.array((0,) * block.read_roi.dims, dtype=float),
                            )
                            / agglomeraged_attrs["size"]
                        )

                        out_graph.add_node(int(out_frag), **agglomeraged_attrs)

                t1 = time.time()
                edges_to_agglomerate = {}
                for u, v in graph.edges():
                    if u in frag_seg_mapping and v in frag_seg_mapping:
                        # all edges between super fragments associated with u and v
                        seg_u, seg_v = (
                            frag_seg_mapping[u],
                            frag_seg_mapping[v],
                        )
                        if seg_u != seg_v:
                            edges_to_agglomerate.setdefault((seg_u, seg_v), []).append(
                                (u, v)
                            )
                for (seg_u, seg_v), edges in edges_to_agglomerate.items():
                    out_graph.add_edge(
                        seg_u,
                        seg_v,
                        **{
                            weight_attr: sum(
                                graph.edges[edge].get(weight_attr, 0) for edge in edges
                            )
                            / len(edges)
                            for weight_attr in self.weights.keys()
                        },
                    )

                out_g = nx.subgraph_view(
                    out_graph,
                    filter_node=lambda node: out_graph.nodes[node].get("position")
                    is not None,
                )
                out_rag_provider.write_nodes(out_g.nodes, block.write_roi)
                out_rag_provider.write_edges(out_graph.nodes, out_graph.edges)

            yield process_block

            block_luts = [LUT(path=block_lut) for block_lut in tmp_path.iterdir()]
            self.lut.save(LUTS(luts=block_luts).load())

class FragSegEdgeAgglom(BlockwiseTask):
    """
    Given a LUT of fragments to segments, a fragment db containing nodes and edges
    and a segments db containing only nodes. Agglomerate the edges between fragments
    into edges between segments.
    """

    task_type: Literal["frag-seg-edge-agglom"] = "frag-seg-edge-agglom"

    fragments_db: DB
    segments_db: DB
    """
    The db in which to store segment nodes and their attributes.
    """
    edge_attrs: list[str]
    """
    A list of edge attributes to agglomerate via size weighted mean.
    """
    roi: tuple[PydanticCoordinate, PydanticCoordinate]
    """
    The roi to process. This is the roi of the full graph to process.
    """
    block_size: PydanticCoordinate

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.segments_db.id}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        return Roi(*self.roi)

    @property
    def write_size(self) -> Coordinate:
        return self.block_size

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.write_size.dims)

    @property
    def num_voxels_in_block(self) -> int:
        # We currently can't process in blocks
        return 1

    def drop_artifacts(self):
        self.segments_db.drop_edges()

    @contextmanager
    def process_block_func(self):
        frag_provider = self.fragments_db.open("r+")
        seg_provider = self.segments_db.open("w")

        def process_block(block: daisy.Block):
            # mutex watershed inside write roi only to get super fragments
            frag_graph = frag_provider.read_graph(
                block.write_roi,
                node_attrs=["size", "position"],
                edge_attrs=self.edge_attrs,
            )
            seg_graph = seg_provider.read_graph(
                block.read_roi,
                node_attrs=["size", "position"],
                edge_attrs=self.edge_attrs,
            )

            frag_seg_mapping = ...

            seg_frag_edge_mapping = {}
            for u, v in frag_graph.edges():
                if u in frag_seg_mapping and v in frag_seg_mapping:
                    # all edges between super fragments associated with u and v
                    seg_u, seg_v = (
                        frag_seg_mapping[u],
                        frag_seg_mapping[v],
                    )
                    if seg_u != seg_v:
                        seg_frag_edge_mapping.setdefault((seg_u, seg_v), []).append(
                            (u, v)
                        )
            for (seg_u, seg_v), edges in seg_frag_edge_mapping.items():
                seg_graph.add_edge(
                    seg_u,
                    seg_v,
                    **{
                        weight_attr: sum(
                            frag_graph.edges[edge].get(weight_attr, 0) for edge in edges
                        )
                        / len(edges)
                        for weight_attr in self.weights.keys()
                    },
                )

            seg_provider.write_edges(seg_graph.nodes, seg_graph.edges)

        yield process_block


class GraphMWS(BlockwiseTask):
    task_type: Literal["create-lut"] = "create-lut"
    frags_data: Labels
    db: DB
    lut: Path
    starting_lut: Path | None = None
    bias: dict[str, float]
    roi: tuple[PydanticCoordinate, PydanticCoordinate] | None = None
    edge_per_attr: bool = True
    store_segment_intensities: dict[str, str] | None = None
    out_db: DB | None = None
    bounded_read: bool = True

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.frags_data.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        if self.roi is not None:
            return Roi(*self.roi)
        else:
            return self.frags_data.array("r").roi

    @property
    def write_size(self) -> Coordinate:
        return self.write_roi.shape

    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0,) * self.write_size.dims)

    @property
    def num_voxels_in_block(self) -> int:
        return 1

    def drop_artifacts(self):
        if self.lut.exists():
            self.lut.unlink()
        if self.out_db is not None:
            self.out_db.drop()

    @contextmanager
    def process_block_func(self):
        rag_provider = self.db.open("r+")

        if self.out_db is not None:
            out_rag_provider = self.out_db.open("w")

        if self.starting_lut is not None:
            starting_lut = (
                self.starting_lut
                if self.starting_lut.name.endswith(".npz")
                else f"{self.starting_lut}.npz"
            )
            starting_frags, starting_segs = np.load(starting_lut)[
                "fragment_segment_lut"
            ]
            starting_map = {
                in_frag: out_frag
                for in_frag, out_frag in zip(starting_frags, starting_segs)
            }
        else:
            starting_map = None

        def process_block(block):
            read_roi = block.write_roi if self.bounded_read else None
            node_attrs = (
                ["size"] + list(self.store_segment_intensities.keys())
                if self.store_segment_intensities is not None
                else []
            )
            graph = rag_provider.read_graph(
                read_roi, node_attrs=node_attrs, edge_attrs=list(self.bias.keys())
            )

            edges = []

            for u, v, edge_attrs in graph.edges(data=True):
                scores = [
                    edge_attrs.get(b, None) + self.bias[b]
                    for b in self.bias
                    if edge_attrs.get(b, None) is not None
                ]
                if self.edge_per_attr:
                    for score in scores:
                        edges.append((score, u, v))
                else:
                    edges.append((sum(scores), u, v))

            prefix_edges = []
            if starting_map is not None:
                groups = {}
                for node in graph.nodes:
                    groups.setdefault(starting_map[node], set()).add(node)
                for group in groups.values():
                    group = list(group)
                    for u, v in zip(group, group[1:]):
                        prefix_edges.append((True, u, v))

            edges = sorted(
                edges,
                key=lambda edge: abs(edge[0]),
                reverse=True,
            )
            edges = [(bool(aff > 0), u, v) for aff, u, v in edges]
            lut = mws.cluster(prefix_edges + edges)
            if len(lut) > 0:
                inputs, outputs = zip(*lut)
            else:
                inputs, outputs = [], []

            lut = np.array([inputs, outputs])

            np.savez_compressed(self.lut, fragment_segment_lut=lut, edges=edges)

            if self.store_segment_intensities is not None:
                assert self.out_db is not None, self.out_db
                out_graph = out_rag_provider.read_graph(block.write_roi)
                assert out_graph.number_of_nodes() == 0, out_graph.number_of_nodes
                mapping = {}
                for in_frag, out_frag in zip(inputs, outputs):
                    in_group = mapping.setdefault(out_frag, set())
                    in_group.add(in_frag)

                for out_frag, in_group in mapping.items():
                    # update size. Each fragment will
                    computed_codes = {
                        "seg_size": sum(
                            [graph.nodes[in_frag]["size"] for in_frag in in_group]
                        )
                    }
                    for in_code, out_code in self.store_segment_intensities.items():
                        out_codes = [
                            np.array(graph.nodes[in_frag][in_code])
                            * graph.nodes[in_frag]["size"]
                            for in_frag in in_group
                        ]
                        out_data = np.mean(np.array(out_codes), axis=0)
                        computed_codes[out_code] = out_data
                    for in_frag in in_group:
                        frag_attrs = graph.nodes[in_frag]
                        frag_attrs.update(computed_codes)
                        out_graph.add_node(in_frag, **frag_attrs)

                out_rag_provider.write_graph(out_graph, block.write_roi)

        yield process_block
