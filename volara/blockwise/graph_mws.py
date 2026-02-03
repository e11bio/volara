import functools
import itertools
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

import daisy
import mwatershed as mws
import networkx as nx
import numpy as np
from funlib.geometry import Coordinate, Roi
from pydantic import Field

from volara.lut import LUT, LUTS
from volara.tmp import replace_values

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
        assert self.roi is not None, "ROI must be set for GraphMWS task"
        return self.roi

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
        benchmark_logger = self.get_benchmark_logger()
        rag_provider = self.db.open("r+")

        if self.out_db is not None:
            out_rag_provider = self.out_db.open("w")

        if self.starting_lut is not None:
            starting_lut = self.starting_lut.load()
            assert starting_lut is not None, "Unable to load starting LUT"
            starting_frags, starting_segs = starting_lut
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
            with benchmark_logger.trace("Read graph"):
                graph = rag_provider.read_graph(
                    read_roi,
                    node_attrs=node_attrs,
                    edge_attrs=list(self.weights.keys()),
                )

            with benchmark_logger.trace("Prepare MWS edges"):
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

            with benchmark_logger.trace("Run MWS"):
                # generate the look up table via mutex watershed clustering
                mws_lut: list[tuple[int, int]] = mws.cluster(prefix_edges + edges)

            inputs: list[int]
            outputs: list[int]
            if len(mws_lut) > 0:
                inputs, outputs = [list(x) for x in zip(*mws_lut)]
            else:
                inputs, outputs = [], []
            lut = np.array([inputs, outputs])

            with benchmark_logger.trace("Save LUT"):
                self.lut.save(lut, edges=edges)

            if self.mean_attrs is not None:
                with benchmark_logger.trace("Agglomerate Mean Attrs"):
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

                with benchmark_logger.trace("Write out graph"):
                    out_rag_provider.write_graph(out_graph, block.write_roi)

        yield process_block


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

    fit: Literal["overhang"] = "overhang"
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
        return self.write_size

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
                    block.read_roi,
                    node_attrs=["size", "position"],
                    edge_attrs=list(self.weights.keys())
                    + [f"{attr}__size" for attr in self.weights.keys()],
                    both_sides=True,
                )

                edges = []
                inputs = set(
                    node
                    for node, attrs in graph.nodes(data=True)
                    if attrs.get("position") is not None
                    and block.write_roi.contains(attrs["position"])
                )
                for u, v, edge_attrs in graph.edges(data=True):
                    u_pos = graph.nodes[u].get("position", None)
                    v_pos = graph.nodes[v].get("position", None)
                    if (
                        u_pos is not None and v_pos is not None
                        # and block.write_roi.contains(u_pos)  # run on full read roi graph
                        # and block.write_roi.contains(v_pos)  # relabel and ignore out of bounds nodes
                    ):
                        assert graph.nodes[u].get("size", None) is not None
                        assert graph.nodes[v].get("size", None) is not None
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

                contained_nodes = set(
                    node
                    for node, attrs in graph.nodes(data=True)
                    if "position" in attrs
                    and block.write_roi.contains(attrs["position"])
                )

                relabeled_nodes = {}

                edges = sorted(
                    edges,
                    key=lambda edge: abs(edge[0]),
                    reverse=True,
                )
                edges = [(bool(aff > 0), u, v) for aff, u, v in edges]

                # generate the look up table via mutex watershed clustering
                mws_lut: list[tuple[int, int]] = mws.cluster(edges)
                mws_lut_relabelled = []
                for in_frag, out_frag in mws_lut:
                    if in_frag in contained_nodes:
                        if out_frag in contained_nodes:
                            mws_lut_relabelled.append((in_frag, out_frag))
                        else:
                            out_frag = relabeled_nodes.get(out_frag, in_frag)
                            relabeled_nodes[out_frag] = in_frag
                            mws_lut_relabelled.append((in_frag, out_frag))

                mws_mapping = [list(x) for x in zip(*mws_lut)]
                inputs = np.array(list(inputs), dtype=int)
                if len(mws_lut) > 0:
                    outputs = replace_values(
                        inputs,
                        np.array(mws_mapping[0], dtype=int),
                        np.array(mws_mapping[1], dtype=int),
                    )
                else:
                    outputs = inputs

                new_frag_to_seg_mapping = {
                    int(in_frag): int(out_frag)
                    for in_frag, out_frag in zip(inputs, outputs)
                    if out_frag is not None
                }
                new_seg_to_frag_mapping = {}
                for in_frag, out_frag in new_frag_to_seg_mapping.items():
                    in_group = new_seg_to_frag_mapping.setdefault(int(out_frag), set())
                    in_group.add(int(in_frag))

                # save the lut to a temporary file for this block
                block_lut = LUT(
                    path=f"{tmp_path}/{'-'.join([str(o) for o in block.write_roi.offset])}-lut"
                )
                lut = np.array([inputs, outputs])
                block_lut.save(lut, edges=edges)

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

                assert isinstance(total_lut, np.ndarray), "LUTs failed to load"

                frag_seg_mapping: dict[int, int] = {
                    int(k): int(v) for k, v in total_lut.T
                }
                seg_frag_mapping: dict[int, set[int]] = {}
                for in_frag, out_frag in frag_seg_mapping.items():
                    in_group = seg_frag_mapping.setdefault(out_frag, set())
                    in_group.add(in_frag)

                out_graph = out_rag_provider.read_graph(block.read_roi)

                for out_seg, in_frags in new_seg_to_frag_mapping.items():
                    agglomerated_attrs = {
                        "size": sum(
                            [graph.nodes[in_frag]["size"] for in_frag in in_frags]
                        )
                    }
                    agglomerated_attrs["position"] = (
                        sum(
                            [
                                np.array(graph.nodes[in_frag]["position"], dtype=float)
                                * graph.nodes[in_frag]["size"]
                                for in_frag in in_frags
                                if in_frag in graph.nodes
                            ],
                            start=np.array((0,) * block.read_roi.dims, dtype=float),
                        )
                        / agglomerated_attrs["size"]
                    )

                    out_graph.add_node(int(out_seg), **agglomerated_attrs)

                edges_to_agglomerate = {}
                for u, v in graph.edges():
                    if (
                        (u in new_frag_to_seg_mapping or v in new_frag_to_seg_mapping)
                        and u in frag_seg_mapping
                        and v in frag_seg_mapping
                    ):
                        assert graph.nodes[u].get("size", None) is not None
                        assert graph.nodes[v].get("size", None) is not None
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
                    agglomerated_edge_attrs = {}
                    for weight_attr in self.weights.keys():
                        # switch to edge size
                        magnitudes = [
                            graph.edges[edge][f"{weight_attr}__size"]
                            for edge in edges
                            if weight_attr in graph.edges[edge]
                        ]
                        weights = [
                            graph.edges[edge][weight_attr]
                            for edge in edges
                            if weight_attr in graph.edges[edge]
                        ]
                        magnitudes_and_weights = [
                            (magnitude, weight * magnitude)
                            for weight, magnitude in zip(weights, magnitudes)
                            if weight is not None
                        ]
                        if len(magnitudes_and_weights) > 0:
                            magnitudes, weights = zip(*magnitudes_and_weights)
                            agglomerated_edge_attrs[weight_attr] = sum(weights) / sum(
                                magnitudes
                            )
                            agglomerated_edge_attrs[weight_attr + "__size"] = sum(
                                magnitudes
                            )
                    out_graph.add_edge(
                        seg_u,
                        seg_v,
                        **agglomerated_edge_attrs,
                    )

                out_rag_provider.write_graph(
                    out_graph, block.write_roi, both_sides=True
                )

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

                # save the lut to a temporary file for this block
                block_lut = LUT(
                    path=f"{tmp_path}/{'-'.join([str(o) for o in block.write_roi.offset])}-lut"
                )
                lut = np.array([inputs, outputs])
                block_lut.save(lut, edges=edges)

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
                assert isinstance(total_lut, np.ndarray), "LUTs failed to load"

                frag_seg_mapping: dict[int, int] = {
                    int(k): int(v) for k, v in total_lut.T
                }
                seg_frag_mapping: dict[int, set[int]] = {}
                for in_frag, out_frag in frag_seg_mapping.items():
                    in_group = seg_frag_mapping.setdefault(out_frag, set())
                    in_group.add(in_frag)

                out_graph = out_rag_provider.read_graph(block.read_roi)
                assert out_graph.number_of_nodes() == 0, out_graph.number_of_nodes

                for out_frag in np.unique(outputs):
                    if out_frag is not None and out_frag not in out_graph.nodes:
                        in_group = seg_frag_mapping[out_frag]

                        agglomerated_attrs = {
                            "size": sum(
                                [graph.nodes[in_frag]["size"] for in_frag in in_group]
                            )
                        }
                        agglomerated_attrs["position"] = (
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
                            / agglomerated_attrs["size"]
                        )

                        out_graph.add_node(int(out_frag), **agglomerated_attrs)

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

            raise ValueError("Not finished yet")
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
