from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

import mwatershed as mws
import numpy as np
from funlib.geometry import Coordinate, Roi
from pydantic import Field

from ..datasets import Labels
from ..dbs import PostgreSQL, SQLite
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask

DB = Annotated[
    PostgreSQL | SQLite,
    Field(discriminator="db_type"),
]


class GlobalMWS(BlockwiseTask):
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
            graph = rag_provider.read_graph(block.write_roi)

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
