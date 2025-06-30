from volara.blockwise.graph_mws import IterativeGraphMWS
from volara.dbs import SQLite
from volara.lut import LUT
from funlib.geometry import Coordinate
import numpy as np
import time as time

import itertools

fragments_db = SQLite(
    path="fragments.sqlite",
    node_attrs={},
    edge_attrs={"aff": "float"},
)
super_fragments_db = SQLite(
    path="super_fragments.sqlite",
    node_attrs={},
    edge_attrs={"aff": "float"},
)
segments_db = SQLite(
    path="segments.sqlite",
    node_attrs={},
    edge_attrs={"aff": "float"},
)

lut1 = LUT(path="lut1")
lut2 = LUT(path="lut2")

full_roi_size = Coordinate(128, 128, 128)
full_roi = (Coordinate(0, 0, 0), Coordinate(full_roi_size))
block_size_round_1 = Coordinate(full_roi_size // 4)
block_size_round_2 = Coordinate(full_roi_size)

fragments_db.drop()
super_fragments_db.drop()
segments_db.drop()

frag_gdb = fragments_db.open("w")
initial_graph = frag_gdb.read_graph()


def pos_to_id(pos):
    return (
        pos[0] * full_roi_size[1] * full_roi_size[2]
        + pos[1] * full_roi_size[2]
        + pos[2]
    )


for position in itertools.product(
    range(0, full_roi_size[0]),
    range(0, full_roi_size[1]),
    range(0, full_roi_size[2]),
):
    node_id = pos_to_id(position)
    initial_graph.add_node(
        node_id,
        position=[p + 0.5 for p in position],
        size=1,
    )
    if position[0] > 0:
        z_neighbor = pos_to_id((position[0] - 1, position[1], position[2]))
        initial_graph.add_edge(
            node_id,
            z_neighbor,
            aff=-1.0,
        )
    if position[1] > 0:
        y_neighbor = pos_to_id((position[0], position[1] - 1, position[2]))
        initial_graph.add_edge(
            node_id,
            y_neighbor,
            aff=1.0,
        )
    if position[2] > 0:
        x_neighbor = pos_to_id((position[0], position[1], position[2] - 1))
        initial_graph.add_edge(
            node_id,
            x_neighbor,
            aff=1.0,
        )

frag_gdb.write_graph(initial_graph)

round_1 = IterativeGraphMWS(
    fragments_db=fragments_db,
    segments_db=super_fragments_db,
    lut=lut1,
    weights={"aff": (1.0, 0.0)},
    roi=full_roi,
    block_size=block_size_round_1,
)
round_2 = IterativeGraphMWS(
    fragments_db=super_fragments_db,
    segments_db=segments_db,
    lut=lut2,
    weights={"aff": (1.0, 0.0)},
    roi=full_roi,
    block_size=block_size_round_2,
)

pipeline = round_1 + round_2
pipeline.drop()

t1 = time.time()
pipeline.run_blockwise(multiprocessing=False)
t2 = time.time()
print(f"Pipeline completed in {t2 - t1:.2f} seconds")

mapping1 = lut1.load()

num_nodes = np.prod(full_roi_size)
frag_size = full_roi_size[1] // 4 * full_roi_size[2] // 4

assert mapping1 is not None
assert mapping1.shape == (2, num_nodes), mapping1.shape
assert len(np.unique(mapping1[0, :])) == num_nodes, (
    f"Expected {num_nodes} unique IDs, got {len(np.unique(mapping1[0, :]))}"
)
assert len(np.unique(mapping1[1, :])) == num_nodes // frag_size, (
    f"Expected {num_nodes // frag_size} unique segments, got {len(np.unique(mapping1[1, :]))}"
)

mapping2 = lut2.load()
assert mapping2 is not None
assert len(np.unique(mapping2[0, :])) == num_nodes // frag_size, (
    f"Expected {num_nodes // frag_size} unique IDs, got {len(np.unique(mapping2[0, :]))}"
)
assert len(np.unique(mapping2[1, :])) == full_roi_size[0], (
    f"Expected {full_roi_size[0]} unique segments, got {len(np.unique(mapping2[1, :]))}"
)

mapping = (lut1 + lut2).load_iterated()
