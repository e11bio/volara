# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "pympler",
#     "volara",
# ]
#
# [tool.uv.sources]
# volara = { git = "https://github.com/e11bio/volara" }
# ///
from volara.dbs import SQLite
import networkx as nx
import random
import time
import itertools
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from pympler import asizeof

from funlib.geometry import Coordinate, Roi

if not Path("benchmark_volara_graphs.pickle").exists():
    NODE_COUNTS = [
        2_000,
        64_000,
        1_024_000,
        # 2_048_000,
    ]
    memory_usage = []
    write_full_graph_times = []
    read_full_graph_times = []
    update_block_times = []
    read_block_times = []

    for NUM_NODES in NODE_COUNTS:
        db = SQLite(
            path="test.sqlite",
            node_attrs={"skeleton_id": "int", "color": 3},
            edge_attrs={"aff": "float"},
        )
        db.drop()

        gdb = db.open("w")

        # p = 5 / num_nodes gives approximately 2.5 nodes per edge
        random_graph: nx.Graph = nx.fast_gnp_random_graph(
            n=NUM_NODES, p=5 / NUM_NODES, directed=False, seed=1
        )

        for node in random_graph.nodes:
            random_graph.nodes[node]["color"] = tuple(
                random.randint(0, 255) for _ in range(3)
            )
            random_graph.nodes[node]["position"] = tuple(
                random.random() * 1000 for _ in range(3)
            )

        for i, cc in enumerate(nx.connected_components(random_graph)):
            for node in cc:
                random_graph.nodes[node]["skeleton_id"] = i + 1

        for edge in random_graph.edges:
            random_graph.edges[edge]["aff"] = random.random()

        # write the full graph
        t1 = time.time()
        gdb.write_graph(random_graph)
        t_write = time.time() - t1

        # memmory = Path("test.sqlite").stat().st_size / 1024 / 1024 / 1024
        memmory = asizeof.asizeof(random_graph) / 1024 / 1024 / 1024

        # read the full graph
        t1 = time.time()
        random_graph_2 = gdb.read_graph()
        t_read = time.time() - t1
        assert random_graph_2.number_of_nodes() == random_graph.number_of_nodes()
        assert random_graph_2.number_of_edges() == random_graph.number_of_edges()

        # read the full graph in blocks
        # (some edges will be missing if crossing block boundaries, so only check num nodes)
        block_size = Coordinate(500, 500, 500)
        base_block_roi = Roi((0, 0, 0), block_size)

        num_nodes = 0
        t_read_blocks = time.time()
        for a, b, c in itertools.product(range(2), repeat=3):
            block_roi = base_block_roi + Coordinate(a, b, c) * block_size
            t1 = time.time()
            block_g = gdb.read_graph(block_roi)
            t_read_blocks += time.time() - t1
            for node in list(block_g.nodes):
                if "position" not in block_g.nodes[node]:
                    block_g.remove_node(node)
            num_nodes += block_g.number_of_nodes()
        t_read_blocks = time.time() - t1
        assert num_nodes == random_graph.number_of_nodes(), (
            num_nodes,
            random_graph.number_of_nodes(),
        )

        # write graph in blocks
        num_nodes = 0
        num_edges = 0
        t_write_blocks = 0
        for a, b, c in itertools.product(range(2), repeat=3):
            block_roi = base_block_roi + (Coordinate(a, b, c)) * block_size
            block_g = gdb.read_graph(block_roi)
            for node in list(block_g.nodes):
                if "position" not in block_g.nodes[node]:
                    block_g.remove_node(node)
                else:
                    block_g.nodes[node]["position"] = (
                        Coordinate(block_g.nodes[node]["position"]) + block_size * 2
                    )
            mapping = {x: x + NUM_NODES for x in block_g.nodes}
            nx.relabel.relabel_nodes(block_g, mapping, copy=False)

            t1 = time.time()
            gdb.write_graph(block_g, roi=block_roi + block_size * 2)
            t_write_blocks += time.time() - t1
            num_nodes += block_g.number_of_nodes()
            num_edges += block_g.number_of_edges()

        random_graph_3 = gdb.read_graph()
        assert random_graph_3.number_of_nodes() == random_graph.number_of_nodes() * 2

        print(
            f"NUM_NODES: {NUM_NODES}\n",
            f"NUM_EDGES: {random_graph.number_of_edges()}\n",
            f"Memmory: {memmory:.2f} GB\n",
            f"t_write_full: {t_write:.2f}\n",
            f"t_read_full: {t_read:.2f}\n",
            f"t_write_blocks: {t_write_blocks:.2f}\n",
            f"t_read_blocks: {t_read_blocks:.2f}\n",
        )

        memory_usage.append(memmory)
        write_full_graph_times.append(t_write)
        read_full_graph_times.append(t_read)
        update_block_times.append(t_write_blocks)
        read_block_times.append(t_read_blocks)

    pickle.dump(
        {
            "NODE_COUNTS": NODE_COUNTS,
            "memory_usage": memory_usage,
            "write_full_graph_times": write_full_graph_times,
            "read_full_graph_times": read_full_graph_times,
            "update_block_times": update_block_times,
            "read_block_times": read_block_times,
        },
        open("benchmark_volara_graphs.pickle", "wb"),
    )

else:
    data_dict = pickle.load(open("benchmark_volara_graphs.pickle", "rb"))
    (
        NODE_COUNTS,
        memory_usage,
        write_full_graph_times,
        read_full_graph_times,
        update_block_times,
        read_block_times,
    ) = (
        data_dict["NODE_COUNTS"],
        data_dict["memory_usage"],
        data_dict["write_full_graph_times"],
        data_dict["read_full_graph_times"],
        data_dict["update_block_times"],
        data_dict["read_block_times"],
    )


plt.loglog(
    NODE_COUNTS,
    memory_usage,
    label="Memory usage",
)
plt.loglog(
    NODE_COUNTS,
    write_full_graph_times,
    label="Write full graph",
)
plt.loglog(
    NODE_COUNTS,
    read_full_graph_times,
    label="Read full graph",
)
plt.loglog(
    NODE_COUNTS,
    update_block_times,
    label="Update block",
)
plt.loglog(
    NODE_COUNTS,
    read_block_times,
    label="Read block",
)
plt.xlabel("Number of nodes")
plt.ylabel("Time (s) / Memory (GB)")
plt.legend()
plt.show()
