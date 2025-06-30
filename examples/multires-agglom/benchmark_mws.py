# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mwatershed",
# ]
#
# [tool.uv.sources]
# mwatershed = { git = "https://github.com/pattonw/mwatershed" }
#
# ///

import mwatershed as mws
import random
import time

NUM_EDGES = [1_000_000, 10_000_000, 100_000_000]

for num_edges in NUM_EDGES:
    t1 = time.time()
    edges = [
        (
            random.random() > 0.5,
            random.randint(0, num_edges // 10),
            random.randint(0, num_edges // 10),
        )
        for _ in range(num_edges)
    ]
    t2 = time.time()
    mws.cluster(edges)
    t3 = time.time()
    print(
        f"NUM_EDGES: {num_edges}\n",
        f"TIME_TO_GENERATE: {t2 - t1:.2f} seconds\n",
        f"TIME_TO_CLUSTER: {t3 - t2:.2f} seconds\n",
    )
