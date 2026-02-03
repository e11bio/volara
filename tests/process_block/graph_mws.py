from pathlib import Path

import daisy
import numpy as np
import pytest
from funlib.geometry import Roi

from volara.blockwise import (
    GraphMWS,
)
from volara.dbs import SQLite
from volara.lut import LUT


@pytest.mark.parametrize("y_bias", [0.5, -0.5])
def test_graph_mws(tmpdir, y_bias: float):
    tmpdir = Path(tmpdir)

    block = daisy.Block(
        total_roi=Roi((0, 0), (10, 10)),
        read_roi=Roi((0, 0), (10, 10)),
        write_roi=Roi((0, 0), (10, 10)),
    )

    data_dir = tmpdir / "test_data.zarr"
    if not data_dir.exists():
        data_dir.mkdir()
    db_config = SQLite(
        path=tmpdir / "test_data.zarr" / "db.sqlite",
        node_attrs={"raw_intensity": 1},
        edge_attrs={
            "y_aff": "float",
        },
        ndim=2,
    )
    db_config.init()

    config = GraphMWS(
        roi=block.read_roi,
        db=db_config,
        lut=LUT(path=tmpdir / "fragment_segment_lut.npz"),
        weights={"y_aff": (1, y_bias)},
    )

    db = config.db.open("r+")
    graph = db.read_graph()
    graph.add_node(1, position=(4, 2), size=600, raw_intensity=(0.1,))
    graph.add_node(2, position=(4, 7), size=400, raw_intensity=(0.1,))
    graph.add_edge(
        1,
        2,
        y_aff=0,
    )
    db.write_graph(graph)

    with config.process_block_func() as process_block:
        process_block(block)

    lut = config.lut.load()
    assert lut is not None
    fragments, segments = lut
    assert len(np.unique(fragments)) == 2, fragments
    assert len(np.unique(segments)) == 1 + (y_bias < 0), segments
