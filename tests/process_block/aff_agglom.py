from pathlib import Path

import daisy
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import (
    AffAgglom,
)
from volara.datasets import Affs, Labels
from volara.dbs import SQLite


def test_aff_agglom(tmpdir):
    tmpdir = Path(tmpdir)

    # define the block to process
    block = daisy.Block(
        total_roi=Roi((0, 0), (10, 10)),
        read_roi=Roi((0, 0), (10, 10)),
        write_roi=Roi((0, 0), (10, 10)),
    )

    # Create the database to containe edge affinities
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

    # create fragment labels array
    frags_data = np.zeros((10, 10), dtype=np.uint32)
    frags_data[:, :] = np.arange(1, 11)[:, None]  # horizontal stripes
    frags_arr = prepare_ds(
        tmpdir / "test_data.zarr" / "frags",
        shape=frags_data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=frags_data.dtype,
        mode="w",
    )
    frags_arr[:] = frags_data
    frags = Labels(store=tmpdir / "test_data.zarr" / "frags")

    # add fragment nodes to db
    db = db_config.open("r+")
    empty_graph = db.read_graph()
    for i in range(1, 11):
        empty_graph.add_node(i, position=(i, 5), size=1, raw_intensity=(i,))
    db.write_graph(empty_graph)

    # create affs data
    affs_data = np.zeros((1, 10, 10), dtype=np.uint32)
    affs_data[0, ::2, :] = np.ones((5, 1))  # vertical affs 1 in every other row
    affs_arr = prepare_ds(
        tmpdir / "test_data.zarr" / "affs",
        shape=affs_data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=affs_data.dtype,
        mode="w",
    )
    affs_arr[:] = affs_data
    affs = Affs(
        store=tmpdir / "test_data.zarr" / "affs", neighborhood=[Coordinate(1, 0)]
    )

    # affs config
    aff_agglom_config = AffAgglom(
        db=db_config,
        frags_data=frags,
        affs_data=affs,
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        scores={"y_aff": [Coordinate(1, 0)]},
    )

    with aff_agglom_config.process_block_func() as process_block:
        process_block(block)

    # Check that we got the expected results
    db = db_config.open("r")
    g = db.read_graph(block.write_roi)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 9
    for u, v, data in g.edges(data=True):
        if u % 2 == 0:
            assert data["y_aff"] == 0.0
        else:
            assert data["y_aff"] == 1.0
