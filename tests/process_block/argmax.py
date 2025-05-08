from pathlib import Path

import daisy
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import (
    Argmax,
)
from volara.datasets import Labels, Raw


def test_argmax(tmpdir):
    tmpdir = Path(tmpdir)

    # Define the block to process
    block = daisy.Block(
        total_roi=Roi((0, 0), (10, 10)),
        read_roi=Roi((0, 0), (10, 10)),
        write_roi=Roi((0, 0), (10, 10)),
    )

    # create probabilities array
    probs_data = np.arange(1, 201, dtype=np.float32).reshape(2, 10, 10)
    probs_arr = prepare_ds(
        tmpdir / "test_data.zarr" / "probs",
        shape=probs_data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=probs_data.dtype,
        mode="w",
    )
    probs_arr[:] = probs_data
    probs = Raw(store=tmpdir / "test_data.zarr" / "probs")

    # create labels array
    prepare_ds(
        tmpdir / "test_data.zarr" / "labels",
        shape=probs_data[1:],
        voxel_size=Coordinate(1, 1),
        dtype=np.uint32,
        mode="w",
    )
    labels = Labels(store=tmpdir / "test_data.zarr" / "labels")

    # argmax config
    argmax_config = Argmax(
        probs_data=probs,
        sem_data=labels,
        block_size=Coordinate(10, 10),
    )

    with argmax_config.process_block_func() as process_block:
        process_block(block)

    assert np.isclose(labels.array("r")[:], np.ones((10, 10))).all()
