import daisy
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import Threshold
from volara.datasets import Labels, Raw


def test_threshold_init_and_drop(zarr_2d, tmp_path):
    """init() creates output zarr, drop_artifacts() removes it."""
    raw_path, _ = zarr_2d
    mask_path = tmp_path / "test.zarr" / "mask"
    task = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=mask_path),
        threshold=0.5,
        block_size=Coordinate(10, 10),
    )
    task.init()
    assert mask_path.exists()
    task.drop_artifacts()
    assert not mask_path.exists()


def test_threshold_drop_outputs_false_keeps_output(zarr_2d, tmp_path):
    """drop(drop_outputs=False) resets the block-done cache but keeps the output.

    drop() with the default (True) still deletes it.
    """
    raw_path, _ = zarr_2d
    mask_path = tmp_path / "test.zarr" / "mask"
    task = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=mask_path),
        threshold=0.5,
        block_size=Coordinate(10, 10),
    )
    task.init()
    task.init_block_array()
    assert mask_path.exists()
    assert task.block_ds.exists()

    task.drop(drop_outputs=False)
    assert mask_path.exists(), "drop_outputs=False must not delete the output"
    assert not task.meta_dir.exists()

    task.drop()
    assert not mask_path.exists()


def test_threshold_basic(zarr_2d, block_2d, tmp_path):
    """Threshold at 0.5 produces correct binary mask."""
    raw_path, data = zarr_2d
    mask_path = tmp_path / "test.zarr" / "mask"
    task = Threshold(
        in_data=Raw(store=raw_path),
        mask=Labels(store=mask_path),
        threshold=0.5,
        block_size=Coordinate(10, 10),
    )
    task.init()

    with task.process_block_func() as process_block:
        process_block(block_2d)

    result = task.mask.array("r")[:]
    expected = (data > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_threshold_multiblock(tmp_path):
    """Two blocks tile a 20x10 array and produce a correct full-coverage mask."""
    data = np.linspace(0, 1, 200, dtype=np.float32).reshape(20, 10)
    in_path = tmp_path / "data.zarr" / "raw"
    prepare_ds(
        in_path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )[:] = data

    mask_path = tmp_path / "data.zarr" / "mask"
    task = Threshold(
        in_data=Raw(store=in_path),
        mask=Labels(store=mask_path),
        threshold=0.5,
        block_size=Coordinate(10, 10),
    )
    task.init()

    block1 = daisy.Block(
        total_roi=Roi((0, 0), (20, 10)),
        read_roi=Roi((0, 0), (10, 10)),
        write_roi=Roi((0, 0), (10, 10)),
    )
    block2 = daisy.Block(
        total_roi=Roi((0, 0), (20, 10)),
        read_roi=Roi((10, 0), (10, 10)),
        write_roi=Roi((10, 0), (10, 10)),
    )

    with task.process_block_func() as process_block:
        process_block(block1)
        process_block(block2)

    result = task.mask.array("r")[:]
    expected = (data > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)
