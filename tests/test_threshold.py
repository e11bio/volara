import numpy as np
from funlib.geometry import Coordinate

from volara.blockwise import Threshold
from volara.datasets import Labels, Raw


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
    task.run_blockwise(multiprocessing=False)
    assert mask_path.exists()
    assert (task.meta_dir / "blocks_done").exists()

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


def test_threshold_multiblock(gradient_20x10, blocks_20x10, tmp_path):
    """Two blocks tile a 20x10 array and produce a correct full-coverage mask."""
    in_path, data = gradient_20x10

    mask_path = tmp_path / "data.zarr" / "mask"
    task = Threshold(
        in_data=Raw(store=in_path),
        mask=Labels(store=mask_path),
        threshold=0.5,
        block_size=Coordinate(10, 10),
    )
    task.init()

    with task.process_block_func() as process_block:
        for block in blocks_20x10:
            process_block(block)

    result = task.mask.array("r")[:]
    expected = (data > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)
