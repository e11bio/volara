import importlib

import daisy
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.datasets import Labels, Raw

# `lambda` is a Python keyword; `from volara.blockwise.lambda import` is a syntax error
LambdaTask = importlib.import_module("volara.blockwise.lambda").LambdaTask


def test_lambda_task_init_and_drop(zarr_2d, tmp_path):
    """init() creates output zarr, drop_artifacts() removes it."""
    raw_path, _ = zarr_2d
    out_path = tmp_path / "test.zarr" / "out"
    task = LambdaTask(
        in_data=Raw(store=raw_path),
        out_data=Labels(store=out_path),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(10, 10),
    )
    task.init()
    assert out_path.exists()
    task.drop_artifacts()
    assert not out_path.exists()


def test_lambda_task_basic(zarr_2d, block_2d, tmp_path):
    """Lambda function is applied correctly to a single block."""
    raw_path, data = zarr_2d
    out_path = tmp_path / "test.zarr" / "out"
    task = LambdaTask(
        in_data=Raw(store=raw_path),
        out_data=Labels(store=out_path),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(10, 10),
    )
    task.init()

    with task.process_block_func() as process_block:
        process_block(block_2d)

    result = task.out_data.array("r")[:]
    expected = (data > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_lambda_task_multiblock(tmp_path):
    """Two blocks tile a 20x10 array and produce a correct full-coverage output."""
    data = np.linspace(0, 1, 200, dtype=np.float32).reshape(20, 10)
    in_path = tmp_path / "data.zarr" / "raw"
    prepare_ds(
        in_path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )[:] = data

    out_path = tmp_path / "data.zarr" / "out"
    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=out_path),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
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

    result = task.out_data.array("r")[:]
    expected = (data > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_lambda_task_properties(zarr_2d, tmp_path):
    """Task properties are derived correctly from inputs."""
    raw_path, _ = zarr_2d
    out_path = tmp_path / "test.zarr" / "out"
    out_data = Labels(store=out_path)
    task = LambdaTask(
        in_data=Raw(store=raw_path),
        out_data=out_data,
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(5, 5),
    )

    assert task.task_name == f"{out_data.name}-lambda"
    assert task.context_size == Coordinate(0, 0)
    assert task.write_size == Coordinate(5, 5)
    assert task.output_datasets == [out_data]
    assert task.write_roi == Raw(store=raw_path).array("r").roi
