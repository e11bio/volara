import os
import shutil

import daisy
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.blockwise.lambda_task import LambdaTask
from volara.datasets import Labels, Raw
from volara.workers import LSFWorker, LocalWorker, SlurmWorker

slurm_available = pytest.mark.skipif(
    shutil.which("sbatch") is None or os.environ.get("VOLARA_SLURM_TEST_QUEUE") is None,
    reason="sbatch not found or VOLARA_SLURM_TEST_QUEUE not set",
)
lsf_available = pytest.mark.skipif(
    shutil.which("bsub") is None or os.environ.get("VOLARA_LSF_TEST_QUEUE") is None,
    reason="bsub not found or VOLARA_LSF_TEST_QUEUE not set",
)


def test_lambda_task_init_and_drop(zarr_2d, tmp_path):
    """init() creates output zarr, drop_artifacts() removes it.

    pytest tests/test_lambda_task.py::test_lambda_task_init_and_drop
    """
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
    """Lambda function is applied correctly to a single block.

    pytest tests/test_lambda_task.py::test_lambda_task_basic
    """
    raw_path, data = zarr_2d
    out_path = tmp_path / "test.zarr" / "out"
    task = LambdaTask(
        in_data=Raw(store=raw_path),
        out_data=Labels(store=out_path),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(10, 10),
        out_array_dtype=np.dtype(np.uint8),
    )
    task.init()

    with task.process_block_func() as process_block:
        process_block(block_2d)

    result = task.out_data.array("r")[:]
    expected = (data > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


def test_lambda_task_multiblock(tmp_path):
    """Two blocks tile a 20x10 array and produce a correct full-coverage output.

    pytest tests/test_lambda_task.py::test_lambda_task_multiblock
    """
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


def test_lambda_task_multiprocessing_local_worker(tmp_path):
    """LambdaTask produces correct output when dispatched to LocalWorker subprocesses.

    pytest tests/test_lambda_task.py::test_lambda_task_multiprocessing_local_worker
    """
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
        num_workers=2,
        worker_config=LocalWorker(),
    )
    task.run_blockwise(multiprocessing=True)

    result = task.out_data.array("r")[:]
    expected = (data > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)


@slurm_available
def test_lambda_task_multiprocessing_slurm_worker(tmp_path):
    """LambdaTask produces correct output when dispatched to SlurmWorker subprocesses.

    Can run this test with a Slurm worker if you have access to a Slurm cluster and set up a queue for testing:
    VOLARA_SLURM_TEST_QUEUE=<queue> pytest tests/test_lambda_task.py::test_lambda_task_multiprocessing_slurm_worker
    """
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
        num_workers=2,
        worker_config=SlurmWorker(queue=os.environ["VOLARA_SLURM_TEST_QUEUE"]),
    )
    task.run_blockwise(multiprocessing=True)

    result = task.out_data.array("r")[:]
    expected = (data > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)


@lsf_available
def test_lambda_task_multiprocessing_lsf_worker(tmp_path):
    """LambdaTask produces correct output when dispatched to LSFWorker subprocesses.

    Can run this test with an LSF worker if you have access to an LSF cluster and set up a queue for testing:
    VOLARA_LSF_TEST_QUEUE=<queue> pytest tests/test_lambda_task.py::test_lambda_task_multiprocessing_lsf_worker
    """
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
        num_workers=2,
        worker_config=LSFWorker(queue=os.environ["VOLARA_LSF_TEST_QUEUE"]),
    )
    task.run_blockwise(multiprocessing=True)

    result = task.out_data.array("r")[:]
    expected = (data > 0.5).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_lambda_task_properties(zarr_2d, tmp_path):
    """Task properties are derived correctly from inputs.

    pytest tests/test_lambda_task.py::test_lambda_task_properties
    """
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
