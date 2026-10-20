import inspect
import os
import shutil

import daisy
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.blockwise.lambda_task import LambdaTask
from volara.datasets import Labels, Raw
from volara.workers import LocalWorker, LSFWorker, SlurmWorker

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


def test_lambda_task_init_out_false_skips_prepare_and_drop(tmp_path):
    """With init_out=False, init() does not create out_data and drop_artifacts() is a no-op.

    pytest tests/test_lambda_task.py::test_lambda_task_init_out_false_skips_prepare_and_drop
    """
    data = np.random.default_rng(0).random((10, 10), dtype=np.float32)
    in_path = tmp_path / "data.zarr" / "raw"
    prepare_ds(
        in_path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )[:] = data

    # Pre-create the output dataset (simulating an already-existing volume)
    out_path = tmp_path / "data.zarr" / "out"
    out_data_arr = prepare_ds(
        out_path,
        shape=(20, 20),
        voxel_size=Coordinate(1, 1),
        dtype=np.uint8,
        mode="w",
    )
    out_data_arr[:] = 0

    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=out_path),
        init_out=False,
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(10, 10),
    )

    # init() should not recreate or overwrite the output
    task.init()
    result_shape = task.out_data.array("r").shape
    assert result_shape == (20, 20), (
        "init_out=False should not recreate the output array"
    )

    # drop_artifacts() should not remove the output
    task.drop_artifacts()
    assert out_path.exists(), "init_out=False should not drop the output array"


def test_lambda_task_default_block_size_from_out_data(tmp_path):
    """When block_size is None, write_size defaults to out_data's chunk shape.

    pytest tests/test_lambda_task.py::test_lambda_task_default_block_size_from_out_data
    """
    data = np.random.default_rng(0).random((20, 20), dtype=np.float32)
    in_path = tmp_path / "data.zarr" / "raw"
    prepare_ds(
        in_path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )[:] = data

    # Create output with a specific chunk shape (5, 5)
    out_path = tmp_path / "data.zarr" / "out"
    prepare_ds(
        out_path,
        shape=(20, 20),
        voxel_size=Coordinate(1, 1),
        dtype=np.uint8,
        chunk_shape=Coordinate(5, 5),
        mode="w",
    )

    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=out_path),
        init_out=False,
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        # block_size intentionally omitted
    )

    # write_size should be chunk_shape * voxel_size = (5, 5) * (1, 1) = (5, 5)
    assert task.write_size == Coordinate(5, 5)


def test_lambda_task_skips_non_overlapping_blocks(tmp_path):
    """Blocks outside in_data.roi are skipped without error.

    pytest tests/test_lambda_task.py::test_lambda_task_skips_non_overlapping_blocks
    """
    # in_data covers (0,0) to (10,10)
    data = np.ones((10, 10), dtype=np.float32)
    in_path = tmp_path / "data.zarr" / "raw"
    prepare_ds(
        in_path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )[:] = data

    # out_data covers (0,0) to (20,20) -- larger than in_data
    out_path = tmp_path / "data.zarr" / "out"
    prepare_ds(
        out_path,
        shape=(20, 20),
        voxel_size=Coordinate(1, 1),
        dtype=np.uint8,
        mode="w",
    )

    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=out_path),
        init_out=False,
        lambda_func=lambda x: (x * 2).astype(np.uint8),
        block_size=Coordinate(10, 10),
    )
    task.init()

    # Block entirely outside in_data.roi -- should be skipped
    non_overlapping_block = daisy.Block(
        total_roi=Roi((0, 0), (20, 20)),
        read_roi=Roi((10, 10), (10, 10)),
        write_roi=Roi((10, 10), (10, 10)),
    )

    with task.process_block_func() as process_block:
        process_block(non_overlapping_block)

    # The region should remain zeros (untouched)
    result = task.out_data.array("r")[Roi((10, 10), (10, 10))]
    np.testing.assert_array_equal(result, np.zeros((10, 10), dtype=np.uint8))


_FUNLIB_SHARDS = "shard_shape" in inspect.signature(prepare_ds).parameters
needs_sharding = pytest.mark.skipif(
    not _FUNLIB_SHARDS,
    reason="funlib.persistence here has no prepare_ds(shard_shape=...) (funkelab/funlib.persistence#40)",
)


def _threshold_input(tmp_path):
    data = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    in_path = tmp_path / "data.zarr" / "raw"
    prepare_ds(
        in_path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )[:] = data
    return in_path, data


@needs_sharding
def test_lambda_task_blocks_default_to_whole_shards(tmp_path):
    """With no block_size the block is the declared shard, so every block is one
    plain put and no two blocks share a stored object."""
    in_path, data = _threshold_input(tmp_path)
    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(
            store=tmp_path / "data.zarr" / "out",
            shard_shape=(16, 16),
            chunk_shape=(8, 8),
        ),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
    )
    assert task.write_size == Coordinate(16, 16)
    assert task.block_grid_shape == Coordinate(4, 4)

    task.run_blockwise(multiprocessing=False)

    out = task.out_data.array("r")
    assert out.shard_shape == Coordinate(16, 16)  # ty: ignore[unresolved-attribute]
    assert out.chunk_shape == Coordinate(8, 8)
    np.testing.assert_array_equal(out[:], (data > 0.5).astype(np.uint8))


@needs_sharding
def test_lambda_task_blocks_follow_an_existing_sharded_output(tmp_path):
    """`init_out=False` sizes blocks off the destination: its shard, never its inner chunk."""
    in_path, data = _threshold_input(tmp_path)
    out_path = tmp_path / "data.zarr" / "out"
    prepare_ds(
        out_path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        chunk_shape=(8, 8),
        shard_shape=(32, 32),
        dtype=np.uint8,
        mode="w",
    )
    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=out_path),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        init_out=False,
    )
    assert task.write_size == Coordinate(32, 32)


@needs_sharding
def test_lambda_task_refuses_a_block_size_that_splits_a_shard(tmp_path):
    in_path, _ = _threshold_input(tmp_path)
    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(
            store=tmp_path / "data.zarr" / "out",
            shard_shape=(16, 16),
            chunk_shape=(8, 8),
        ),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(8, 8),
    )
    with pytest.raises(ValueError, match="not a whole number of shards"):
        task.init()
    assert not (tmp_path / "data.zarr" / "out").exists()


@needs_sharding
def test_lambda_task_concurrent_workers_write_whole_shards(tmp_path):
    """Four LocalWorker processes writing 16 shard-sized blocks into one sharded
    array: with blocks aligned to shards there is nothing to race over.

    pytest tests/test_lambda_task.py::test_lambda_task_concurrent_workers_write_whole_shards
    """
    in_path, data = _threshold_input(tmp_path)
    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(
            store=tmp_path / "data.zarr" / "out",
            shard_shape=(16, 16),
            chunk_shape=(8, 8),
        ),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        num_workers=4,
        worker_config=LocalWorker(),
    )
    states = task.run_blockwise(multiprocessing=True)
    assert states[task.task_name].failed_count == 0
    np.testing.assert_array_equal(
        task.out_data.array("r")[:], (data > 0.5).astype(np.uint8)
    )
