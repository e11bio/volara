"""Tests for the daisy ``{task_id: TaskState}`` map returned by
``BlockwiseTask.run_blockwise`` and ``Pipeline.run_blockwise``.

Before this behavior existed, the multiprocessing path returned
``daisy.run_blockwise``'s collapsed bool -- which reports ``True`` even when
blocks fail, because ``TaskState.is_done()`` counts failed/orphaned blocks as
"done" -- and ``Pipeline.run_blockwise`` returned ``None``. These tests pin the
map return on both the serial and multiprocessing paths so a caller can see
``failed_count`` / ``completed_count`` and react to a partial run.
"""

import numpy as np
import pytest

# daisy v2 is a Rust extension (not buildable/installed in every env). Skip this whole
# module unless the v2 surface is importable -- `daisy.v2` exists only on v2, so it's a
# clean v2-only sentinel. Runtime validation of these assertions is deferred to a
# v2-built CI environment.
pytest.importorskip("daisy.v2")

from daisy import TaskState  # noqa: E402  (after importorskip guard)
from funlib.geometry import Coordinate  # noqa: E402
from funlib.persistence.arrays import prepare_ds  # noqa: E402

from volara.blockwise.lambda_task import LambdaTask  # noqa: E402
from volara.datasets import Labels, Raw  # noqa: E402


def _write_raw(path, data):
    prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )[:] = data
    return path


# A lambda that raises when a block's data exceeds a threshold. Passed through
# volara's cloudpickle-based PydanticCallable, so it must be an expression --
# hence the generator ``.throw`` idiom to raise inside the conditional.
_FAIL_ON_HIGH = lambda x: (  # noqa: E731
    (x > 0.5).astype(np.uint8)
    if x.max() <= 1.5
    else (_ for _ in ()).throw(RuntimeError("boom: block deliberately failed"))
)


def test_task_serial_returns_state_map(tmp_path):
    """multiprocessing=False returns {task_id: TaskState}; all blocks complete.

    pytest tests/test_run_blockwise_states.py::test_task_serial_returns_state_map
    """
    # 20x10 tiled into two 10x10 blocks
    data = np.linspace(0, 1, 200, dtype=np.float32).reshape(20, 10)
    in_path = _write_raw(tmp_path / "data.zarr" / "raw", data)

    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=tmp_path / "data.zarr" / "out"),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(10, 10),
    )

    states = task.run_blockwise(multiprocessing=False)

    assert isinstance(states, dict)
    assert task.task_name in states
    state = states[task.task_name]
    assert isinstance(state, TaskState)
    assert state.total_block_count == 2
    assert state.completed_count == 2
    assert state.failed_count == 0
    assert state.orphaned_count == 0
    assert state.is_done()


def test_task_multiprocessing_surfaces_failed_blocks(tmp_path):
    """A worker function that raises on one block -> failed_count > 0.

    This is the whole point of the PR: the returned map surfaces the failure,
    whereas the old bool return would have reported success (is_done() is True
    even with a failed block).

    pytest tests/test_run_blockwise_states.py::test_task_multiprocessing_surfaces_failed_blocks
    """
    # block 0 (rows 0..10) is all 0.0 -> succeeds
    # block 1 (rows 10..20) is all 2.0 -> lambda raises
    data = np.zeros((20, 10), dtype=np.float32)
    data[10:, :] = 2.0
    in_path = _write_raw(tmp_path / "data.zarr" / "raw", data)

    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=tmp_path / "data.zarr" / "out"),
        lambda_func=_FAIL_ON_HIGH,
        block_size=Coordinate(10, 10),
        num_workers=1,
    )

    states = task.run_blockwise(multiprocessing=True)

    assert isinstance(states, dict)
    state = states[task.task_name]
    assert isinstance(state, TaskState)
    assert state.total_block_count == 2
    assert state.failed_count > 0
    assert state.completed_count == 1
    # is_done() counts failed blocks as "done" -- the reason a bool return
    # cannot distinguish a clean run from one with failures.
    assert state.is_done()


def test_pipeline_serial_returns_merged_state_map(tmp_path):
    """A sequential (``+``) Pipeline returns a merged map covering both tasks.

    Previously Pipeline.run_blockwise returned None.

    pytest tests/test_run_blockwise_states.py::test_pipeline_serial_returns_merged_state_map
    """
    data = np.linspace(0, 1, 200, dtype=np.float32).reshape(20, 10)
    in_path = _write_raw(tmp_path / "data.zarr" / "raw", data)

    task_a = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=tmp_path / "data.zarr" / "out_a"),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(10, 10),
    )
    task_b = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=tmp_path / "data.zarr" / "out_b"),
        lambda_func=lambda x: (x > 0.3).astype(np.uint8),
        block_size=Coordinate(10, 10),
    )

    pipeline = task_a + task_b
    states = pipeline.run_blockwise(multiprocessing=False)

    assert states is not None
    assert isinstance(states, dict)
    assert task_a.task_name in states
    assert task_b.task_name in states
    for task in (task_a, task_b):
        state = states[task.task_name]
        assert isinstance(state, TaskState)
        assert state.completed_count == 2
        assert state.failed_count == 0
        assert state.is_done()


def test_pipeline_multiprocessing_returns_merged_state_map(tmp_path):
    """The Pipeline multiprocessing path also returns the merged map (not None).

    pytest tests/test_run_blockwise_states.py::test_pipeline_multiprocessing_returns_merged_state_map
    """
    data = np.linspace(0, 1, 200, dtype=np.float32).reshape(20, 10)
    in_path = _write_raw(tmp_path / "data.zarr" / "raw", data)

    task_a = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=tmp_path / "data.zarr" / "out_a"),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(10, 10),
        num_workers=1,
    )
    task_b = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=tmp_path / "data.zarr" / "out_b"),
        lambda_func=lambda x: (x > 0.3).astype(np.uint8),
        block_size=Coordinate(10, 10),
        num_workers=1,
    )

    pipeline = task_a + task_b
    states = pipeline.run_blockwise(multiprocessing=True)

    assert states is not None
    assert isinstance(states, dict)
    assert task_a.task_name in states
    assert task_b.task_name in states
    for task in (task_a, task_b):
        assert states[task.task_name].is_done()
