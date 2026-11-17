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
from daisy import TaskState
from funlib.geometry import Coordinate

from tests.conftest import write_zarr
from volara.blockwise.lambda_task import LambdaTask
from volara.datasets import Labels, Raw

# A lambda that raises when a block's data exceeds a threshold. Passed through
# volara's cloudpickle-based PydanticCallable, so it must be an expression --
# hence the generator ``.throw`` idiom to raise inside the conditional.
_FAIL_ON_HIGH = lambda x: (  # noqa: E731
    (x > 0.5).astype(np.uint8)
    if x.max() <= 1.5
    else (_ for _ in ()).throw(RuntimeError("boom: block deliberately failed"))
)


@pytest.mark.parametrize(
    "combine, n_tasks, multiprocessing",
    [
        # a task with workers is test_lambda_task_multiprocessing_worker
        (lambda a, b: a, 1, False),
        (lambda a, b: a + b, 2, False),
        (lambda a, b: a + b, 2, True),
    ],
    ids=["task-serial", "pipeline-serial", "pipeline-workers"],
)
def test_run_blockwise_returns_a_task_state_per_task(
    gradient_20x10, tmp_path, combine, n_tasks, multiprocessing
):
    """A task and a sequential (``+``) pipeline both return ``{task_id: TaskState}``,
    serially and with workers; every block of every task completes."""
    in_path, _ = gradient_20x10
    tasks = [
        LambdaTask(
            in_data=Raw(store=in_path),
            out_data=Labels(store=tmp_path / "data.zarr" / name),
            lambda_func=lambda x: (x > 0.5).astype(np.uint8),
            block_size=Coordinate(10, 10),
            num_workers=1,
        )
        for name in ("out_a", "out_b")
    ]

    states = combine(*tasks).run_blockwise(multiprocessing=multiprocessing)

    assert {
        name: (
            type(state),
            state.total_block_count,
            state.completed_count,
            state.failed_count,
            state.orphaned_count,
        )
        for name, state in states.items()
    } == {task.task_name: (TaskState, 2, 2, 0, 0) for task in tasks[:n_tasks]}


def test_task_multiprocessing_surfaces_failed_blocks(tmp_path):
    """A worker function that raises on one block is counted in ``failed_count``,
    while ``is_done()`` is still True: it counts failed blocks as done, which is
    why a bool return cannot tell a clean run from one with failures."""
    # block 0 (rows 0..10) is all 0.0 -> succeeds
    # block 1 (rows 10..20) is all 2.0 -> lambda raises
    data = np.zeros((20, 10), dtype=np.float32)
    data[10:, :] = 2.0
    in_path = write_zarr(tmp_path / "data.zarr" / "raw", data)

    task = LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=tmp_path / "data.zarr" / "out"),
        lambda_func=_FAIL_ON_HIGH,
        block_size=Coordinate(10, 10),
        num_workers=1,
    )

    state = task.run_blockwise(multiprocessing=True)[task.task_name]

    assert (state.total_block_count, state.completed_count, state.failed_count) == (
        2,
        1,
        1,
    )
    assert state.is_done()
