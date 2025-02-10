import pytest
from daisy.task_state import TaskState

from tests.dummy import DummyTask
from volara.workers import LocalWorker


@pytest.mark.parametrize("multiprocessing", [True, False])
@pytest.mark.parametrize("worker", [None, LocalWorker()])
def test_dummy_blockwise(
    multiprocessing,
    worker,
):
    config = DummyTask(worker_config=worker)
    task_state: TaskState = config.run_blockwise(multiprocessing=multiprocessing)[
        config.task_name
    ]
    assert task_state.failed_count == 212
    assert task_state.skipped_count == 0
    assert task_state.completed_count == 188

    task_state = config.run_blockwise(multiprocessing=multiprocessing)[config.task_name]
    assert task_state.failed_count == 212
    assert task_state.skipped_count == 188
    assert task_state.completed_count == 188

    config.random_shift = 1
    task_state = config.run_blockwise(multiprocessing=multiprocessing)[config.task_name]
    assert task_state.failed_count == 110
    assert task_state.skipped_count == 188
    assert task_state.completed_count == 290
