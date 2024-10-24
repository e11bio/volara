import random
from contextlib import contextmanager

import pytest
from daisy import Block, BlockStatus
from daisy.task_state import TaskState
from funlib.geometry import Coordinate, Roi

from volara.blockwise import BlockwiseTask, register_task
from volara.workers import LocalWorker
from volara.logging import get_log_basedir


# @pytest.mark.parametrize("multiprocessing", [True, False])
@pytest.mark.parametrize("multiprocessing", [False])
# @pytest.mark.parametrize("worker", [None, LocalWorker()])
def test_dummy_blockwise(
    multiprocessing,
    # worker,
):
    class DummyTask(BlockwiseTask):
        task_type: str = "dummy"
        fit: str = "shrink"
        read_write_conflict: bool = False
        random_shift: int = 0

        @property
        def task_name(self) -> str:
            return "dummy"

        @property
        def write_roi(self):
            return Roi((0, 0), (200, 200))

        @property
        def write_size(self) -> Coordinate:
            return Coordinate(10, 10)

        @property
        def context_size(self) -> Coordinate:
            return Coordinate(0, 0)

        @contextmanager
        def process_block_func(self):
            def process_block(block: Block):
                random.seed(block.block_id[1] + self.random_shift)
                if random.random() < 0.5:
                    block.status = BlockStatus.FAILED
                else:
                    block.status = BlockStatus.SUCCESS

            yield process_block

        def drop_artifacts(self):
            pass

        def init(self):
            pass

    register_task(DummyTask)

    # config = DummyTask(worker_config=worker)
    config = DummyTask()
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
