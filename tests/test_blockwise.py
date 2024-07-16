from contextlib import contextmanager
from typing import Optional, Union

import daisy
import pytest
from funlib.geometry import Coordinate, Roi

from volara.blockwise import BlockwiseTask
from volara.logging import set_log_basedir


@pytest.mark.skip(
    reason="Passes but never exits. Multiprocessing interaction with pytest?"
)
def test_blockwise(tmpdir):
    set_log_basedir(tmpdir)

    class DummyTask(BlockwiseTask):
        task_type: str = "dummy"

        @property
        def write_roi(self):
            return Roi((0, 0), (200, 200))

        @property
        def write_size(self) -> Coordinate:
            return Coordinate(100, 100)

        @contextmanager
        def process_block_func(self):
            def process_block(block):
                pass

            yield process_block

        def task(
            self, upstream_tasks: Optional[Union[daisy.Task, list[daisy.Task]]] = None
        ) -> daisy.Task:
            # create task
            task = daisy.Task(
                self.task_type,
                total_roi=self.write_roi,
                read_roi=Roi((0, 0), (10, 10)),
                write_roi=Roi((0, 0), (10, 10)),
                process_function=self.worker_func(),
                read_write_conflict=False,
                fit="shrink",
                num_workers=1,
                check_function=self.check_block_func(),
            )

            return task

        def init(self):
            pass

    config = DummyTask()
    daisy.run_blockwise([config.task()])
