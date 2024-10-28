import random
from contextlib import contextmanager
from typing import Literal

from daisy import Block, BlockStatus
from funlib.geometry import Coordinate, Roi

from volara.blockwise import BlockwiseTask


class DummyTask(BlockwiseTask):
    task_type: Literal["dummy"] = "dummy"
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
