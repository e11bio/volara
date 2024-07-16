from contextlib import contextmanager
from typing import Optional, Union

import daisy
from funlib.geometry import Roi

from ..dataset import Dataset
from ..utils import PydanticCoordinate
from .blockwise import BlockwiseTask


class Cleanup(BlockwiseTask):
    datasets: list[Dataset]

    @property
    def write_roi(self) -> Roi:
        raise NotImplementedError()

    @property
    def write_size(self) -> PydanticCoordinate:
        raise NotImplementedError()

    def task(
        self, upstream_tasks: Optional[Union[daisy.Task, list[daisy.Task]]] = None
    ) -> daisy.Task:
        raise NotImplementedError()

    @contextmanager
    def process_blocks(self):
        raise NotImplementedError()
