from typing import Annotated, Union

import pkg_resources
from pydantic import Field, TypeAdapter

from .blockwise import BlockwiseTask as BlockwiseTask
from .components import (
    LUT as LUT,
)
from .components import (
    AffAgglom as AffAgglom,
)
from .components import (
    Argmax as Argmax,
)
from .components import (
    DistanceAgglom as DistanceAgglom,
)
from .components import (
    DistanceAgglomSimple as DistanceAgglomSimple,
)
from .components import (
    ExtractFrags as ExtractFrags,
)
from .components import (
    GlobalMWS as GlobalMWS,
)
from .components import (
    Predict as Predict,
)
from .components import (
    PseudoAffs as PseudoAffs,
)
from .components import (
    SeededExtractFrags as SeededExtractFrags,
)
from .components import (
    Threshold as Threshold,
)

BLOCKWISE_TASKS: list[BlockwiseTask] = []


def register_task(task: BlockwiseTask):
    BLOCKWISE_TASKS.append(task)


def discover_tasks():
    if len(BLOCKWISE_TASKS) > 11:
        return
    for entry_point in pkg_resources.iter_entry_points("volara.blockwise_tasks"):
        task_class = entry_point.load()
        register_task(task_class)


discover_tasks()


def get_task(task_type: str) -> BlockwiseTask:
    """
    If you don't know what library a task is in, you can use this function to get
    the task class.
    """
    for task in BLOCKWISE_TASKS:
        parsed_task_type = task.model_fields["task_type"].default
        if parsed_task_type == task_type:
            return task
    raise ValueError(f"Unknown task: {task_type}, {BLOCKWISE_TASKS}")


def get_blockwise_tasks_type():
    return TypeAdapter(
        Annotated[
            Union[tuple(BLOCKWISE_TASKS)],
            Field(discriminator="task_type"),
        ]
    )
