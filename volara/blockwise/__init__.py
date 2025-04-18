from typing import Annotated, Union

from importlib.metadata import entry_points
from pydantic import Field, TypeAdapter

from .blockwise import BlockwiseTask as BlockwiseTask
from .components import (
    AffAgglom,  # noqa
    ApplyShift,  # noqa
    Argmax,  # noqa
    ComputeShift,  # noqa
    DistanceAgglom,  # noqa
    ExtractFrags,  # noqa
    GraphMWS,  # noqa
    Predict,  # noqa
    Relabel,  # noqa
    SeededExtractFrags,  # noqa
    Threshold,  # noqa
)

BLOCKWISE_TASKS = []


def register_task(task: BlockwiseTask):
    if task not in BLOCKWISE_TASKS:
        BLOCKWISE_TASKS.append(task)


def discover_tasks():
    for entry_point in entry_points("volara.blockwise_tasks"):
        task_class = entry_point.load()
        register_task(task_class)


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


TASKS_DISCOVERED = False


def get_blockwise_tasks_type():
    global TASKS_DISCOVERED
    if not TASKS_DISCOVERED:
        discover_tasks()
        TASKS_DISCOVERED = True
    return TypeAdapter(
        Annotated[
            Union[tuple(BLOCKWISE_TASKS)],
            Field(discriminator="task_type"),
        ]
    )
