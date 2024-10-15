from typing import Annotated, Union

import pkg_resources
from pydantic import Field, TypeAdapter

from .blockwise import BlockwiseTask as BlockwiseTask
from .components import (
    LUT,
    AffAgglom,
    ApplyShift,
    Argmax,
    ComputeShift,
    DistanceAgglom,
    DistanceAgglomSimple,
    ExtractFrags,
    GlobalMWS,
    Predict,
    PseudoAffs,
    SeededExtractFrags,
    Threshold,
)

BLOCKWISE_TASKS = [
    LUT,
    AffAgglom,
    ApplyShift,
    Argmax,
    ComputeShift,
    DistanceAgglom,
    DistanceAgglomSimple,
    ExtractFrags,
    GlobalMWS,
    Predict,
    SeededExtractFrags,
    PseudoAffs,
    Threshold,
]


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
