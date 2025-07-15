from importlib.metadata import entry_points
from typing import Annotated, Union

from pydantic import Field, TypeAdapter

from .aff_agglom import AffAgglom as AffAgglom
from .argmax import Argmax as Argmax
from .blockwise import BlockwiseTask
from .distance_agglom import (
    DistanceAgglom as DistanceAgglom,
)
from .extract_frags import ExtractFrags as ExtractFrags
from .graph_mws import GraphMWS as GraphMWS
from .register import ApplyShift as ApplyShift
from .register import ComputeShift as ComputeShift
from .relabel import Relabel as Relabel
from .seeded_extract_frags import SeededExtractFrags as SeededExtractFrags
from .threshold import Threshold as Threshold

BLOCKWISE_TASKS = []


def register_task(task: BlockwiseTask):
    if task not in BLOCKWISE_TASKS:
        BLOCKWISE_TASKS.append(task)


def discover_tasks():
    for entry_point in entry_points().select(group="volara.blockwise_tasks"):
        task_class = entry_point.load()
        register_task(task_class)


def get_task(task_type: str) -> BlockwiseTask:
    """
    Get the task class for a given task type. This is useful for dynamically fetching
    tasks from the base `volara.blockwise` module, despite some tasks potentially
    being defined in other modules as plugins via entry points. This allows `volara`
    to serialize and execute external tasks in isolated environments such as on
    cluster workers.
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


__all__ = [
    "BlockwiseTask",
    "AffAgglom",
    "ApplyShift",
    "Argmax",
    "ComputeShift",
    "DistanceAgglom",
    "ExtractFrags",
    "GraphMWS",
    "Relabel",
    "SeededExtractFrags",
    "Threshold",
]
