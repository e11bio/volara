from typing import Annotated, Union

from pydantic import Field, TypeAdapter, Discriminator

from .blockwise import BlockwiseTask as BlockwiseTask
from .components import (
    LUT,
    AffAgglom,
    Argmax,
    DistanceAgglom,
    DistanceAgglomSimple,
    ExtractFrags,
    GlobalMWS,
    Predict,
    SeededExtractFrags,
)

BLOCKWISE_TASKS = [
    LUT,
    AffAgglom,
    Argmax,
    DistanceAgglom,
    DistanceAgglomSimple,
    ExtractFrags,
    GlobalMWS,
    Predict,
    SeededExtractFrags,
]


def register_task(task: BlockwiseTask):
    BLOCKWISE_TASKS.append(task)


def get_blockwise_tasks_type():
    return TypeAdapter(
        Annotated[
            Union[tuple(BLOCKWISE_TASKS)],
            Field(discriminator="task_type"),
        ]
    )
