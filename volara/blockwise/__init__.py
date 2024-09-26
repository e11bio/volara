from typing import Annotated, Union

from pydantic import Field, TypeAdapter

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
    PseudoAffs,
    SeededExtractFrags,
    Threshold,
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
    PseudoAffs,
    Threshold,
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
