from typing import Annotated, Union

from pydantic import Field, TypeAdapter

from .blockwise import BlockwiseTask
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

BlockwiseTasks = TypeAdapter(
    Annotated[
        Union[
            Predict,
            ExtractFrags,
            SeededExtractFrags,
            AffAgglom,
            DistanceAgglom,
            DistanceAgglomSimple,
            GlobalMWS,
            LUT,
            Argmax,
        ],
        Field(discriminator="task_type"),
    ]
)

__all__ = [
    "PostProcess",
    "Predict",
    "DB",
    "SQLite",
    "PostgreSQL",
    "Agglom",
    "MWatershed",
    "EmbeddingFrags",
    "Waterz",
    "Model",
    "DaCapo",
    "Checkpoint",
    "Contrastive",
    "Raw",
    "Affs",
    "Labels",
    "Worker",
    "AffAgglom",
    "DistanceAgglom",
    "DistanceAgglomSimple",
    "LUT",
    "GlobalMWS",
    "ExtractFrags",
    "SeededExtractFrags",
    "Argmax",
    "BlockwiseTask",
    "BlockwiseTasks",
]
