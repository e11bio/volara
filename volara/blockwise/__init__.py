from typing import Annotated

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
    SeededExtractFrags,
)

BlockwiseTasks = TypeAdapter(
    Annotated[
        Predict
        | ExtractFrags
        | SeededExtractFrags
        | AffAgglom
        | DistanceAgglom
        | DistanceAgglomSimple
        | GlobalMWS
        | LUT
        | Argmax,
        Field(discriminator="task_type"),
    ]
)
