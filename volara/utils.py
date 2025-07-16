from typing import Annotated, Any

from funlib.geometry import Coordinate, Roi
from pydantic import (
    BaseModel,
    ConfigDict,
)
from pydantic_core import core_schema


class _CoordinatePydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Any,
    ) -> core_schema.CoreSchema:
        def coerce_coordinate(value: list[int]) -> Coordinate:
            return Coordinate(*value)

        # Schema to handle incoming data (lists of integers)
        from_list_schema = core_schema.chain_schema(
            [
                core_schema.list_schema(core_schema.int_schema()),
                core_schema.no_info_plain_validator_function(coerce_coordinate),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_list_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(Coordinate),
                    from_list_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: tuple(instance)
            ),
        )


# Create a convenient type alias using Annotated
PydanticCoordinate = Annotated[Coordinate, _CoordinatePydanticAnnotation]


class _RoiPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Any,
    ) -> core_schema.CoreSchema:
        """Defines the Pydantic core schema for the Roi class."""

        coordinate_schema = _handler(_CoordinatePydanticAnnotation)

        def validate_from_coordinates(data: tuple[Coordinate, Coordinate]) -> Roi:
            return Roi(data[0], data[1])

        two_coordinates_schema = core_schema.chain_schema(
            [
                core_schema.tuple_positional_schema(
                    [coordinate_schema, coordinate_schema]
                ),
                core_schema.no_info_plain_validator_function(validate_from_coordinates),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=two_coordinates_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(Roi),
                    two_coordinates_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: (tuple(instance.offset), tuple(instance.shape))
            ),
        )


PydanticRoi = Annotated[Roi, _RoiPydanticAnnotation]


class StrictBaseModel(BaseModel):
    """
    A BaseModel that does not allow for extra fields.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )
