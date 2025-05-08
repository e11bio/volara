from funlib.geometry import Coordinate
from pydantic import BaseModel, ConfigDict
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema


class PydanticCoordinate(Coordinate):
    """A Pydantic-compatible Coordinate type."""

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        return core_schema.chain_schema(
            [
                core_schema.list_schema(
                    items_schema=core_schema.int_schema(), min_length=1
                ),
                core_schema.no_info_plain_validator_function(cls.validate),
            ]
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema, _handler) -> JsonSchemaValue:
        """Defines the JSON schema for OpenAPI compatibility."""
        return {"type": "array", "items": {"type": "integer"}, "minItems": 1}

    @classmethod
    def validate(cls, v):
        if isinstance(v, Coordinate):
            return v  # Already a Coordinate
        if isinstance(v, (tuple, list)) and all(isinstance(i, int) for i in v):
            return Coordinate(v)  # Convert tuple/list to Coordinate
        raise ValueError(f"Invalid coordinate: {v}")


class StrictBaseModel(BaseModel):
    """
    A BaseModel that does not allow for extra fields.
    """

    model_config = ConfigDict(extra="forbid")
