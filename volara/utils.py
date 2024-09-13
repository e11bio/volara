from funlib.geometry import Coordinate
from pydantic import BaseModel, ConfigDict


class PydanticCoordinate(Coordinate):
    """
    A thin wrapper around the Coordinate class that allows for Pydantic
    serialization, deserilization, and validation.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, val_info) -> Coordinate:
        return Coordinate(*v)


class StrictBaseModel(BaseModel):
    """
    A BaseModel that does not allow for extra fields.
    """

    model_config = ConfigDict(extra="forbid")
