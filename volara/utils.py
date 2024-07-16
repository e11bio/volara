from funlib.geometry import Coordinate
from pydantic import BaseModel, ConfigDict


class PydanticCoordinate(Coordinate):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, val_info) -> Coordinate:
        return Coordinate(*v)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
