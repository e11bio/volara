
from funlib.geometry import Coordinate, Roi
from pydantic import ValidationError

from volara.utils import PydanticCoordinate, PydanticRoi, StrictBaseModel


class CoordinateModel(StrictBaseModel):
    c: PydanticCoordinate


class RoiModel(StrictBaseModel):
    r: PydanticRoi


def test_pydantic_coordinate_from_list():
    m = CoordinateModel(c=[1, 2, 3])  # type: ignore
    assert isinstance(m.c, Coordinate)
    assert tuple(m.c) == (1, 2, 3)


def test_pydantic_coordinate_from_coordinate():
    m = CoordinateModel(c=Coordinate(4, 5))
    assert isinstance(m.c, Coordinate)
    assert tuple(m.c) == (4, 5)


def test_pydantic_coordinate_serialization():
    m = CoordinateModel(c=[10, 20])  # type: ignore
    dumped = m.model_dump_json()
    loaded = CoordinateModel.model_validate_json(dumped)
    assert tuple(loaded.c) == (10, 20)


def test_pydantic_roi_from_tuples():
    m = RoiModel(r=([0, 0], [10, 10]))  # type: ignore
    assert isinstance(m.r, Roi)
    assert m.r == Roi((0, 0), (10, 10))


def test_pydantic_roi_from_roi():
    m = RoiModel(r=Roi((1, 2), (3, 4)))
    assert isinstance(m.r, Roi)
    assert m.r == Roi((1, 2), (3, 4))


def test_pydantic_roi_serialization():
    m = RoiModel(r=Roi((0, 0), (10, 20)))
    dumped = m.model_dump_json()
    loaded = RoiModel.model_validate_json(dumped)
    assert loaded.r == Roi((0, 0), (10, 20))


def test_strict_base_model_forbids_extra():
    class MyModel(StrictBaseModel):
        x: int

    try:
        MyModel(x=1, y=2)  # type: ignore
        assert False, "Should have raised"
    except ValidationError:
        pass
