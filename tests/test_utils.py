from funlib.geometry import Coordinate, Roi
from pydantic import ValidationError

from volara.utils import (
    PydanticCallable,
    PydanticCoordinate,
    PydanticRoi,
    StrictBaseModel,
)


class CoordinateModel(StrictBaseModel):
    c: PydanticCoordinate


class RoiModel(StrictBaseModel):
    r: PydanticRoi


def test_pydantic_coordinate_from_list():
    m = CoordinateModel(c=[1, 2, 3])  # type: ignore[invalid-argument-type]
    assert isinstance(m.c, Coordinate)
    assert tuple(m.c) == (1, 2, 3)


def test_pydantic_coordinate_from_coordinate():
    m = CoordinateModel(c=Coordinate(4, 5))
    assert isinstance(m.c, Coordinate)
    assert tuple(m.c) == (4, 5)


def test_pydantic_coordinate_serialization():
    m = CoordinateModel(c=[10, 20])  # type: ignore[invalid-argument-type]
    dumped = m.model_dump_json()
    loaded = CoordinateModel.model_validate_json(dumped)
    assert tuple(loaded.c) == (10, 20)


def test_pydantic_roi_from_tuples():
    m = RoiModel(r=([0, 0], [10, 10]))  # type: ignore[invalid-argument-type]
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
        MyModel(x=1, y=2)  # type: ignore[unknown-argument]
        assert False, "Should have raised"
    except ValidationError:
        pass


def test_pydantic_callable_has_a_json_schema():
    """A PydanticCallable field must not break `model_json_schema()`.

    Without an explicit `__get_pydantic_json_schema__`, pydantic cannot derive a
    schema from the plain-validator-function core schema and raises
    PydanticInvalidForJsonSchema -- which takes the whole model with it, so any
    task or dataset carrying a callable field becomes un-introspectable.

    pytest tests/test_utils.py::test_pydantic_callable_has_a_json_schema
    """

    class M(StrictBaseModel):
        fn: PydanticCallable | None = None

    schema = M.model_json_schema()["properties"]["fn"]
    # `fn` is optional, so the callable schema sits inside the anyOf.
    variants = schema.get("anyOf", [schema])
    callable_schema = next(v for v in variants if v.get("type") == "string")
    assert callable_schema["format"] == "base64-cloudpickle"
    # The unpickle-is-arbitrary-code caveat belongs where a schema consumer will see it.
    assert "arbitrary code" in callable_schema["description"]
