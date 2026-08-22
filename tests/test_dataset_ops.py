"""Lazy operations are an ORDERED list of declarative models.

Two properties carry the design. Order changes the result, so it must be stated rather than implied
by the order of `if` statements. And an op must SERIALISE, because a blockwise worker rebuilds the
task from `model_dump_json()` and re-opens the array itself — a `list[Callable]` applies in the
driver and not in the worker.
"""

import warnings

import numpy as np
import pytest
from funlib.persistence import prepare_ds
from pydantic import BaseModel

from volara.datasets import Raw
from volara.ops import AnyDatasetOp, ReverseAxes, ScaleShift, SelectChannels


def _store(tmp_path, shape, name="t.zarr"):
    p = tmp_path / name
    arr = prepare_ds(p, shape=shape, dtype=np.uint16, voxel_size=(1,) * len(shape),
                     offset=(0,) * len(shape), axis_names=list("tczyx"[-len(shape):]),
                     units=["nm"] * len(shape), mode="w")
    data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    arr[:] = data
    return p, data


# --------------------------------------------------------------- serialisation

def test_ops_survive_the_round_trip_a_worker_makes():
    """The property that rules out callables. A worker reads config.json and revalidates."""
    class _Task(BaseModel):
        ops: list[AnyDatasetOp] = []

    t = _Task(ops=[SelectChannels(channels=[0, 0]), ReverseAxes(axes=[0]),
                   ScaleShift(scale=0.5, shift=1.0)])
    back = _Task.model_validate_json(t.model_dump_json())
    assert [type(o).__name__ for o in back.ops] == ["SelectChannels", "ReverseAxes", "ScaleShift"]
    assert back.ops[2].scale == 0.5


def test_a_list_of_callables_could_not_have_worked():
    """Documents WHY these are models. Pydantic refuses to dump a function."""
    from typing import Any, Callable

    from pydantic_core import PydanticSerializationError

    class _WithCallables(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        ops: list[Callable[[Any], Any]] = []

    with pytest.raises(PydanticSerializationError):
        _WithCallables(ops=[lambda d: d]).model_dump_json()


# --------------------------------------------------------------- order

def test_order_changes_the_result_and_is_now_expressible(tmp_path):
    """Collapse-then-reverse and reverse-then-collapse are different arrays."""
    p, data = _store(tmp_path, (2, 3, 5, 4, 3))
    kw = dict(store=p, voxel_size=(1,) * 5, offset=(0,) * 5)
    collapse_first = np.asarray(
        Raw(ops=[SelectChannels(channels=[1, 2]), ReverseAxes(axes=[0])], **kw).array("r")[:])
    reverse_first = np.asarray(
        Raw(ops=[ReverseAxes(axes=[0]), SelectChannels(channels=[1, 2])], **kw).array("r")[:])
    assert np.array_equal(collapse_first, data[1, 2][::-1])
    assert not np.array_equal(collapse_first, reverse_first)


def test_reverse_axes_names_the_array_AS_IT_STANDS(tmp_path):
    """Not the store. After a 5-D collapse, axis 0 is z; before it, axis 0 is the timepoint."""
    p, data = _store(tmp_path, (2, 3, 5, 4, 3))
    kw = dict(store=p, voxel_size=(1,) * 5, offset=(0,) * 5)
    after = np.asarray(
        Raw(ops=[SelectChannels(channels=[1, 2]), ReverseAxes(axes=[0])], **kw).array("r")[:])
    assert after.shape == (5, 4, 3) and np.array_equal(after, data[1, 2][::-1])
    before = np.asarray(
        Raw(ops=[ReverseAxes(axes=[0]), SelectChannels(channels=[1, 2])], **kw).array("r")[:])
    assert np.array_equal(before, data[::-1][1, 2])


def test_an_out_of_range_axis_names_where_in_the_list_it_failed(tmp_path):
    p, _ = _store(tmp_path, (2, 3, 5, 4, 3))
    r = Raw(store=p, ops=[SelectChannels(channels=[1, 2]), ReverseAxes(axes=[4])],
            voxel_size=(1,) * 5, offset=(0,) * 5)
    with pytest.raises(ValueError, match=r"at this point in the op list"):
        r.array("r")


# --------------------------------------------------------------- deprecation

def test_the_keyword_form_still_works_and_is_byte_identical(tmp_path):
    """The migration's whole argument: a dataset that sets neither behaves exactly as before."""
    p, data = _store(tmp_path, (2, 3, 5, 4, 3))
    kw = dict(store=p, voxel_size=(1,) * 5, offset=(0,) * 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = np.asarray(Raw(channels=[1, 2], flip=[0], **kw).array("r")[:])
    explicit = np.asarray(
        Raw(ops=[SelectChannels(channels=[1, 2]), ReverseAxes(axes=[0])], **kw).array("r")[:])
    assert np.array_equal(legacy, explicit) and np.array_equal(legacy, data[1, 2][::-1])


def test_the_keyword_form_warns_and_names_the_equivalent(tmp_path):
    p, _ = _store(tmp_path, (2, 3, 5, 4, 3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Raw(store=p, channels=[1, 2], voxel_size=(1,) * 5, offset=(0,) * 5).array("r")
    msgs = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert msgs and "SelectChannels" in msgs[0]


def test_no_ops_and_no_keywords_warns_about_nothing(tmp_path):
    p, data = _store(tmp_path, (5, 4, 3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        got = np.asarray(Raw(store=p, voxel_size=(1, 1, 1), offset=(0, 0, 0)).array("r")[:])
    assert not [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert np.array_equal(got, data)


def test_setting_BOTH_is_refused_rather_than_silently_ordered(tmp_path):
    """There is no honest order for "these ops, plus that keyword" — so refuse."""
    p, _ = _store(tmp_path, (2, 3, 5, 4, 3))
    r = Raw(store=p, channels=[1, 2], ops=[ReverseAxes(axes=[0])],
            voxel_size=(1,) * 5, offset=(0,) * 5)
    with pytest.raises(ValueError, match="cannot express one unambiguous order"):
        r.array("r")


def test_a_reversal_still_refuses_a_write_mode(tmp_path):
    p, _ = _store(tmp_path, (5, 4, 3))
    r = Raw(store=p, ops=[ReverseAxes(axes=[0])], voxel_size=(1, 1, 1), offset=(0, 0, 0))
    with pytest.raises(ValueError, match="read-only"):
        r.array("a")


def test_repeated_channel_lists_do_not_late_bind(tmp_path):
    """The old inline loop closed over its variable, so several list elements all used the last."""
    p, data = _store(tmp_path, (4, 5, 4, 3))
    got = np.asarray(Raw(store=p, ops=[SelectChannels(channels=[[0, 1], [1]])],
                         voxel_size=(1,) * 4, offset=(0,) * 4).array("r")[:])
    assert np.array_equal(got, data[[0, 1]][[1]])
