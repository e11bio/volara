"""Lazy operations are an ORDERED list of declarative models.

Order changes the result, so it must be stated rather than implied by the order of `if` statements.

An op is a named model or a plain callable. Both reach a worker: volara's `PydanticCallable`
cloudpickles a callable to base64, the same way `LambdaTask.lambda_func` ships one. A *bare*
`Callable` annotation would not — that is the trap, not callables themselves.
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


def test_a_BARE_callable_annotation_would_not_have_worked():
    """The trap is the annotation, not callables. `PydanticCallable` is the one that ships."""
    from typing import Any, Callable

    from pydantic_core import PydanticSerializationError

    class _Bare(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        ops: list[Callable[[Any], Any]] = []

    with pytest.raises(PydanticSerializationError):
        _Bare(ops=[lambda d: d]).model_dump_json()


def test_a_plain_callable_op_reaches_the_worker(tmp_path):
    """Cloudpickled and base64'd, so it survives the config.json a worker reads."""
    p, data = _store(tmp_path, (5, 4, 3))
    r = Raw(store=p, ops=[lambda d: d * 2], voxel_size=(1, 1, 1), offset=(0, 0, 0))
    rebuilt = Raw.model_validate_json(r.model_dump_json())
    assert np.array_equal(np.asarray(rebuilt.array("r")[:]), data * 2)


def test_named_ops_and_callables_compose_in_order(tmp_path):
    p, data = _store(tmp_path, (2, 3, 5, 4, 3))
    r = Raw(store=p, ops=[SelectChannels(channels=[1, 2]), lambda d: d + 1,
                          ReverseAxes(axes=[0])],
            voxel_size=(1,) * 5, offset=(0,) * 5)
    rebuilt = Raw.model_validate_json(r.model_dump_json())
    assert np.array_equal(np.asarray(rebuilt.array("r")[:]), (data[1, 2] + 1)[::-1])


def test_a_named_op_stays_readable_in_the_config(tmp_path):
    """Why to prefer one: a consumer can see and hash it. A cloudpickle blob is opaque, and its
    bytes move with the Python and cloudpickle versions -- so hashing it would make every worker
    upgrade look like a change."""
    p, _ = _store(tmp_path, (5, 4, 3))
    named = Raw(store=p, ops=[ReverseAxes(axes=[0])], voxel_size=(1, 1, 1),
                offset=(0, 0, 0)).model_dump_json()
    assert '"op":"reverse_axes"' in named and '"axes":[0]' in named
    opaque = Raw(store=p, ops=[lambda d: d[::-1]], voxel_size=(1, 1, 1),
                 offset=(0, 0, 0)).model_dump_json()
    assert "reverse" not in opaque


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
