"""Lazy operations are an ORDERED list of declarative models.

The keyword fields drive the named ops, in the one order that is correct. `Dataset.ops` is a list
of plain callables applied AFTER all of them — the escape hatch for a transform with no keyword.

Callables reach a worker via volara's `PydanticCallable`, cloudpickled to base64, the same way
`LambdaTask.lambda_func` ships one. A *bare* `Callable` annotation would not — that is the trap,
not callables themselves.
"""


import numpy as np
import pytest
from funlib.persistence import prepare_ds
from pydantic import BaseModel

from volara.datasets import Raw
from volara.ops import ReverseAxes


def _store(tmp_path, shape, name="t.zarr"):
    p = tmp_path / name
    arr = prepare_ds(p, shape=shape, dtype=np.uint16, voxel_size=(1,) * len(shape),
                     offset=(0,) * len(shape), axis_names=list("tczyx"[-len(shape):]),
                     units=["nm"] * len(shape), mode="w")
    data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    arr[:] = data
    return p, data


# --------------------------------------------------------------- serialisation

def test_the_keyword_order_is_the_one_that_is_correct(tmp_path):
    """ome_norm indexes the channel axis so it precedes channels; flip names the collapsed array
    so it follows. Both constraints hold at once, which is why the order is not a preference."""
    p, data = _store(tmp_path, (2, 3, 5, 4, 3))
    r = Raw(store=p, channels=[1, 2], flip=[0], voxel_size=(1,) * 5, offset=(0,) * 5)
    assert [type(o).__name__ for o in r.resolved_ops()] == ["SelectChannels", "ReverseAxes"]
    assert np.array_equal(np.asarray(r.array("r")[:]), data[1, 2][::-1])


def test_callables_apply_AFTER_every_keyword_op(tmp_path):
    p, data = _store(tmp_path, (2, 3, 5, 4, 3))
    r = Raw(store=p, channels=[1, 2], flip=[0], ops=[lambda d: d + 1],
            voxel_size=(1,) * 5, offset=(0,) * 5)
    assert np.array_equal(np.asarray(r.array("r")[:]), (data[1, 2][::-1]) + 1)


def test_callables_run_in_the_order_given(tmp_path):
    p, data = _store(tmp_path, (5, 4, 3))
    r = Raw(store=p, ops=[lambda d: d + 1, lambda d: d * 2],
            voxel_size=(1, 1, 1), offset=(0, 0, 0))
    assert np.array_equal(np.asarray(r.array("r")[:]), (data + 1) * 2)


def test_a_callable_survives_the_round_trip_a_worker_makes(tmp_path):
    """Cloudpickled to base64, so it crosses the config.json a worker reads."""
    p, data = _store(tmp_path, (5, 4, 3))
    r = Raw(store=p, channels=None, ops=[lambda d: d * 2],
            voxel_size=(1, 1, 1), offset=(0, 0, 0))
    rebuilt = Raw.model_validate_json(r.model_dump_json())
    assert np.array_equal(np.asarray(rebuilt.array("r")[:]), data * 2)


def test_a_BARE_callable_annotation_would_not_have_worked():
    """The trap is the annotation, not callables. PydanticCallable is the one that ships."""
    from typing import Any, Callable

    from pydantic_core import PydanticSerializationError

    class _Bare(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        ops: list[Callable[[Any], Any]] = []

    with pytest.raises(PydanticSerializationError):
        _Bare(ops=[lambda d: d]).model_dump_json()


def test_nothing_set_is_unchanged_and_op_free(tmp_path):
    p, data = _store(tmp_path, (5, 4, 3))
    r = Raw(store=p, voxel_size=(1, 1, 1), offset=(0, 0, 0))
    assert r.resolved_ops() == []
    assert np.array_equal(np.asarray(r.array("r")[:]), data)


def test_a_flip_still_refuses_a_write_mode(tmp_path):
    p, _ = _store(tmp_path, (5, 4, 3))
    r = Raw(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0))
    with pytest.raises(ValueError, match="read-only"):
        r.array("a")


def test_repeated_channel_lists_do_not_late_bind(tmp_path):
    """The old inline loop closed over its variable, so several elements all used the last."""
    p, data = _store(tmp_path, (4, 5, 4, 3))
    got = np.asarray(Raw(store=p, channels=[[0, 1], [1]],
                         voxel_size=(1,) * 4, offset=(0,) * 4).array("r")[:])
    assert np.array_equal(got, data[[0, 1]][[1]])


def test_named_ops_stay_introspectable(tmp_path):
    """A consumer can see that a dataset reverses z; a cloudpickled callable does not expose it.

    slabreg depends on this -- it hashes the ops defining a slab's pixel frame.
    """
    p, _ = _store(tmp_path, (5, 4, 3))
    r = Raw(store=p, flip=[0], ops=[lambda d: d], voxel_size=(1, 1, 1), offset=(0, 0, 0))
    named = [o for o in r.resolved_ops() if isinstance(o, ReverseAxes)]
    assert named and named[0].axes == [0]
