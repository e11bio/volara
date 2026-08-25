"""Lazy operations are an ORDERED list of declarative models.

The keyword fields drive the named ops, in the one order that is correct. `Dataset.ops` is a list
of plain callables applied AFTER all of them — the escape hatch for a transform with no keyword.

Callables reach a worker via volara's `PydanticCallable`, cloudpickled to base64, the same way
`LambdaTask.lambda_func` ships one. A *bare* `Callable` annotation would not — that is the trap,
not callables themselves.
"""

import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds
from pydantic import BaseModel

from volara import datasets
from volara.datasets import CloudVolumeWrapper, Raw
from volara.ops import ReverseAxes


def _store(tmp_path, shape, name="t.zarr"):
    p = tmp_path / name
    arr = prepare_ds(
        p,
        shape=shape,
        dtype=np.uint16,
        voxel_size=Coordinate((1,) * len(shape)),
        offset=Coordinate((0,) * len(shape)),
        axis_names=list("tczyx"[-len(shape) :]),
        units=["nm"] * len(shape),
        mode="w",
    )
    data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    arr[:] = data
    return p, data


def _omero(tmp_path, windows, name="meta.zarr"):
    p = tmp_path / name
    zarr.open_group(str(p), mode="w").attrs["omero"] = {
        "channels": [{"window": {"min": lo, "max": hi}} for lo, hi in windows]
    }
    return p


# --------------------------------------------------------------- serialisation


def test_the_keyword_order_is_the_one_that_is_correct(tmp_path):
    """ome_norm indexes the channel axis so it precedes channels; flip names the collapsed array
    so it follows. Both constraints hold at once, which is why the order is not a preference."""
    p, data = _store(tmp_path, (2, 3, 5, 4, 3))
    r = Raw(store=p, channels=[1, 2], flip=[0], voxel_size=(1,) * 5, offset=(0,) * 5)
    assert [type(o).__name__ for o in r.resolved_ops()] == [
        "SelectChannels",
        "ReverseAxes",
    ]
    assert np.array_equal(np.asarray(r.array("r")[:]), data[1, 2][::-1])


def test_callables_apply_AFTER_every_keyword_op(tmp_path):
    p, data = _store(tmp_path, (2, 3, 5, 4, 3))
    r = Raw(
        store=p,
        channels=[1, 2],
        flip=[0],
        ops=[lambda d: d + 1],
        voxel_size=(1,) * 5,
        offset=(0,) * 5,
    )
    assert np.array_equal(np.asarray(r.array("r")[:]), (data[1, 2][::-1]) + 1)


def test_callables_run_in_the_order_given(tmp_path):
    p, data = _store(tmp_path, (5, 4, 3))
    r = Raw(
        store=p,
        ops=[lambda d: d + 1, lambda d: d * 2],
        voxel_size=(1, 1, 1),
        offset=(0, 0, 0),
    )
    assert np.array_equal(np.asarray(r.array("r")[:]), (data + 1) * 2)


def test_a_callable_survives_the_round_trip_a_worker_makes(tmp_path):
    """Cloudpickled to base64, so it crosses the config.json a worker reads."""
    p, data = _store(tmp_path, (5, 4, 3))
    r = Raw(
        store=p,
        channels=None,
        ops=[lambda d: d * 2],
        voxel_size=(1, 1, 1),
        offset=(0, 0, 0),
    )
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


def test_a_list_of_channel_selections_applies_in_order(tmp_path):
    """Each element indexes axis 0 of the array left by the one before it."""
    p, data = _store(tmp_path, (4, 5, 4, 3))
    got = np.asarray(
        Raw(
            store=p, channels=[[0, 1], [1]], voxel_size=(1,) * 4, offset=(0,) * 4
        ).array("r")[:]
    )
    assert np.array_equal(got, data[[0, 1]][[1]])


def test_ome_norm_runs_before_channels(tmp_path):
    """The constraint the order exists for: `ome_norm` needs the channel axis `channels` collapses.

    Swap the two and `ome_norm` reads the z extent as a channel count, so this is behavioural, not
    a check that the op list is spelled in a particular order.
    """
    p, data = _store(tmp_path, (2, 5, 4, 3))
    meta = _omero(tmp_path, [(0.0, 100.0), (0.0, 200.0)])
    r = Raw(store=p, ome_norm=meta, channels=[1], voxel_size=(1,) * 4, offset=(0,) * 4)
    assert np.allclose(np.asarray(r.array("r")[:]), data[1] / 200.0)


def test_ome_norm_takes_only_as_many_windows_as_the_array_has_channels(tmp_path):
    """OMERO metadata routinely describes more channels than the store carries."""
    p, data = _store(tmp_path, (2, 5, 4, 3))
    meta = _omero(tmp_path, [(0.0, 100.0), (0.0, 200.0), (0.0, 300.0)])
    r = Raw(store=p, ome_norm=meta, voxel_size=(1,) * 4, offset=(0,) * 4)
    got = np.asarray(r.array("r")[:])
    assert np.allclose(got, data / np.array([100.0, 200.0]).reshape(2, 1, 1, 1))
    assert r.attrs["bounds"] == [(0.0, 100.0), (0.0, 200.0)]


def test_ome_norm_names_the_shortfall_when_windows_run_out(tmp_path):
    p, _ = _store(tmp_path, (3, 5, 4, 3))
    meta = _omero(tmp_path, [(0.0, 100.0)])
    r = Raw(store=p, ome_norm=meta, voxel_size=(1,) * 4, offset=(0,) * 4)
    with pytest.raises(ValueError, match="1 OMERO channels"):
        r.array("r")


def test_scale_shift_is_float32_multiply_then_add(tmp_path):
    p, data = _store(tmp_path, (5, 4, 3))
    r = Raw(store=p, scale_shift=(2.0, -1.0), voxel_size=(1, 1, 1), offset=(0, 0, 0))
    got = np.asarray(r.array("r")[:])
    assert got.dtype == np.float32
    assert np.allclose(got, data.astype(np.float32) * 2.0 - 1.0)


def test_stack_concatenates_the_other_dataset_on_axis_0(tmp_path):
    p, data = _store(tmp_path, (2, 5, 4, 3))
    q, other = _store(tmp_path, (1, 5, 4, 3), name="u.zarr")
    stacked = Raw(store=q, voxel_size=(1,) * 4, offset=(0,) * 4)
    r = Raw(store=p, stack=stacked, voxel_size=(1,) * 4, offset=(0,) * 4)
    assert np.array_equal(
        np.asarray(r.array("r")[:]), np.concatenate([data, other], axis=0)
    )


def test_cloudvolume_refuses_ops_it_would_silently_drop(monkeypatch):
    """`array()` there never reaches the lazy-op path, so an accepted `ops` would be inert.

    Refusing has to happen before the volume is opened, hence the exploding stand-in.
    """

    def _explode(*a, **k):
        raise AssertionError("opened the volume instead of refusing `ops`")

    monkeypatch.setattr(datasets, "CloudVolume", _explode)
    c = CloudVolumeWrapper(store="gs://no/such/bucket", ops=[lambda d: d * 2])
    with pytest.raises(NotImplementedError, match="ops"):
        c.array("r")


def test_named_ops_stay_introspectable(tmp_path):
    """A consumer can see that a dataset reverses z; a cloudpickled callable does not expose it.

    slabreg depends on this -- it hashes the ops defining a slab's pixel frame.
    """
    p, _ = _store(tmp_path, (5, 4, 3))
    r = Raw(
        store=p, flip=[0], ops=[lambda d: d], voxel_size=(1, 1, 1), offset=(0, 0, 0)
    )
    named = [o for o in r.resolved_ops() if isinstance(o, ReverseAxes)]
    assert named and named[0].axes == [0]
