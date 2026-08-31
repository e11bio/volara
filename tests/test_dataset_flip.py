"""`Dataset.flip` reverses axes of the FINAL array, and travels in the config.

The point of a declared field rather than a wrapper: a blockwise worker rebuilds the model from
`model_dump_json()` and calls `array()` itself, so the reversal applies there too. A subclass or an
in-memory wrapper applies in the driver and silently not in the worker.
"""

import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence import prepare_ds

from volara.datasets import Raw


def _store(tmp_path, shape, name="t.zarr"):
    p = tmp_path / name
    arr = prepare_ds(
        p, shape=shape, dtype=np.uint16, voxel_size=(1,) * len(shape),
        offset=(0,) * len(shape), axis_names=list("tczyx"[-len(shape):]),
        units=["nm"] * len(shape), mode="w",
    )
    data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    arr[:] = data
    return p, data


def test_flip_reverses_the_named_axis(tmp_path):
    p, data = _store(tmp_path, (6, 5, 4))
    r = Raw(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0))
    assert np.array_equal(np.asarray(r.array("r")[:]), data[::-1])


def test_flip_takes_several_axes(tmp_path):
    p, data = _store(tmp_path, (6, 5, 4))
    r = Raw(store=p, flip=[0, 2], voxel_size=(1, 1, 1), offset=(0, 0, 0))
    assert np.array_equal(np.asarray(r.array("r")[:]), data[::-1, :, ::-1])


def test_flip_indexes_the_array_AFTER_channels(tmp_path):
    """The whole ndim trap: on a 5-D store `z` is axis 2, but axis 0 once channels collapse."""
    p, data = _store(tmp_path, (2, 3, 6, 5, 4))
    r = Raw(store=p, channels=[1, 2], flip=[0], voxel_size=(1,) * 5, offset=(0,) * 5)
    got = np.asarray(r.array("r")[:])
    assert got.shape == (6, 5, 4)
    assert np.array_equal(got, data[1, 2][::-1])


def test_sub_roi_reads_through_the_reversal_are_exact(tmp_path):
    """A blockwise worker reads BLOCKS, so whole-array correctness is not enough."""
    p, data = _store(tmp_path, (12, 5, 4))
    ref = data[::-1]
    arr = Raw(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0)).array("r")
    for z0, zs in ((0, 3), (3, 4), (8, 4), (5, 2), (10, 2)):
        got = np.asarray(arr[Roi(Coordinate(z0, 0, 0), Coordinate(zs, 5, 4))])
        assert np.array_equal(got, ref[z0:z0 + zs]), f"z[{z0}:{z0 + zs}]"


def test_the_field_survives_a_round_trip_through_json(tmp_path):
    """This is the property the whole design rests on: the flip reaches a worker."""
    p, data = _store(tmp_path, (6, 5, 4))
    r = Raw(store=p, channels=None, flip=[0, 2], voxel_size=(1, 1, 1), offset=(0, 0, 0))
    rebuilt = Raw.model_validate_json(r.model_dump_json())
    assert rebuilt.flip == [0, 2]
    assert np.array_equal(np.asarray(rebuilt.array("r")[:]), data[::-1, :, ::-1])


def test_a_task_typed_field_keeps_the_flip(tmp_path):
    """A `Raw`-typed field on a task model must not erase it -- a SUBCLASS would be erased here."""
    from pydantic import BaseModel

    class _Task(BaseModel):
        intensities: Raw

    p, data = _store(tmp_path, (6, 5, 4))
    t = _Task(intensities=Raw(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0)))
    rebuilt = _Task.model_validate_json(t.model_dump_json())
    assert rebuilt.intensities.flip == [0]
    assert np.array_equal(np.asarray(rebuilt.intensities.array("r")[:]), data[::-1])


def test_no_flip_is_byte_identical_to_before(tmp_path):
    p, data = _store(tmp_path, (6, 5, 4))
    for flip in (None, []):
        r = Raw(store=p, flip=flip, voxel_size=(1, 1, 1), offset=(0, 0, 0))
        assert np.array_equal(np.asarray(r.array("r")[:]), data)


def test_an_out_of_range_axis_raises_naming_the_final_ndim(tmp_path):
    """`flip` indexes the FINAL array; a store-axis index is the mistake to catch."""
    p, _ = _store(tmp_path, (2, 3, 6, 5, 4))
    r = Raw(store=p, channels=[1, 2], flip=[4], voxel_size=(1,) * 5, offset=(0,) * 5)
    with pytest.raises(ValueError, match=r"out of range for the 3-D array"):
        r.array("r")


def test_a_write_mode_is_refused_up_front(tmp_path):
    """`is_writeable` reports True through a slice lazy op and `__setitem__` then raises."""
    p, _ = _store(tmp_path, (6, 5, 4))
    r = Raw(store=p, flip=[0], voxel_size=(1, 1, 1), offset=(0, 0, 0))
    with pytest.raises(ValueError, match="read-only"):
        r.array("a")


def test_cloudvolume_refuses_rather_than_ignoring(tmp_path):
    """It overrides array() without the lazy-op path, so a flip there would be silently inert."""
    from volara.datasets import CloudVolumeWrapper

    cv = CloudVolumeWrapper(store="gs://nonexistent/x", flip=[0])
    with pytest.raises(NotImplementedError, match="does not apply flip"):
        cv.array("r")
