"""``ComputeShift`` measures, per block and per channel, the translation that
registers a channel onto channel 0 (or onto ``target``); ``ApplyShift`` resamples
the channels by those shifts."""

import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds
from scipy.ndimage import gaussian_filter

from volara.blockwise import ApplyShift, ComputeShift
from volara.datasets import Raw
from volara.logging import set_log_basedir


def _smooth_volume(shape) -> np.ndarray:
    """A smooth float32 volume in [0, 1] with structure at every position."""
    noise = np.random.default_rng(0).random(shape, dtype=np.float32)
    smooth = gaussian_filter(noise, 1.0)
    return (smooth - smooth.min()) / (smooth.max() - smooth.min())


@pytest.fixture(scope="module")
def registered(tmp_path_factory) -> tuple[ComputeShift, np.ndarray]:
    """A completed ``ComputeShift`` run over 18^3 voxels of 2 nm in 3x3x3 blocks,
    where channel 1 is channel 0 moved two voxels along z; and the data."""
    tmp_path = tmp_path_factory.mktemp("register")
    set_log_basedir(tmp_path / "volara_logs")
    base = _smooth_volume((18, 18, 18))
    data = np.stack([base, np.roll(base, 2, axis=0)])
    path = tmp_path / "data.zarr" / "raw"
    prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(2, 2, 2),
        dtype=data.dtype,
        axis_names=["c^", "z", "y", "x"],
        units=["nm", "nm", "nm"],
        types=["channel", "space", "space", "space"],
        mode="w",
    )[:] = data
    task = ComputeShift(
        intensities=Raw(store=path),
        shifts=Raw(store=tmp_path / "data.zarr" / "shifts"),
        block_size=Coordinate(6, 6, 6),
        context=Coordinate(6, 6, 6),
    )
    state = task.run_blockwise(multiprocessing=False)[task.task_name]
    assert (state.completed_count, state.failed_count) == (27, 0)
    return task, data


@pytest.mark.parametrize(
    "moved_by, shift",
    [((0, 0, 0), (0, 0, 0)), ((2, 0, 0), (-6, 0, 0)), ((0, -1, 2), (0, 2, -8))],
)
def test_compute_shift_is_the_translation_back_onto_the_target_in_world_units(
    moved_by, shift
):
    target = _smooth_volume((12, 12, 12))
    moving = np.roll(target, moved_by, axis=(0, 1, 2))

    shifts = ComputeShift.compute_shift(
        np.stack([target, moving]), target, voxel_size=np.array([3, 2, 4])
    )

    assert shifts.shape == (2, 3, 1, 1, 1)
    np.testing.assert_array_equal(shifts[:, :, 0, 0, 0], [(0, 0, 0), shift])


def test_compute_shift_is_zero_under_an_empty_mask():
    target = _smooth_volume((12, 12, 12))
    moving = np.roll(target, 2, axis=0)

    shifts = ComputeShift.compute_shift(
        np.stack([target, moving]),
        target,
        voxel_size=np.array([1, 1, 1]),
        mask=np.zeros((12, 12, 12), dtype=np.uint8),
    )

    np.testing.assert_array_equal(shifts, np.zeros((2, 3, 1, 1, 1)))


def test_compute_shift_task_stores_one_shift_per_block_and_channel(registered):
    task, _ = registered

    shifts = task.shifts.array("r")
    assert shifts.shape == (2, 3, 3, 3, 3)  # channel, axis, one cell per block
    assert shifts.voxel_size == Coordinate(12, 12, 12)
    np.testing.assert_array_equal(shifts[:][0], np.zeros((3, 3, 3, 3)))
    # the central block reads only data: two voxels of 2 nm back along z
    np.testing.assert_array_equal(shifts[:][1, :, 1, 1, 1], [-4, 0, 0])


def test_apply_shift_task_moves_each_channel_back_onto_channel_0(registered, tmp_path):
    """Applied to the central block, where every neighbouring block measured
    the same shift."""
    compute_shift, data = registered

    task = ApplyShift(
        intensities=compute_shift.intensities,
        shifts=compute_shift.shifts,
        aligned=Raw(store=tmp_path / "aligned.zarr"),
        roi=Roi((12, 12, 12), (12, 12, 12)),
    )
    state = task.run_blockwise(multiprocessing=False)[task.task_name]

    assert (state.completed_count, state.failed_count) == (1, 0)
    aligned = task.aligned.array("r")[:]
    channel_0 = (data[0, 6:12, 6:12, 6:12] * 255).astype(np.uint8)
    np.testing.assert_array_equal(aligned[0], channel_0)
    np.testing.assert_allclose(aligned[1], channel_0, atol=1)
