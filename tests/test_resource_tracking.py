"""``BlockwiseTask.resource_tracking`` is handed to daisy, which then writes
per-block resource figures beside the done markers."""

import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate
from funlib.persistence.arrays import prepare_ds

from volara.blockwise.lambda_task import LambdaTask
from volara.datasets import Labels, Raw


def _task(tmp_path, **kwargs):
    data = np.linspace(0, 1, 400, dtype=np.float32).reshape(40, 10)
    in_path = tmp_path / "data.zarr" / "raw"
    prepare_ds(
        in_path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )[:] = data
    return LambdaTask(
        in_data=Raw(store=in_path),
        out_data=Labels(store=tmp_path / "data.zarr" / "out"),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(10, 10),
        **kwargs,
    )


@pytest.mark.parametrize("multiprocessing", [False, True])
def test_every_block_is_measured_by_default(tmp_path, multiprocessing):
    task = _task(tmp_path)
    assert task.resource_tracking

    task.run_blockwise(multiprocessing=multiprocessing)

    group = zarr.open_group(task.tracking_path, mode="r")
    assert np.asarray(group["done"][:]).sum() == 4
    assert (np.asarray(group["wall_seconds"][:]) > 0).all()
    assert (np.asarray(group["peak_rss_bytes"][:]) > 0).all()


def test_tracking_off_keeps_only_the_done_markers(tmp_path):
    task = _task(tmp_path, resource_tracking=False)
    task.run_blockwise(multiprocessing=False)

    group = zarr.open_group(task.tracking_path, mode="r")
    assert np.asarray(group["done"][:]).sum() == 4
    assert "wall_seconds" not in group
