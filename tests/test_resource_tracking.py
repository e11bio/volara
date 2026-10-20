"""Per-block resource tracking through daisy v2 (``resource_tracking=True``).

Every ``BlockwiseTask`` asks daisy to measure its blocks by default; the
figures land in the tracking group and ``tracking_summary()`` folds them per
task, on both the serial and the multiprocessing path.
"""

import numpy as np
import pytest
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
def test_tracking_summary_counts_every_block(tmp_path, multiprocessing):
    task = _task(tmp_path)
    assert task.resource_tracking
    assert task.tracking_summary() is None

    states = task.run_blockwise(multiprocessing=multiprocessing)
    assert states[task.task_name].completed_count == 4

    summary = task.tracking_summary()
    assert summary is not None and summary["measured"]
    assert summary["blocks"] == 4 and summary["failures"] == 0
    assert summary["wall_seconds"] > 0
    assert summary["cpu_seconds"] >= 0
    assert summary["peak_rss_bytes"] > 0
    assert summary["max_block_seconds"] >= summary["mean_block_seconds"] > 0
    arrays = task.tracking_arrays()
    assert arrays is not None and arrays["done"].sum() == 4
    assert arrays["wall_seconds"].shape == (4,)

    task.drop()
    assert task.tracking_summary() is None


def test_tracking_off_keeps_the_done_markers(tmp_path):
    task = _task(tmp_path, resource_tracking=False)
    task.run_blockwise(multiprocessing=False)
    summary = task.tracking_summary()
    assert summary == {"blocks": 4, "failures": 0, "measured": False}
