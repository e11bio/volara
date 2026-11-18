"""``BlockwiseTask.resource_tracking`` is handed to daisy, which then writes
per-block resource figures beside the done markers."""

import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate

from tests.conftest import write_zarr
from volara.blockwise.lambda_task import LambdaTask
from volara.datasets import Labels, Raw


def _array(task, name):
    """One per-block array of the task's tracking group, flattened."""
    return np.asarray(zarr.open_array(task.tracking_path / name, mode="r")[:]).ravel()


def _task(tmp_path, **kwargs):
    data = np.linspace(0, 1, 400, dtype=np.float32).reshape(40, 10)
    in_path = tmp_path / "data.zarr" / "raw"
    write_zarr(in_path, data)
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

    assert _array(task, "done").sum() == 4
    assert (_array(task, "wall_seconds") > 0).all()
    assert (_array(task, "peak_rss_bytes") > 0).all()


@pytest.mark.filterwarnings("ignore:Object at runs.jsonl:zarr.errors.ZarrUserWarning")
def test_tracking_off_keeps_only_the_done_and_failure_markers(tmp_path):
    task = _task(tmp_path, resource_tracking=False)
    task.run_blockwise(multiprocessing=False)

    assert _array(task, "done").sum() == 4
    tracked = zarr.open_group(task.tracking_path, mode="r")
    assert sorted(tracked.array_keys()) == ["done", "failures"]
