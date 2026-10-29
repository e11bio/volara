import daisy
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import CLAHE
from volara.datasets import Raw


def _make_clahe_task(tmp_path):
    """Helper: 20x20 gradient input and a CLAHE task targeting it."""
    data = np.linspace(0, 255, 400, dtype=np.uint8).reshape(20, 20)
    in_path = tmp_path / "data.zarr" / "raw"
    prepare_ds(
        in_path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )[:] = data

    out_path = tmp_path / "data.zarr" / "clahe"
    task = CLAHE(
        in_arr=Raw(store=in_path),
        out_arr=Raw(store=out_path),
        block_size=Coordinate(20, 20),
        kernel=Coordinate(8, 8),
    )
    return task, data, out_path


def test_clahe_init_and_drop(tmp_path):
    """init() creates output zarr, drop_artifacts() removes it."""
    task, _, out_path = _make_clahe_task(tmp_path)
    task.init()
    assert out_path.exists()
    task.drop_artifacts()
    assert not out_path.exists()


def test_clahe_basic(tmp_path):
    """CLAHE on a gradient image produces different (contrast-enhanced) output in [0,255]."""
    task, data, _ = _make_clahe_task(tmp_path)
    task.init()

    block = daisy.Block(
        total_roi=Roi((0, 0), (20, 20)),
        read_roi=Roi((0, 0), (20, 20)),
        write_roi=Roi((0, 0), (20, 20)),
    )

    with task.process_block_func() as process_block:
        process_block(block)

    result = task.out_arr.array("r")[:]
    assert not np.array_equal(result, data)
    assert result.min() >= 0
    assert result.max() <= 255
