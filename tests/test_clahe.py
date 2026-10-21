import daisy
import numpy as np
import pytest
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


def test_clahe_refuses_a_roi_off_the_output_grid(tmp_path):
    """CLAHE prepares `out_arr` over the WHOLE input but anchors its blocks at the roi, so a roi
    that starts mid-chunk puts every block across two stored objects; with more than one worker
    that lost data silently (measured on main). Refused on the driver, before any block runs."""
    data = np.linspace(0, 255, 256 * 256, dtype=np.uint8).reshape(256, 256)
    in_path = tmp_path / "data.zarr" / "raw"
    prepare_ds(in_path, shape=data.shape, voxel_size=Coordinate(1, 1), dtype=data.dtype,
               mode="w")[:] = data
    out_path = tmp_path / "data.zarr" / "clahe"

    def task(roi):
        return CLAHE(in_arr=Raw(store=in_path), out_arr=Raw(store=out_path),
                     block_size=Coordinate(64, 64), kernel=Coordinate(8, 8), roi=roi)

    assert [ds.store for ds in task(None).output_datasets] == [out_path]
    with pytest.raises(ValueError, match="off the chunk grid on axes \\[0, 1\\]"):
        task(Roi((32, 32), (224, 224))).run_blockwise(multiprocessing=False)
    assert not task(None).out_arr.array("r")[:].any(), "refused before any block ran"
    # the same window on the grid runs
    task(Roi((64, 64), (192, 192))).run_blockwise(multiprocessing=False)
    assert task(None).out_arr.array("r")[Roi((64, 64), (192, 192))].any()
