from contextlib import contextmanager
from typing import Literal

import daisy
import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate, Roi

from volara.blockwise.blockwise import _SEED_ATTR, BlockwiseTask
from volara.datasets import MaskDataset
from volara.logging import set_log_basedir


class MaskableTask(BlockwiseTask):
    """Minimal concrete task over a 4x1 block grid; the body touches one file
    per call under ``count_dir`` (daisy may run the body in worker processes,
    so an in-process list would count nothing)."""

    task_type: Literal["maskable"] = "maskable"
    label: str = "maskable-task"
    count_dir: str = ""

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return self.label

    @property
    def write_roi(self) -> Roi:
        return Roi((0, 0), (40, 10))

    @property
    def write_size(self) -> Coordinate:
        return Coordinate(10, 10)

    @property
    def context_size(self) -> Coordinate:
        return Coordinate(0, 0)

    def drop_artifacts(self):
        pass

    def init(self):
        pass

    @contextmanager
    def process_block_func(self):
        count_dir = self.count_dir

        def process_block(block):
            from pathlib import Path as _P

            name = "_".join(str(int(o)) for o in block.write_roi.offset)
            (_P(count_dir) / name).touch()

        yield process_block


def _counted(cls, tmp_path, **kw):
    d = tmp_path / "calls"
    d.mkdir(exist_ok=True)
    return cls(count_dir=str(d), **kw)


def _mask(tmp_path, bits, name="mask.zarr"):
    arr = np.asarray(bits, dtype=np.uint8)
    z = zarr.open(
        str(tmp_path / name), mode="w", shape=arr.shape, chunks=arr.shape,
        dtype="uint8",
    )
    z[:] = arr
    return MaskDataset(store=tmp_path / name)


def _calls(task) -> list[str]:
    from pathlib import Path as _P

    return sorted(p.name for p in _P(task.count_dir).iterdir())


def _run(task) -> int:
    from pathlib import Path as _P

    d = _P(task.count_dir)
    for f in d.iterdir():
        f.unlink()
    with task.task(multiprocessing=False) as t:
        daisy.run_blockwise([t], progress=False)
    return len(_calls(task))


def test_fresh_seed_runs_exactly_the_unmasked_blocks(tmp_path):
    set_log_basedir(tmp_path / "logs")
    t = _counted(MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [1], [0]]))
    assert _run(t) == 2
    assert _calls(t) == ["10_0", "30_0"]
    done = zarr.open(str(t.tracking_path / "done"), mode="r")
    assert np.asarray(done[:]).ravel().tolist() == [1, 1, 1, 1]


def test_a_narrower_mask_unskips_what_it_no_longer_covers(tmp_path):
    set_log_basedir(tmp_path / "logs")
    t = _counted(MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [1], [0], [0]], "a.zarr"))
    assert _run(t) == 2  # cells 2, 3 computed for real
    t2 = _counted(MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [0], [0]], "b.zarr"))
    assert _run(t2) == 1, "cell 1 was only ever mask-skipped; the narrower mask must re-run it"
    assert _calls(t2) == ["10_0"]


def test_a_daisy_written_store_counts_as_real_completions(tmp_path):
    set_log_basedir(tmp_path / "logs")
    assert _run(_counted(MaskableTask, tmp_path)) == 4  # mask-less: daisy writes its own store
    t = _counted(MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [0], [0]]))
    assert _run(t) == 0, "every block was really computed; the mask must not erase that"


def test_a_wrong_shape_mask_is_refused_in_python_naming_the_grid(tmp_path):
    set_log_basedir(tmp_path / "logs")
    t = _counted(MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [1]]))
    with pytest.raises(ValueError, match=r"block grid \(4, 1\)"):
        t._seed_block_done_mask()


def test_a_layout_change_refuses_to_reuse_the_seeded_store(tmp_path):
    set_log_basedir(tmp_path / "logs")
    t = _counted(MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [1], [0]]))
    assert _run(t) == 2

    class ShiftedTask(MaskableTask):
        # same grid shape (4, 1), different world placement: the shape check
        # cannot catch this; only the recorded layout can
        @property
        def write_roi(self) -> Roi:
            return Roi((10, 0), (40, 10))

    t2 = _counted(ShiftedTask, tmp_path, block_done_mask=_mask(tmp_path, [[1], [0], [1], [0]], "c.zarr"))
    with pytest.raises(RuntimeError, match="different task geometry"):
        t2._seed_block_done_mask()


def test_the_seed_record_is_written_alongside_the_done_array(tmp_path):
    set_log_basedir(tmp_path / "logs")
    t = _counted(MaskableTask, tmp_path, block_done_mask=_mask(tmp_path, [[0], [1], [0], [0]]))
    t._seed_block_done_mask()
    group = zarr.open_group(str(t.tracking_path), mode="r")
    assert np.asarray(group["seed"][:]).ravel().tolist() == [0, 1, 0, 0]
    record = group.attrs[_SEED_ATTR]
    assert record["layout"]["grid_shape"] == [4, 1]
    assert len(record["mask_sha256"]) == 64


def test_mask_dataset_survives_the_task_config_round_trip(tmp_path):
    t = MaskableTask(block_done_mask=MaskDataset(store=tmp_path / "m.zarr"))
    round_tripped = MaskableTask.model_validate_json(t.model_dump_json())
    assert round_tripped.block_done_mask is not None
    assert str(round_tripped.block_done_mask.store) == str(tmp_path / "m.zarr")


def test_block_grid_slices_anchor_at_the_write_offset():
    class OffsetContextTask(MaskableTask):
        @property
        def write_roi(self) -> Roi:
            return Roi((100, 0), (40, 10))

        @property
        def context_size(self) -> Coordinate:
            return Coordinate(9, 0)

    t = OffsetContextTask()
    # grown total spans [91, 149) -> ceil(58 / 10) = 6 grid cells in dim 0
    assert t.block_grid_shape == (6, 1)
    # cell 0 covers the write window starting at write_roi.offset (100), NOT
    # at the grown total's offset (91)
    assert t.block_grid_slices(Roi((100, 0), (10, 10))) == (slice(0, 1), slice(0, 1))
    assert t.block_grid_slices(Roi((130, 0), (10, 10))) == (slice(3, 4), slice(0, 1))
    # a roi below the anchor clips to the grid's origin
    assert t.block_grid_slices(Roi((91, 0), (9, 10))) == (slice(0, 0), slice(0, 1))
