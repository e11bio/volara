"""Tests for ``BlockwiseTask.block_done_mask`` and ``MaskDataset``.

Two things here would otherwise produce a green suite over a broken feature:

- A block body runs in a daisy worker, so an in-process counter counts nothing
  and cannot tell a skip from a worker. Blocks record themselves by touching a
  file instead.
- daisy sizes ``done`` over the context-grown total ROI but indexes it by
  ``(block.write_roi.offset - total_roi.offset) // block``, so the block written
  at ``write_roi.offset + j*block`` is cell ``j + context_low // block``. The
  shift is 0 whenever the halo is under one block, and a naively-built mask has
  the right *shape*, so daisy's shape check passes and the bits land in the
  wrong cells with no error. The anchor tests run at ``context >= block`` and
  compare against daisy's own ``done``.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate, Roi

from volara.blockwise.blockwise import BlockwiseTask
from volara.datasets import MaskDataset
from volara.logging import set_log_basedir


class _Recorder:
    """Picklable block body; touches a file per block so a worker's run is seen."""

    def __init__(self, ran_dir: Path):
        self.ran_dir = Path(ran_dir)

    def __call__(self, block):
        offset = "_".join(str(int(v)) for v in block.write_roi.offset)
        (self.ran_dir / offset).write_text("")


class RecordingTask(BlockwiseTask):
    task_type: Literal["recording"] = "recording"
    label: str = "recording-task"

    ran_dir: Path
    context: tuple[int, int] = (0, 0)

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
        return Coordinate(self.context)

    def drop_artifacts(self):
        pass

    @contextmanager
    def process_block_func(self):
        yield _Recorder(self.ran_dir)


def _task(tmp_path: Path, context: tuple[int, int] = (0, 0)) -> RecordingTask:
    set_log_basedir(tmp_path / "logs")
    ran_dir = tmp_path / "ran"
    ran_dir.mkdir(exist_ok=True)
    return RecordingTask(ran_dir=ran_dir, context=context)


def _ran(task: RecordingTask) -> list[int]:
    """z offsets of the blocks that actually ran."""
    return sorted(int(p.name.split("_")[0]) for p in task.ran_dir.iterdir())


def _grid(task: RecordingTask) -> tuple[int, ...]:
    context = Coordinate(task.context)
    total = task.write_roi.grow(context, context)
    return tuple(-(-t // b) for t, b in zip(total.shape, task.write_size))


def _mask(path: Path, array: np.ndarray, dtype: str = "uint8") -> MaskDataset:
    z = zarr.create_array(
        store=str(path),
        shape=array.shape,
        chunks=array.shape,
        dtype=dtype,
        overwrite=True,
    )
    z[:] = array
    return MaskDataset(store=path)


def _done(task: RecordingTask) -> list[int]:
    store = task.meta_dir / "blocks_done" / "done"
    return np.asarray(zarr.open(str(store), mode="r")[:], dtype=np.uint8)[:, 0].tolist()


def _run_with(task: RecordingTask, cells: list[int], name: str = "mask") -> None:
    mask = np.zeros(_grid(task), dtype=np.uint8)
    for cell in cells:
        mask[cell, 0] = 1
    task.block_done_mask = _mask(task.ran_dir.parent / f"{name}.zarr", mask)
    task.run_blockwise(multiprocessing=False)


def test_an_unmasked_run_touches_every_block(tmp_path):
    task = _task(tmp_path)
    task.run_blockwise(multiprocessing=False)
    assert _ran(task) == [0, 10, 20, 30]


def test_masked_blocks_are_never_run(tmp_path):
    task = _task(tmp_path)
    _run_with(task, [0, 3])
    assert _ran(task) == [10, 20]


@pytest.mark.parametrize(
    "context, done",
    [
        (0, [1, 1, 1, 1]),
        (9, [1, 1, 1, 1, 0, 0]),
        (10, [0, 1, 1, 1, 1, 0]),
        (25, [0, 0, 1, 1, 1, 1, 0, 0, 0]),
    ],
)
def test_daisy_anchors_done_at_context_low_over_block(tmp_path, context, done):
    """Four blocks run in every row; only where their bits land moves."""
    task = _task(tmp_path, context=(context, 0))
    task.run_blockwise(multiprocessing=False)
    assert _ran(task) == [0, 10, 20, 30]
    assert _done(task) == done


def test_a_mask_cell_skips_the_block_daisy_maps_it_to(tmp_path):
    """At context 25, block 10, the shift is 2: cell 2 is the block at z=0."""
    task = _task(tmp_path, context=(25, 0))
    assert _grid(task)[0] == 9, "fixture needs room for the shift to hide in"
    _run_with(task, [2])
    assert _ran(task) == [10, 20, 30]


def test_the_naive_cell_index_skips_nothing(tmp_path):
    """Cell 0 at shift 2 is context daisy never schedules, so nothing is skipped."""
    task = _task(tmp_path, context=(25, 0))
    _run_with(task, [0])
    assert _ran(task) == [0, 10, 20, 30]


def test_a_resume_preserves_earlier_completions(tmp_path):
    task = _task(tmp_path)
    task.run_blockwise(multiprocessing=False)
    for p in task.ran_dir.iterdir():
        p.unlink()

    _run_with(task, [0])

    assert _ran(task) == []
    assert _done(task) == [1, 1, 1, 1]


def test_a_narrower_mask_does_not_unskip(tmp_path):
    """A deliberate limitation, pinned so it stays a choice.

    The seed is OR'd into ``done`` and daisy cannot tell a bit a worker earned
    from one the mask asserted, so a skip survives any later mask -- including no
    mask. Re-running a skipped block means dropping the tracking store.
    """
    task = _task(tmp_path)
    _run_with(task, [0], name="wide")
    assert _ran(task) == [10, 20, 30]
    for p in task.ran_dir.iterdir():
        p.unlink()

    _run_with(task, [], name="narrow")

    assert _ran(task) == []
    assert _done(task)[0] == 1


def test_a_wrong_shape_mask_is_refused_before_any_block_runs(tmp_path):
    task = _task(tmp_path)
    task.block_done_mask = _mask(tmp_path / "bad.zarr", np.zeros((3, 1), np.uint8))

    with pytest.raises(RuntimeError, match="shape inconsistent with task layout"):
        task.run_blockwise(multiprocessing=False)

    assert _ran(task) == []


def test_a_shape_change_since_the_last_run_is_refused(tmp_path):
    """The resume path has its own check, with its own message."""
    task = _task(tmp_path)
    _run_with(task, [])

    task.block_done_mask = _mask(tmp_path / "bad.zarr", np.zeros((5, 1), np.uint8))
    with pytest.raises(ValueError, match="does not match"):
        task.run_blockwise(multiprocessing=False)


def test_the_seeded_done_array_is_uncompressed_and_single_chunk(tmp_path):
    """daisy mmaps ``done`` raw; a short or compressed chunk reads as "not done"."""
    task = _task(tmp_path)
    _run_with(task, [0])

    done = zarr.open(str(task.meta_dir / "blocks_done" / "done"), mode="r")
    assert done.dtype == np.uint8
    assert tuple(done.chunks) == tuple(done.shape)
    assert not done.compressors


def test_any_nonzero_value_means_skip(tmp_path):
    """Not "1 means skip" -- a seed narrowed to ``== 1`` passes every other test."""
    task = _task(tmp_path)
    mask = np.zeros(_grid(task), dtype=np.uint8)
    mask[0, 0], mask[2, 0] = 7, 255
    task.block_done_mask = _mask(tmp_path / "mask.zarr", mask)
    task.run_blockwise(multiprocessing=False)

    assert _ran(task) == [10, 30]


def test_mask_dataset_round_trips_through_its_config(tmp_path):
    """The mask reaches a worker as config, so it has to survive the trip."""
    array = np.array([[0], [1], [0], [0]], dtype=np.uint8)
    dataset = _mask(tmp_path / "mask.zarr", array)

    rebuilt = MaskDataset.model_validate_json(dataset.model_dump_json())

    assert rebuilt.dataset_type == "mask"
    assert rebuilt == dataset
    assert rebuilt.read_mask().tolist() == array.tolist()


@pytest.mark.parametrize("dtype", ["bool", "int32"])
def test_read_mask_normalises_a_foreign_dtype_to_uint8(tmp_path, dtype):
    """Asserting uint8 off an already-uint8 mask would not test the conversion."""
    array = np.array([[0], [1], [0], [0]], dtype=dtype)
    read = _mask(tmp_path / f"{dtype}.zarr", array, dtype=dtype).read_mask()

    assert read.dtype == np.uint8
    assert read.tolist() == [[0], [1], [0], [0]]


def test_a_compressed_mask_still_reads_back(tmp_path):
    """The codec restriction is on daisy's ``done``, not on the caller's mask."""
    path = tmp_path / "compressed.zarr"
    array = np.array([[0], [0], [1], [0]], dtype=np.uint8)
    z = zarr.create_array(
        store=str(path), shape=array.shape, chunks=(1, 1), dtype="uint8", overwrite=True
    )
    z[:] = array

    assert MaskDataset(store=path).read_mask().tolist() == array.tolist()
