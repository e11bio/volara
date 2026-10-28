from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate, Roi

from volara.blockwise.blockwise import BlockwiseTask
from volara.blockwise.pipeline import Pipeline
from volara.datasets import MaskDataset


class DummyTask(BlockwiseTask):
    """Minimal concrete subclass for testing the ABC."""

    task_type: Literal["dummy"] = "dummy"
    label: str = "dummy-task"

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return self.label

    @property
    def write_roi(self) -> Roi:
        return Roi((0, 0), (10, 10))

    @property
    def write_size(self) -> Coordinate:
        return Coordinate(10, 10)

    @property
    def context_size(self) -> Coordinate:
        return Coordinate(0, 0)

    def drop_artifacts(self):
        pass

    @contextmanager
    def process_block_func(self):
        def process_block(block):
            pass

        yield process_block


def test_dummy_task_properties():
    t = DummyTask()
    assert t.task_name == "dummy-task"
    assert t.write_roi == Roi((0, 0), (10, 10))
    assert t.write_size == Coordinate(10, 10)
    assert t.context_size == Coordinate(0, 0)


def test_meta_dir_and_config_file(tmp_path):
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    t = DummyTask()
    assert t.meta_dir == tmp_path / "logs" / "dummy-task-meta"
    assert t.config_file == tmp_path / "logs" / "dummy-task-meta" / "config.json"
    assert t.tracking_path == tmp_path / "logs" / "dummy-task-meta" / "blocks_done"


def test_drop_removes_meta_dir(tmp_path):
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    t = DummyTask()
    t.meta_dir.mkdir(parents=True)
    assert t.meta_dir.exists()
    t.drop()
    assert not t.meta_dir.exists()


DROP_ARTIFACTS_CALLS: list[str] = []


class RecordingTask(DummyTask):
    """DummyTask that records every ``drop_artifacts`` call."""

    def drop_artifacts(self):
        DROP_ARTIFACTS_CALLS.append(self.task_name)


def test_drop_calls_drop_artifacts_by_default(tmp_path):
    """drop() is a full reset: meta dir removed and outputs dropped."""
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    DROP_ARTIFACTS_CALLS.clear()
    t = RecordingTask()
    t.meta_dir.mkdir(parents=True)
    t.drop()
    assert not t.meta_dir.exists()
    assert DROP_ARTIFACTS_CALLS == [t.task_name]


def test_drop_outputs_false_skips_drop_artifacts(tmp_path):
    """drop(drop_outputs=False) clears the meta dir but keeps the outputs."""
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    DROP_ARTIFACTS_CALLS.clear()
    t = RecordingTask()
    t.meta_dir.mkdir(parents=True)
    t.drop(drop_outputs=False)
    assert not t.meta_dir.exists()
    assert DROP_ARTIFACTS_CALLS == []


def test_run_creates_meta_dir_and_tracking_store(tmp_path):
    """A run leaves daisy's block-tracking store under the task's meta dir."""
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    t = DummyTask()
    assert not t.meta_dir.exists()

    state = t.run_blockwise(multiprocessing=False)[t.task_name]

    assert state.completed_count == 1
    assert state.failed_count == 0
    assert t.tracking_path.exists()


def test_pipeline_add_operator():
    t1 = DummyTask(label="task-a")
    t2 = DummyTask(label="task-b")
    p = t1 + t2
    assert isinstance(p, Pipeline)
    assert len(p.task_graph.nodes()) == 2
    assert len(p.task_graph.edges()) == 1


def test_pipeline_or_operator():
    t1 = DummyTask(label="task-a")
    t2 = DummyTask(label="task-b")
    p = t1 | t2
    assert isinstance(p, Pipeline)
    assert len(p.task_graph.nodes()) == 2
    assert len(p.task_graph.edges()) == 0


def test_process_block_func_context_manager():
    t = DummyTask()
    with t.process_block_func() as process_block:
        assert callable(process_block)


def test_serialization_roundtrip():
    t = DummyTask()
    json_str = t.model_dump_json()
    t2 = DummyTask.model_validate_json(json_str)
    assert t2.task_name == t.task_name


PROCESSED_BLOCKS: list[int] = []


class CountingTask(DummyTask):
    """DummyTask that records every block it processes."""

    @contextmanager
    def process_block_func(self):
        def process_block(block):
            PROCESSED_BLOCKS.append(block.block_id)

        yield process_block


def test_second_run_skips_blocks_already_done(tmp_path):
    """Resuming a finished task reruns nothing; drop() makes it rerun everything.

    This is the contract volara's own ``blocks_done.zarr`` used to provide and
    daisy's built-in tracking (``Task(tracking_path=...)``) now does. daisy
    counts a skipped block as completed, so the process function's call count
    is what distinguishes the two.
    """
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    PROCESSED_BLOCKS.clear()
    t = CountingTask()

    first = t.run_blockwise(multiprocessing=False)[t.task_name]
    assert first.skipped_count == 0
    assert len(PROCESSED_BLOCKS) == 1

    second = t.run_blockwise(multiprocessing=False)[t.task_name]
    assert second.skipped_count == 1
    assert len(PROCESSED_BLOCKS) == 1, "a done block was processed again"

    t.drop()
    third = t.run_blockwise(multiprocessing=False)[t.task_name]
    assert third.skipped_count == 0
    assert len(PROCESSED_BLOCKS) == 2


RAN_OFFSETS: list[tuple[int, ...]] = []


class MaskedTask(DummyTask):
    """Four blocks along z, a configurable context, records which blocks ran."""

    context: tuple[int, int] = (0, 0)

    @property
    def write_roi(self) -> Roi:
        return Roi((0, 0), (40, 10))

    @property
    def context_size(self) -> Coordinate:
        return Coordinate(self.context)

    @contextmanager
    def process_block_func(self):
        def process_block(block):
            RAN_OFFSETS.append(tuple(int(v) for v in block.write_roi.offset))

        yield process_block


def _mask_dataset(path: Path, array: np.ndarray) -> MaskDataset:
    z = zarr.create_array(
        store=str(path), shape=array.shape, chunks=array.shape, dtype=array.dtype
    )
    z[:] = array
    return MaskDataset(store=path)


def _masked_task(tmp_path, cells, context=(0, 0), **kw) -> MaskedTask:
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    RAN_OFFSETS.clear()
    mask = np.zeros((4, 1), dtype=np.uint8)
    for cell, value in cells.items():
        mask[cell, 0] = value
    return MaskedTask(
        context=context,
        skip_mask=_mask_dataset(tmp_path / f"mask-{len(cells)}.zarr", mask),
        **kw,
    )


def test_block_grid_shape_matches_daisy(tmp_path):
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    for context in [(0, 0), (10, 10), (15, 5)]:
        t = MaskedTask(context=context)
        assert t.block_grid_shape == Coordinate(4, 1)
        with t.task(multiprocessing=False) as daisy_task:
            assert list(daisy_task.block_grid_shape) == [4, 1]


def test_masked_blocks_are_skipped_and_any_nonzero_value_skips(tmp_path):
    t = _masked_task(tmp_path, {0: 1, 2: 7})

    state = t.run_blockwise(multiprocessing=False)[t.task_name]

    assert state.skipped_count == 2
    assert state.completed_count == 4
    assert sorted(RAN_OFFSETS) == [(10, 0), (30, 0)]


def test_mask_cell_zero_is_the_first_block_with_a_block_wide_context(tmp_path):
    """The context equals the block size here. daisy's tracking grid would put
    the first block at cell 1; the mask is anchored at the first block."""
    t = _masked_task(tmp_path, {0: 1}, context=(10, 10))

    t.run_blockwise(multiprocessing=False)

    assert sorted(RAN_OFFSETS) == [(10, 0), (20, 0), (30, 0)]


def test_mask_applies_to_one_run_and_done_blocks_stay_done(tmp_path):
    t = _masked_task(tmp_path, {1: 1, 3: 1})
    first = t.run_blockwise(multiprocessing=False)[t.task_name]
    assert first.skipped_count == 2
    assert sorted(RAN_OFFSETS) == [(0, 0), (20, 0)]

    # Same task without the mask: the two masked blocks run now, the two
    # done ones are skipped via daisy's tracking store.
    RAN_OFFSETS.clear()
    t2 = MaskedTask()
    second = t2.run_blockwise(multiprocessing=False)[t2.task_name]
    assert second.skipped_count == 2
    assert sorted(RAN_OFFSETS) == [(10, 0), (30, 0)]

    RAN_OFFSETS.clear()
    third = t2.run_blockwise(multiprocessing=False)[t2.task_name]
    assert third.skipped_count == 4
    assert RAN_OFFSETS == []


def test_mask_of_the_wrong_shape_is_rejected_before_the_run(tmp_path):
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    bad = _mask_dataset(tmp_path / "bad.zarr", np.zeros((5, 1), dtype=np.uint8))
    t = MaskedTask(skip_mask=bad)

    with pytest.raises(ValueError, match="block grid"):
        t.run_blockwise(multiprocessing=False)
    assert not t.tracking_path.exists()


def test_skip_mask_round_trips_through_the_task_config(tmp_path):
    t = _masked_task(tmp_path, {2: 1})
    rebuilt = MaskedTask.model_validate_json(t.model_dump_json())
    assert rebuilt.skip_mask == t.skip_mask
    assert rebuilt.skip_mask.array("r")[:].tolist() == [[0], [0], [1], [0]]


@pytest.mark.parametrize("dtype", ["bool", "int32"])
def test_a_mask_of_another_dtype_is_cast_by_daisy(tmp_path, dtype):
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    RAN_OFFSETS.clear()
    mask = np.array([[0], [0], [1], [0]], dtype=dtype)
    t = MaskedTask(skip_mask=_mask_dataset(tmp_path / "mask.zarr", mask))

    state = t.run_blockwise(multiprocessing=False)[t.task_name]

    assert state.skipped_count == 1
    assert sorted(RAN_OFFSETS) == [(0, 0), (10, 0), (30, 0)]


def test_a_url_basedir_puts_the_meta_dir_and_tracking_store_in_one_local_place(
    tmp_path, monkeypatch
):
    """``meta_dir``, ``config_file`` and ``tracking_path`` derive from the basedir,
    so a URL basedir lands them under the remote root as plain local paths; the
    run writes there, resumes from there, and no ``s3:`` directory appears.

    pytest tests/test_blockwise_base.py::test_a_url_basedir_puts_the_meta_dir_and_tracking_store_in_one_local_place
    """
    from volara.logging import set_log_basedir, set_remote_root

    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    set_remote_root(tmp_path / "remote")
    with pytest.warns(UserWarning):
        set_log_basedir("s3://bucket/x")

    t = DummyTask()
    local = (tmp_path / "remote" / "s3" / "bucket" / "x").resolve()
    assert t.meta_dir == local / "dummy-task-meta"
    assert t.tracking_path == local / "dummy-task-meta" / "blocks_done"
    assert type(t.meta_dir) is type(Path())  # a plain pathlib.Path, not a UPath

    first = t.run_blockwise(multiprocessing=False)[t.task_name]
    assert first.completed_count == 1 and first.skipped_count == 0
    assert t.tracking_path.exists()
    assert not (launch_dir / "s3:").exists()

    second = t.run_blockwise(multiprocessing=False)[t.task_name]
    assert second.skipped_count == 1

    t.drop(drop_outputs=False)
    assert not t.meta_dir.exists()
    third = t.run_blockwise(multiprocessing=False)[t.task_name]
    assert third.skipped_count == 0


SEEN_SCHEDULERS: list[str | None] = []


class SchedulerRecordingTask(DummyTask):
    """Records the array scheduler in force during setup and per block."""

    @contextmanager
    def process_block_func(self):
        from volara.datasets import current_array_scheduler

        SEEN_SCHEDULERS.append(current_array_scheduler())

        def process_block(block):
            SEEN_SCHEDULERS.append(current_array_scheduler())

        yield process_block


@pytest.mark.parametrize("scheduler", ["single-threaded", "threads", None])
def test_workers_open_arrays_with_the_task_array_scheduler(scheduler):
    from volara.datasets import current_array_scheduler

    SEEN_SCHEDULERS.clear()
    task = SchedulerRecordingTask(array_scheduler=scheduler)
    task.process_roi(Roi((0, 0), (10, 10)))
    # setup and the block both ran under the task's scheduler, and it is gone after
    assert SEEN_SCHEDULERS == [scheduler, scheduler]
    assert current_array_scheduler() is None


def test_array_scheduler_defaults_to_single_threaded_and_round_trips():
    assert DummyTask().array_scheduler == "single-threaded"
    dumped = DummyTask(array_scheduler=None).model_dump_json()
    assert DummyTask.model_validate_json(dumped).array_scheduler is None
    with pytest.raises(ValueError):
        DummyTask(array_scheduler="distributed")
