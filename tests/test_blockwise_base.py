from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate, Roi
from pydantic import ValidationError

from volara.blockwise import CLAHE, Argmax, ExtractFrags, Relabel, Threshold
from volara.blockwise.blockwise import BlockwiseTask
from volara.blockwise.lambda_task import LambdaTask
from volara.blockwise.pipeline import Pipeline
from volara.datasets import Affs, Labels, MaskDataset, Raw
from volara.dbs import SQLite
from volara.lut import LUT


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


def test_worker_func_writes_the_config_file_the_worker_command_reads(tmp_path):
    from volara.logging import set_log_basedir
    from volara.workers import LocalWorker

    set_log_basedir(tmp_path / "logs")
    t = DummyTask(worker_config=LocalWorker())
    t.meta_dir.mkdir(parents=True)

    t.worker_func()

    assert t.config_file.read_text() == t.model_dump_json()


class PayloadTask(DummyTask):
    """DummyTask that carries a megabyte of state."""

    payload: str = "x" * 1_000_000


def test_spawn_function_pickles_to_a_few_kilobytes_for_a_megabyte_task(tmp_path):
    """daisy pickles the spawn function into one driver-side helper per worker
    slot, so its pickle holds the worker config, the config file path and the
    task name (about 1.4 kB here) and stays that size however large the task is."""
    import cloudpickle

    from volara.logging import set_log_basedir
    from volara.workers import LocalWorker

    set_log_basedir(tmp_path / "logs")
    t = PayloadTask(worker_config=LocalWorker())
    t.meta_dir.mkdir(parents=True)

    assert len(cloudpickle.dumps(t.worker_func())) < 10_000


DROP_ARTIFACTS_CALLS: list[str] = []


class RecordingTask(DummyTask):
    """DummyTask that records every ``drop_artifacts`` call."""

    def drop_artifacts(self):
        DROP_ARTIFACTS_CALLS.append(self.task_name)


@pytest.mark.parametrize(
    "kwargs, dropped",
    [
        ({}, ["dummy-task"]),
        ({"drop_outputs": False}, []),
    ],
)
def test_drop_outputs_decides_whether_drop_calls_drop_artifacts(
    tmp_path, kwargs, dropped
):
    from volara.logging import set_log_basedir

    set_log_basedir(tmp_path / "logs")
    DROP_ARTIFACTS_CALLS.clear()
    t = RecordingTask()
    t.meta_dir.mkdir(parents=True)

    t.drop(**kwargs)

    assert t.meta_dir.exists() is False
    assert DROP_ARTIFACTS_CALLS == dropped


def _argmax(store: Path):
    out = store / "sem"
    task = Argmax(
        probs_data=Raw(store=store / "affs"),
        sem_data=Labels(store=out),
        block_size=Coordinate(10, 10),
    )
    return task, [out]


def _clahe(store: Path):
    out = store / "clahe"
    task = CLAHE(
        in_arr=Raw(store=store / "raw"),
        out_arr=Raw(store=out),
        block_size=Coordinate(10, 10),
        kernel=Coordinate(8, 8),
    )
    return task, [out]


def _threshold(store: Path):
    out = store / "mask"
    task = Threshold(
        in_data=Raw(store=store / "raw"),
        mask=Labels(store=out),
        threshold=0.5,
        block_size=Coordinate(10, 10),
    )
    return task, [out]


def _relabel(store: Path):
    out = store / "seg"
    task = Relabel(
        frags_data=Labels(store=store / "labels"),
        seg_data=Labels(store=out),
        lut=LUT(path=store / "lut.npz"),
        block_size=Coordinate(10, 10),
    )
    return task, [out]


def _extract_frags(store: Path):
    out = store / "frags"
    db = SQLite(path=store / "ef_db.sqlite", ndim=2)
    task = ExtractFrags(
        db=db,
        affs_data=Affs(store=store / "affs"),
        frags_data=Labels(store=out),
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        bias=[-0.5, -0.5],
    )
    return task, [out, db.path]


def _lambda_task(store: Path):
    out = store / "out"
    task = LambdaTask(
        in_data=Raw(store=store / "raw"),
        out_data=Labels(store=out),
        lambda_func=lambda x: (x > 0.5).astype(np.uint8),
        block_size=Coordinate(10, 10),
    )
    return task, [out]


@pytest.mark.parametrize(
    "make_task",
    [_argmax, _clahe, _threshold, _relabel, _extract_frags, _lambda_task],
)
def test_drop_artifacts_removes_what_init_created(
    make_task, zarr_2d, labels_2d, affs_2d, tmp_path
):
    """A new task type gets a factory in this list, not a drop test of its own."""
    task, outputs = make_task(tmp_path / "test.zarr")

    task.init()
    assert [path.exists() for path in outputs] == [True] * len(outputs)

    task.drop_artifacts()
    assert [path.exists() for path in outputs] == [False] * len(outputs)


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


@pytest.mark.parametrize(
    "combine, n_edges",
    [
        (lambda a, b: a + b, 1),
        (lambda a, b: a | b, 0),
        (lambda a, b: a + Pipeline(b), 1),
        (lambda a, b: a | Pipeline(b), 0),
        (lambda a, b: Pipeline(a) + b, 1),
        (lambda a, b: Pipeline(a) | b, 0),
    ],
    ids=[
        "task+task",
        "task|task",
        "task+pipeline",
        "task|pipeline",
        "pipeline+task",
        "pipeline|task",
    ],
)
def test_plus_runs_in_series_and_pipe_runs_in_parallel(combine, n_edges):
    t1 = DummyTask(label="task-a")
    t2 = DummyTask(label="task-b")

    p = combine(t1, t2)

    assert isinstance(p, Pipeline)
    assert list(p.task_graph.nodes()) == [t1, t2]
    assert list(p.task_graph.edges()) == [(t1, t2)][:n_edges]


@pytest.mark.parametrize(
    "combine", [lambda task: task + 3, lambda task: task | 3], ids=["+", "|"]
)
def test_combining_a_task_with_anything_else_is_not_supported(combine):
    with pytest.raises(NotImplementedError, match="other with type <class 'int'>"):
        combine(DummyTask())


def test_get_task_finds_a_registered_task_class_by_its_task_type():
    from volara.blockwise import get_blockwise_tasks_type, get_task

    get_blockwise_tasks_type()  # discovers the tasks registered as entry points

    assert get_task("threshold") is Threshold
    with pytest.raises(ValueError, match="Unknown task: no-such-task"):
        get_task("no-such-task")


def test_process_block_func_context_manager():
    t = DummyTask()
    with t.process_block_func() as process_block:
        assert callable(process_block)


def test_serialization_roundtrip():
    t = DummyTask(label="round-trip", array_scheduler=None)
    assert DummyTask.model_validate_json(t.model_dump_json()) == t


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
    assert rebuilt.skip_mask is not None
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


def test_array_scheduler_defaults_to_single_threaded():
    assert DummyTask().array_scheduler == "single-threaded"


def test_array_scheduler_rejects_an_unknown_scheduler():
    with pytest.raises(ValidationError, match="array_scheduler"):
        DummyTask.model_validate({"array_scheduler": "distributed"})
