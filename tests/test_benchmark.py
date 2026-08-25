import multiprocessing
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from funlib.geometry import Coordinate
from funlib.persistence import open_ds
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import LambdaTask
from volara.blockwise.benchmark import (
    BENCHMARK_DIR_ENV_VAR,
    BenchmarkLogger,
    benchmarking,
    get_benchmark_dir,
    read_traces,
)
from volara.datasets import Labels, Raw
from volara.workers import LocalWorker

# benchmark() writes its traces, spoofed inputs and report relative to the cwd
TRACE_DIR = Path("volara_benchmark_logs/traces")


@pytest.fixture()
def tiny_task(tmp_path, monkeypatch):
    """A 20x10 -> 2 block LambdaTask, with the cwd moved into tmp_path.

    The input is declared ``writable=False`` so that ``spoof`` symlinks it
    instead of expecting the benchmark run to fabricate it.
    """
    monkeypatch.chdir(tmp_path)
    data = np.linspace(0, 1, 200, dtype=np.float32).reshape(20, 10)
    in_path = tmp_path / "data.zarr" / "raw"
    prepare_ds(
        in_path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )[:] = data

    def make(suffix="out", **kwargs):
        kwargs.setdefault("out_array_dtype", np.dtype(np.uint8))
        return LambdaTask(
            in_data=Raw(store=in_path, writable=False),
            out_data=Labels(store=tmp_path / "data.zarr" / suffix),
            lambda_func=lambda x: (x > 0.5).astype(np.uint8),
            block_size=Coordinate(10, 10),
            **kwargs,
        )

    return make


def traced_operations(trace_dir: Path = TRACE_DIR) -> list[tuple[str, str]]:
    """The (task, operation) pairs recorded across every process's trace file."""
    assert trace_dir.exists(), f"{trace_dir} was never created"
    return [(record["task"], record["operation"]) for record in read_traces(trace_dir)]


def test_benchmark_records_traces(tiny_task):
    """benchmark() must actually record traces and write the report.

    Regression test: every ``trace`` in volara goes through
    ``BlockwiseTask.get_benchmark_logger``, which used to hard-code a ``None``
    sink. The logger was therefore always inert and ``benchmark()`` recorded
    nothing, no matter what ran.

    pytest tests/test_benchmark.py::test_benchmark_records_traces
    """
    tiny_task().benchmark(multiprocessing=False)

    traces = traced_operations()
    assert len(traces) > 0, "benchmark() recorded no traces"

    operations = {operation for _, operation in traces}
    assert {"init", "Process Block", "Mark Block Done"} <= operations

    # both blocks of the 20x10 volume should show up
    assert sum(1 for _, op in traces if op == "Process Block") == 2

    for name in ("time.csv", "memory.csv", "io.csv"):
        report = Path("volara_benchmark_report") / name
        assert report.exists(), f"{name} was not written"
        assert "Process Block" in report.read_text()


def test_benchmark_twice_in_one_process(tiny_task):
    """Two benchmark() calls in one interpreter must both work.

    Regression test: ``benchmark_run`` wipes ``volara_benchmark_logs/traces`` at the start of every
    run, but ``_process_trace_file`` caches its handle on ``(pid, trace_dir)``. On a second call
    both still matched, so the cache hit skipped the ``mkdir`` and returned a path whose directory
    had just been removed — the first append raised ``FileNotFoundError`` and the report came back
    empty. Every other test gets a fresh ``tmp_path`` cwd, so none of them hit the cache.

    pytest tests/test_benchmark.py::test_benchmark_twice_in_one_process
    """
    tiny_task().benchmark(multiprocessing=False)
    first = traced_operations()
    assert len(first) > 0, "the first benchmark() recorded nothing"

    # same interpreter, same cwd -> same (pid, trace_dir) cache key as the run above
    tiny_task().benchmark(multiprocessing=False)
    second = traced_operations()

    assert len(second) > 0, "the second benchmark() in one process recorded nothing"
    assert "init" in {operation for _, operation in second}

    # The report must be this run's, written fresh -- the bug left the trace directory missing, so
    # print_report found nothing and said "No benchmark data available".
    report = Path("volara_benchmark_report") / "time.csv"
    assert report.exists() and report.read_text().strip(), "the second run wrote no report"

    # Not asserted: that both runs record the same operations. They do not, and correctly so -- the
    # first run marks the blocks done, so the second honours that cache and skips them. That is
    # `test_benchmark_runs_the_task_for_real`'s territory, not this test's.


def test_benchmark_runs_the_task_for_real(tiny_task):
    """The default benchmark path is an ordinary run, not a dry run.

    Benchmarking a job as it runs normally for the first time is the common
    case, so it is what ``benchmark()`` does with no arguments: the outputs and
    the block done dataset are the real ones and they survive the run.

    pytest tests/test_benchmark.py::test_benchmark_runs_the_task_for_real
    """
    task = tiny_task()
    task.benchmark(multiprocessing=False)

    assert task.out_data.store.exists(), "benchmark() produced no output data"
    assert task.block_ds.exists(), "benchmark() left no block done dataset"
    assert "volara_benchmark_logs" not in task.block_ds.parts, (
        "benchmark() put the block done dataset under the benchmark log "
        f"basedir instead of the ordinary one: {task.block_ds}"
    )

    written = task.out_data.array("r")[:]
    assert written.sum() == 100, f"output was not written: {written.sum()}"

    # every block is marked done, so an ordinary rerun has nothing left to do
    blocks_done = open_ds(task.block_ds, mode="r")[:]
    assert blocks_done.all(), f"blocks were left unmarked: {blocks_done}"


def test_benchmark_spoof_leaves_no_artifacts(tiny_task):
    """spoof=True still supports benchmarking a task that has already run.

    pytest tests/test_benchmark.py::test_benchmark_spoof_leaves_no_artifacts
    """
    task = tiny_task()
    task.benchmark(multiprocessing=False, spoof=True)

    assert len(traced_operations()) > 0, "spoofed benchmark recorded no traces"
    assert not task.out_data.store.exists(), "spoofed benchmark wrote real outputs"
    assert not task.block_ds.exists(), "spoofed benchmark wrote a real block done ds"


def test_pipeline_benchmark_records_traces(tiny_task):
    """Pipeline.benchmark() records traces for every task in the pipeline.

    pytest tests/test_benchmark.py::test_pipeline_benchmark_records_traces
    """
    pipeline = tiny_task("first") + tiny_task("second")
    pipeline.benchmark(multiprocessing=False, out_dir=Path("report"))

    tasks = {task for task, _ in traced_operations()}
    assert len(tasks) == 2, f"expected both tasks to be traced, got {tasks}"
    assert (Path("report") / "time.csv").exists()


def test_benchmark_traces_the_worker_loop(tiny_task):
    """The worker block loop is traced, not just the scheduling process.

    pytest tests/test_benchmark.py::test_benchmark_traces_the_worker_loop
    """
    task = tiny_task(num_workers=1)
    task.benchmark(multiprocessing=True)

    traces = traced_operations()
    # "Acquire Block" is only ever traced from inside the worker block loop
    operations = {operation for _, operation in traces}
    assert "Acquire Block" in operations, (
        f"no worker-loop traces recorded, got {operations}"
    )
    # and the blocks really ran -- guard against a green assertion over a run
    # where every block failed before doing any work
    assert sum(1 for _, op in traces if op == "Process Block") == 2


@pytest.mark.parametrize("spoof", [False, True])
def test_benchmark_traces_a_worker_config_run(tiny_task, spoof):
    """benchmark() + worker_config records rows, in both modes.

    This is the distributed path, and the one that matters: workers are
    separate interpreters started as ``volara-cli blockwise-worker``, so they
    inherit nothing but the environment, and they have to resolve the same log
    basedir as the scheduler. When they did not, every block died on
    ``Expected a zarr Array at volara_logs/<task>-meta/blocks_done.zarr, got
    Group`` while ``benchmark()`` still returned and still wrote a report --
    which is why this asserts on ``Process Block``, not just on the run
    completing.

    pytest tests/test_benchmark.py::test_benchmark_traces_a_worker_config_run
    """
    # out_array_dtype is left unset: a np.dtype does not survive the
    # model_dump_json that worker_config dispatch writes the config with
    task = tiny_task(num_workers=1, worker_config=LocalWorker(), out_array_dtype=None)
    task.benchmark(multiprocessing=True, spoof=spoof)

    traces = traced_operations()
    assert sum(1 for _, op in traces if op == "Process Block") == 2, (
        f"blocks did not run in the worker; recorded {sorted(set(traces))}"
    )
    assert sum(1 for _, op in traces if op == "Mark Block Done") == 2


def test_benchmarking_signal_reaches_child_processes(tmp_path):
    """A freshly spawned interpreter inherits the "we are benchmarking" signal.

    This is why the signal is an environment variable and not a module global:
    ``worker_config`` dispatch starts workers as new interpreters
    (``volara-cli blockwise-worker``), which would not see a global.

    pytest tests/test_benchmark.py::test_benchmarking_signal_reaches_child_processes
    """
    probe = (
        "from volara.blockwise.benchmark import get_benchmark_dir;"
        "print(get_benchmark_dir())"
    )

    outside = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert outside.stdout.strip() == "None"

    trace_dir = tmp_path / "traces"
    with benchmarking(trace_dir, tmp_path):
        inside = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
    assert inside.stdout.strip() == str(trace_dir)


def test_run_blockwise_stays_unbenchmarked(tiny_task):
    """Ordinary runs must not be traced -- that is the point of the guard.

    pytest tests/test_benchmark.py::test_run_blockwise_stays_unbenchmarked
    """
    task = tiny_task()
    assert get_benchmark_dir() is None
    assert task.get_benchmark_logger().trace_dir is None

    task.run_blockwise(multiprocessing=False)

    assert not TRACE_DIR.exists(), "run_blockwise wrote benchmark traces"


def test_benchmark_does_not_leak_the_signal(tiny_task, monkeypatch):
    """benchmark() must leave the environment as it found it.

    pytest tests/test_benchmark.py::test_benchmark_does_not_leak_the_signal
    """
    monkeypatch.delenv(BENCHMARK_DIR_ENV_VAR, raising=False)
    tiny_task().benchmark(multiprocessing=False)

    assert get_benchmark_dir() is None
    assert tiny_task("after").get_benchmark_logger().trace_dir is None


def test_report_of_an_empty_run_is_empty(tmp_path, capsys):
    """A run that traced nothing prints an empty report instead of raising.

    pytest tests/test_benchmark.py::test_report_of_an_empty_run_is_empty
    """
    empty = tmp_path / "traces"
    empty.mkdir()
    BenchmarkLogger(empty, task=None).print_report(tmp_path / "report")

    assert "No benchmark data available." in capsys.readouterr().out
    assert not (tmp_path / "report").exists()


def _emit(trace_dir: Path, count: int):
    logger = BenchmarkLogger(trace_dir, task="task-a")
    for _ in range(count):
        logger.log(worker_id=0, operation="Process Block", duration=1.0)


def test_report_aggregates_across_concurrent_writers(tmp_path, capsys):
    """The driver reduces every process's trace file into one report.

    Concurrent writers are the case that used to lock the sqlite db. Here each
    process owns a file of its own, so there is nothing to contend on and
    nothing to lose.

    pytest tests/test_benchmark.py::test_report_aggregates_across_concurrent_writers
    """
    trace_dir = tmp_path / "traces"
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_emit, args=(trace_dir, 50)) for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0

    assert len(read_traces(trace_dir)) == 4 * 50, "records were lost"
    assert len(list(trace_dir.glob("*.jsonl"))) == 4, "processes shared a trace file"

    BenchmarkLogger(trace_dir, task=None).print_report(tmp_path / "report")
    assert (tmp_path / "report" / "time.csv").exists()
    assert "1.0s ± 0.0" in capsys.readouterr().out
