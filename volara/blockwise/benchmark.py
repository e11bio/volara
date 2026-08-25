import csv
import json
import os
import statistics
import time
import uuid
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

import daisy
import psutil

BENCHMARK_DIR_ENV_VAR = "VOLARA_BENCHMARK_DIR"
"""
Name of the environment variable that carries the directory a benchmark run
writes its traces into. Its presence is also the "we are benchmarking" signal.

An environment variable (rather than a module global) is what makes benchmarking
work at all: block processing happens in worker processes -- forked
(``num_cache_workers``) or freshly spawned (``worker_config`` -> ``volara-cli
blockwise-worker``) -- and a module global would not survive the spawn. The
environment is inherited by both.
"""

BENCHMARK_LOG_BASEDIR_ENV_VAR = "VOLARA_BENCHMARK_LOG_BASEDIR"
"""
Name of the environment variable carrying the ``volara.logging`` basedir that a
benchmark run's workers must adopt.

Workers otherwise take it from daisy's worker context, which is not reliable
across daisy versions: daisy 2.0 dropped ``logdir`` from the worker context and
falls back to the *worker's* own basedir, so a freshly spawned ``volara-cli
blockwise-worker`` resolves ``volara_logs/<task>-meta`` while the scheduler
created the block-done array somewhere else. Exporting it removes the guesswork.
"""

TRACE_SUFFIX = ".jsonl"

_trace_file: Path | None = None
_trace_pid: int | None = None


def get_benchmark_dir() -> Path | None:
    """
    The directory this process should write benchmark traces into, or ``None``
    when we are not inside a :meth:`BlockwiseTask.benchmark` /
    :meth:`Pipeline.benchmark` run.

    ``None`` is the normal case: it is what keeps ordinary ``run_blockwise``
    executions untraced, so no production run pays for tracing.
    """
    benchmark_dir = os.environ.get(BENCHMARK_DIR_ENV_VAR)
    return Path(benchmark_dir) if benchmark_dir else None


def get_benchmark_log_basedir() -> Path | None:
    """
    The ``volara.logging`` basedir a benchmark run's workers must adopt, or
    ``None`` outside a benchmark run.
    """
    log_basedir = os.environ.get(BENCHMARK_LOG_BASEDIR_ENV_VAR)
    return Path(log_basedir) if log_basedir else None


@contextmanager
def benchmarking(benchmark_dir: Path | str, log_basedir: Path | str):
    """
    Mark this process (and any workers it starts) as benchmarking, so that
    :meth:`BlockwiseTask.get_benchmark_logger` hands out live loggers writing
    into ``benchmark_dir``, and so that workers log to ``log_basedir``.

    Both paths are made absolute before being exported: workers do not
    necessarily share the launching process's working directory.

    Note: cluster backends that scrub the environment (e.g. ``bsub`` without
    ``-env``) will not propagate these to remote workers; benchmarking those
    tasks records only what the scheduling process itself traces.
    """
    exported = {
        BENCHMARK_DIR_ENV_VAR: str(Path(benchmark_dir).absolute()),
        BENCHMARK_LOG_BASEDIR_ENV_VAR: str(Path(log_basedir).absolute()),
    }
    previous = {key: os.environ.get(key) for key in exported}
    os.environ.update(exported)
    try:
        yield Path(exported[BENCHMARK_DIR_ENV_VAR])
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def benchmark_run(
    benchmark_dir: Path | str = Path("volara_benchmark_logs"),
    out_dir: Path | None = None,
    relocate_logs: bool = False,
):
    """
    Set a benchmark run up and tear it down: a fresh trace directory, the
    signal exported to workers, and the report written on the way out however
    the run ended.

    Args:
        relocate_logs: move the ``volara.logging`` basedir under
            ``benchmark_dir`` for the duration. Only wanted for spoofed runs,
            which must not touch the meta directories of the real task.
    """
    from volara.logging import get_log_basedir, set_log_basedir

    benchmark_dir = Path(benchmark_dir)
    trace_dir = benchmark_dir / "traces"
    if trace_dir.exists():
        rmtree(trace_dir)
    # The cached handle in _process_trace_file points into the directory just removed. A second
    # benchmark() in one interpreter would hit that cache, skip the mkdir, and fail to append.
    _reset_trace_file()

    previous_log_basedir = get_log_basedir()
    log_basedir = benchmark_dir if relocate_logs else previous_log_basedir
    set_log_basedir(log_basedir)
    try:
        with benchmarking(trace_dir, log_basedir):
            benchmark_logger = BenchmarkLogger(trace_dir, task=None)
            try:
                yield benchmark_logger
            finally:
                benchmark_logger.print_report(out_dir)
    finally:
        set_log_basedir(previous_log_basedir)


def partial_order(task_orders: dict[str, list[str]]) -> list[str]:
    # Generate all local constraints
    precedence = defaultdict(set)
    pair_counts: Counter[tuple[str, str]] = Counter()
    for task_ops in task_orders.values():
        for i in range(len(task_ops)):
            for j in range(i + 1, len(task_ops)):
                x, y = task_ops[i], task_ops[j]
                precedence[x].add(y)
                pair_counts[(x, y)] += 1

    # All nodes
    all_ops = set()
    for ops in task_orders.values():
        all_ops.update(ops)

    # Compute in-degrees
    in_degree = {op: 0 for op in all_ops}
    for x in precedence:
        for y in precedence[x]:
            in_degree[y] += 1

    # Use a queue of nodes with in-degree 0
    queue = deque(sorted([op for op in all_ops if in_degree[op] == 0]))

    ops_list = []
    while queue:
        # Tie-break: pick op that has most total forward constraints
        current = min(
            queue, key=lambda op: -sum(pair_counts[(op, y)] for y in precedence[op])
        )
        queue.remove(current)
        ops_list.append(current)
        for y in precedence[current]:
            in_degree[y] -= 1
            if in_degree[y] == 0:
                queue.append(y)
    return ops_list


def _process_trace_file(trace_dir: Path) -> Path:
    """
    The trace file this process appends to, created on first use.

    One file per process is the whole concurrency story: nothing is shared
    between writers, so there is no lock to contend on. The pid is part of the
    cache key because a forked worker inherits this module's globals and must
    not keep writing to its parent's file.
    """
    global _trace_file, _trace_pid

    pid = os.getpid()
    if _trace_file is None or _trace_pid != pid or _trace_file.parent != trace_dir:
        _trace_file = trace_dir / f"{pid}-{uuid.uuid4().hex[:8]}{TRACE_SUFFIX}"
        _trace_pid = pid
    # Unconditional: the directory can be removed under a live cache entry (benchmark_run wipes it
    # at the start of every run), and a cache hit must still hand back a writable path.
    trace_dir.mkdir(parents=True, exist_ok=True)
    return _trace_file


def _reset_trace_file() -> None:
    """Forget this process's cached trace file, so the next write mints a fresh one."""
    global _trace_file, _trace_pid
    _trace_file = None
    _trace_pid = None


def read_traces(trace_dir: Path) -> list[dict]:
    """
    Every trace record written under ``trace_dir``, across all processes.

    A record that does not parse is skipped rather than raising: a worker killed
    mid-write can leave a truncated final line, and a partial report is more
    useful than a crash in a ``finally`` block.
    """
    records = []
    for trace_file in sorted(trace_dir.glob(f"*{TRACE_SUFFIX}")):
        with open(trace_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def current_worker_id() -> int:
    """
    The daisy worker id of this process, or ``-1`` outside a worker.

    Read straight out of the environment: ``daisy.Client()`` opens a TCP
    connection to the scheduler, which is far too expensive to do once per
    traced operation.
    """
    if daisy.Context.ENV_VARIABLE not in os.environ:
        return -1
    try:
        return int(daisy.Context.from_env()["worker_id"])
    except (KeyError, ValueError):
        return -1


class BenchmarkLogger:
    """
    Appends one JSON record per traced operation to this process's own trace
    file under ``trace_dir``, and reduces all of them into a report.

    ``trace_dir=None`` makes the logger inert -- every ``log``/``trace`` becomes
    a no-op. That is the normal state outside a benchmark run.
    """

    def __init__(self, trace_dir: Path | str | None, task: str | None):
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.task = task

    def log(
        self,
        worker_id: int,
        operation: str,
        duration: float,
        cpu_usage: float = 0.0,
        mem_usage: float = 0.0,
        io_read: int = 0,
        io_write: int = 0,
    ):
        if self.trace_dir is None:
            return
        record = {
            "task": self.task,
            "worker_id": worker_id,
            "operation": operation,
            "duration": duration,
            "cpu_usage": cpu_usage,
            "mem_usage": mem_usage,
            "io_read": io_read,
            "io_write": io_write,
        }
        # One append-mode write of one line: POSIX makes that atomic with
        # respect to the file offset, so a fork that raced ahead of
        # `_process_trace_file` still cannot interleave two records.
        with open(_process_trace_file(self.trace_dir), "a") as f:
            f.write(json.dumps(record) + "\n")

    @contextmanager
    def trace(self, operation: str):
        if self.trace_dir is not None:
            proc = psutil.Process(os.getpid())
            if hasattr(proc, "io_counters"):
                io_before = proc.io_counters()
            else:
                # MacOS does not support io_counters
                io_before = None
            cpu_before = proc.cpu_times()
            mem_before = proc.memory_info()
            start = time.time()
            try:
                yield
            finally:
                end = time.time()
                mem_after = proc.memory_info()
                cpu_after = proc.cpu_times()
                try:
                    io_after = proc.io_counters()
                    io_read = io_after.read_bytes - io_before.read_bytes  # type: ignore
                    io_write = io_after.write_bytes - io_before.write_bytes  # type: ignore
                except AttributeError:
                    # MacOS does not support io_counters
                    io_read = 0
                    io_write = 0
                cpu_usage = cpu_after.user - cpu_before.user
                mem_usage = mem_after.rss - mem_before.rss
                self.log(
                    current_worker_id(),
                    operation,
                    end - start,
                    cpu_usage,
                    mem_usage,
                    io_read,
                    io_write,
                )
        else:
            yield

    def print_report(self, out_dir: Path | None = None):
        """
        Reduce every process's traces into a time/memory/io report, print the
        timing table, and write all three as csv under ``out_dir``.
        """
        if out_dir is None:
            out_dir = Path("./volara_benchmark_report")

        records = (
            read_traces(self.trace_dir)
            if self.trace_dir is not None and self.trace_dir.exists()
            else []
        )
        if len(records) == 0:
            print("No benchmark data available.")
            return

        # Operation order comes from the order operations were first seen within
        # each task, reconciled across tasks into a single partial order.
        task_orders: dict[str, list[str]] = {}
        seen = set()
        for record in records:
            key = (record["task"], record["operation"])
            if key not in seen:
                task_orders.setdefault(record["task"], []).append(record["operation"])
                seen.add(key)
        ops_order = partial_order(task_orders)
        task_order = list(task_orders.keys())

        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for record in records:
            grouped[(record["task"], record["operation"])].append(record)

        time_profiles: dict[tuple[str, str], str] = {}
        mem_profiles: dict[tuple[str, str], str] = {}
        io_profiles: dict[tuple[str, str], str] = {}
        for key, group in grouped.items():
            durations = [record["duration"] for record in group]
            wall_mean = round(statistics.fmean(durations), 3)
            wall_std = round(statistics.stdev(durations), 3) if len(group) > 1 else 0.0
            cpu_mean = round(statistics.fmean(r["cpu_usage"] for r in group), 3)
            max_mem = max(record["mem_usage"] for record in group)
            read_mean = statistics.fmean(record["io_read"] for record in group)
            write_mean = statistics.fmean(record["io_write"] for record in group)

            time_profiles[key] = (
                f"{wall_mean}s ± {wall_std} (idle: {round(wall_mean - cpu_mean, 3)}s)"
            )
            mem_profiles[key] = f"{round(max_mem / (1024 * 1024), 2)} MB"
            io_profiles[key] = (
                f"read/write: {round(read_mean / (1024 * 1024), 2)}/"
                f"{round(write_mean / (1024 * 1024), 2)} MB"
            )

        header = ["task"] + ops_order
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, profiles in (
            ("time.csv", time_profiles),
            ("memory.csv", mem_profiles),
            ("io.csv", io_profiles),
        ):
            rows = [
                [task] + [profiles.get((task, op), "") for op in ops_order]
                for task in task_order
            ]
            with open(out_dir / name, "w", newline="") as f:
                csv.writer(f).writerows([header] + rows)
            if name == "time.csv":
                print_table([header] + rows)
        print(f"Benchmark report written to {out_dir}")


def print_table(rows: list[list[str]]):
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))
