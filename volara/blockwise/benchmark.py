import sqlite3
import time
from contextlib import contextmanager
import daisy
from pathlib import Path
import os
import polars as pl
import psutil
from collections import defaultdict, Counter
from collections import deque


def partial_order(task_orders: dict[str, list[str]]) -> list[str]:
    # Generate all local constraints
    precedence = defaultdict(set)
    pair_counts = Counter()
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


class BenchmarkLogger:
    def __init__(self, db_path: str | None, task: str | None):
        self.task = task
        if db_path is not None:
            if not Path(db_path).parent.exists():
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        else:
            self.conn = None

    def _init_db(self):
        if self.conn is not None:
            # Table with flexible key:value columns
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark (
                    task TEXT,
                    worker_id INT,
                    operation TEXT,
                    duration REAL,
                    cpu_usage REAL,
                    mem_usage REAL,
                    io_read INT,
                    io_write INT
                )
            """)

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
        if self.conn is not None:
            self.conn.execute(
                "INSERT INTO benchmark VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.task,
                    worker_id,
                    operation,
                    duration,
                    cpu_usage,
                    mem_usage,
                    io_read,
                    io_write,
                ),
            )
            self.conn.commit()

    @contextmanager
    def trace(self, operation: str):
        if self.conn is not None:
            if daisy.Context.ENV_VARIABLE in os.environ:
                worker_id = daisy.Client().worker_id
            else:
                worker_id = -1
            proc = psutil.Process(os.getpid())
            try:
                io_before = proc.io_counters()
            except AttributeError:
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
                    io_read = io_after.read_bytes - io_before.read_bytes
                    io_write = io_after.write_bytes - io_before.write_bytes
                except AttributeError:
                    # MacOS does not support io_counters
                    io_read = 0
                    io_write = 0
                cpu_usage = cpu_after.user - cpu_before.user
                mem_usage = mem_after.rss - mem_before.rss
                self.log(
                    worker_id,
                    operation,
                    end - start,
                    cpu_usage,
                    mem_usage,
                    io_read,
                    io_write,
                )
        else:
            yield

    def print_report(self):
        if self.conn is not None:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM benchmark;")
            rows = list(cursor.fetchall())
            cursor.close()

            # Extract orderings from `rows`
            task_orders = {}
            seen = set()

            for row in rows:
                task, _, op, *_ = row
                if (task, op) not in seen:
                    task_orders.setdefault(task, []).append(op)
                    seen.add((task, op))

            ops_order = partial_order(task_orders)
            task_order = list(task_orders.keys())

            # Convert to Polars DataFrame
            df = pl.DataFrame(
                rows,
                schema=[
                    "task",
                    "worker_id",
                    "operation",
                    "duration",
                    "cpu_usage",
                    "mem_usage",
                    "io_read",
                    "io_write",
                ],
            )
            task_rank = {task: i for i, task in enumerate(task_order)}

            # Group by task and operation, compute mean and std
            agg_df = (
                df.group_by(["task", "operation"])
                .agg(
                    [
                        pl.col("duration").mean().alias("wall_mean"),
                        pl.col("duration").std().alias("wall_std"),
                        pl.col("cpu_usage").mean().alias("cpu_mean"),
                        pl.col("mem_usage").max().alias("max_mem"),
                        pl.col("io_read").mean().alias("read_mean"),
                        pl.col("io_write").mean().alias("write_mean"),
                    ]
                )
                .with_columns(
                    pl.col("task")
                    .map_elements(lambda x: task_rank.get(x, float("inf")))
                    .alias("_rank")
                )
                .sort(by="_rank")
                .drop("_rank")
            )

            # Combine mean ± std into a formatted string
            time_df = agg_df.with_columns(
                [
                    pl.format(
                        "{}s ± {} (idle: {}s)",
                        pl.col("wall_mean").round(3),
                        pl.col("wall_std").fill_null(0).round(3),
                        (
                            pl.col("wall_mean").round(3) - pl.col("cpu_mean").round(3)
                        ).round(3),
                    ).alias("time_profile")
                ]
            )
            mem_df = agg_df.with_columns(
                [
                    pl.format(
                        "{} MB",
                        (pl.col("max_mem") / (1024 * 1024)).round(2),
                    ).alias("mem_profile")
                ]
            )
            io_df = agg_df.with_columns(
                [
                    pl.format(
                        "read/write: {}/{} MB",
                        (pl.col("read_mean") / (1024 * 1024)).round(2),
                        (pl.col("write_mean") / (1024 * 1024)).round(2),
                    ).alias("io_profile")
                ]
            )

            # Pivot to wide table: rows = task, columns = operation, values = duration_str
            time_df = (
                (
                    time_df.pivot(
                        values="time_profile", index="task", columns="operation"
                    )
                )
                .select(["task"] + ops_order)
            )
            mem_df = (
                mem_df.pivot(values="mem_profile", index="task", columns="operation")
                .select(["task"] + ops_order)
            )
            io_df = (
                io_df.pivot(values="io_profile", index="task", columns="operation")
                .select(["task"] + ops_order)
            )

            time_df.write_csv("benchmark_time.csv")
            mem_df.write_csv("benchmark_memory.csv")
            io_df.write_csv("benchmark_io.csv")
        else:
            print("No benchmark data available.")
