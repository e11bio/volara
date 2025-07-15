import sqlite3
import time
from contextlib import contextmanager
import daisy
from pathlib import Path
import os
import polars as pl
import psutil


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

            # Group by task and operation, compute mean and std
            agg_df = df.group_by(["task", "operation"]).agg(
                [
                    pl.col("duration").mean().alias("wall_mean"),
                    pl.col("duration").std().alias("wall_std"),
                    pl.col("cpu_usage").mean().alias("cpu_mean"),
                    pl.col("mem_usage").max().alias("max_mem"),
                    pl.col("io_read").mean().alias("read_mean"),
                    pl.col("io_write").mean().alias("write_mean"),
                ]
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
            time_df = time_df.pivot(
                values="time_profile", index="task", columns="operation"
            ).sort("task")
            mem_df = mem_df.pivot(
                values="mem_profile", index="task", columns="operation"
            ).sort("task")
            io_df = io_df.pivot(
                values="io_profile", index="task", columns="operation"
            ).sort("task")

            time_df.write_csv("benchmark_time.csv")
            mem_df.write_csv("benchmark_memory.csv")
            io_df.write_csv("benchmark_io.csv")
        else:
            print("No benchmark data available.")
