import glob
import logging
import multiprocessing
import subprocess
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Iterator

import daisy
import numpy as np
from daisy.block import BlockStatus
from funlib.geometry import Coordinate, Roi
from funlib.math import cantor_number
from funlib.persistence import open_ds, prepare_ds

from volara.logging import get_log_basedir, set_log_basedir

from ..utils import PydanticRoi, StrictBaseModel
from ..workers import Worker
from .benchmark import BenchmarkLogger

if TYPE_CHECKING:
    from .pipeline import Pipeline

logger = logging.getLogger(__name__)


class BlockwiseTask(StrictBaseModel, ABC):
    roi: PydanticRoi | None = None
    """
    An optional `roi` defining the total region to output.
    """
    num_workers: int = 1
    """
    The number of workers that will be started to process blocks in parallel
    """
    num_cache_workers: int | None = None
    """
    The number of threads running the process_block function per worker.
    This allows you to start e.g. 4 gpu workers, each with 1 copy of the gpu loaded
    and running 8 threads to read, pre/postprocess, and write your data; maximizing
    your gpu utilization.
    """
    worker_config: Worker | None = None
    """
    The configuration for each worker you start. This allows you to specify
    arguments for running workers on various platforms such as slurm/lsf clusters
    or AWS EC2.
    """
    _out_array_dtype: np.dtype = np.dtype(np.uint8)
    """
    The output array data type
    """

    fit: str
    """
    The strategy to use for blocks that overhang your total write roi.
    """
    read_write_conflict: bool
    """
    Whether blocks have read/write dependencies on neighborhing blocks requiring
    a specific ordering to the block processing to compute a seamless result.
    """

    def __hash__(self):
        return hash(self.task_name)

    @property
    @abstractmethod
    def task_name(self) -> str:
        """
        A unique identifier for a task. This allows us to store log files
        in an unambiguous location as well as storing a block_done dataset
        that allows us to cache completed blocks and resume processing
        at a later time.
        """
        pass

    @property
    @abstractmethod
    def write_roi(self) -> Roi:
        """
        The total roi of any data output by a task.
        """
        pass

    @property
    @abstractmethod
    def write_size(self) -> Coordinate:
        """
        The write size of each block processed as part of a task.
        """
        pass

    @property
    @abstractmethod
    def context_size(self) -> Coordinate | tuple[Coordinate, Coordinate]:
        """
        The amount of context needed to process each block for a task.
        """
        pass

    def init(self):
        """
        Any one time initializations that need to be made before starting a
        task such as creating dbs and zarrs.
        """
        pass

    @abstractmethod
    def process_block_func(self):
        """
        A constructor for a function that will take a single block
        as input and process it.
        """
        pass

    @abstractmethod
    def drop_artifacts(self):
        """
        A helper function to reset anything produced by a task
        to a clean state equivalent to not having run the task at all
        """
        pass

    @property
    def block_write_roi(self) -> Roi:
        """
        The write roi of a block with zero offset
        """
        return Roi((0,) * self.write_size.dims, self.write_size)

    @property
    def meta_dir(self) -> Path:
        """
        The path to the meta directory where we will store log files
        and a block done cache for resuming work if processing is
        interrupted.
        """
        return get_log_basedir() / f"{self.task_name}-meta"

    @property
    def config_file(self) -> Path:
        """
        The config file that will be used to serialize this task for
        logging purposes.
        """
        return self.meta_dir / "config.json"

    @property
    def block_ds(self) -> Path:
        """
        The dataset that will be used to track which blocks have been
        successfully completed.
        """
        return self.meta_dir / "blocks_done.zarr"

    def process_roi(self, roi: Roi, context: Coordinate | None = None):
        """
        A helper function to process a given roi without needing to start a
        whole blockwise job.
        """
        block = daisy.Block(
            roi, roi if context is None else roi.grow(context, context), roi
        )
        process_block = self.process_block_func()
        process_block(block)

    def drop(self, drop_outputs: bool = False) -> None:
        """
        A helper function to drop any artifacts produced by a task
        and return to a state identical to before having executed the
        task.
        """
        # reset the blocks_done ds so that the task is rerun
        if self.meta_dir.exists():
            rmtree(self.meta_dir)
        self.drop_artifacts()

    def check_block_func(self):
        """
        A function to check whether a block has been completed.
        """

        def check_block(block):
            block_array = open_ds(self.block_ds, mode="r")
            coordinate = (
                block.write_roi.offset - block_array.offset
            ) / block_array.voxel_size
            chunk_size = Coordinate(block_array._source_data.chunks[-coordinate.dims :])
            chunk_index = coordinate // chunk_size
            chunk_key = "/".join(str(i) for i in chunk_index)
            prefix = "/".join("*" for _ in range(block_array.dims - len(chunk_index)))

            glob_pattern = f"{prefix}/{chunk_key}" if len(prefix) > 0 else chunk_key
            glob_pattern = f"{block_array._source_data.store.path}/{glob_pattern}"
            matches = glob.glob(glob_pattern, recursive=True)
            return len(list(matches)) > 0

        return check_block

    def mark_block_done_func(self):
        """
        A helper function to mark a block as completed so that it
        can be skipped if we have to pause and resume processing later.
        """

        def write_check_block(block):
            if not block.status == BlockStatus.FAILED:
                # Unless the block is explicitly marked as failed, we assume
                # successful processing if there was no error
                block_array = open_ds(self.block_ds, mode="a")
                write_roi = block.write_roi.intersect(block_array.roi)
                write_roi.shape = block_array.voxel_size
                block_array[write_roi] = np.full(
                    write_roi.shape // block_array.voxel_size,
                    fill_value=block.block_id[1] + 1,
                )
                block.status = BlockStatus.SUCCESS

        return write_check_block

    def worker_func(self):
        """
        The function defining how workers are started.
        """
        if self.worker_config is not None:
            config_file = self.config_file

            with open(config_file, "w") as f:
                f.write(self.model_dump_json())

            logging.info("Running block with config %s..." % config_file)

            def run_worker():
                assert (
                    self.worker_config is not None
                )  # TODO: remove this, shouldn't be needed for type checker
                cmd = self.worker_config.get_command(config_file, self.task_name)
                return subprocess.run(cmd)

            return run_worker

        else:
            return self.process_blocks

    def process_blocks(self):
        """
        Start our workers and run through every block until a task
        is complete.
        """
        benchmark_logger = self.get_benchmark_logger()
        with ExitStack() as stack:
            with benchmark_logger.trace("Process Block Setup"):
                process_block = stack.enter_context(self.process_block_func())

            def worker_loop():
                worker_benchmark_logger = self.get_benchmark_logger()
                client = daisy.Client()
                # TODO: this shouldn't be necessary, daisy should be doing this for us
                try:
                    set_log_basedir(client.context["logdir"])  # type: ignore[non-subscriptable]
                except KeyError as e:
                    raise ValueError(client.context) from e
                mark_block_done = self.mark_block_done_func()

                while True:
                    logger.info("getting block")
                    with ExitStack() as worker_stack:
                        with worker_benchmark_logger.trace("Acquire Block"):
                            block = worker_stack.enter_context(client.acquire_block())
                        logger.info(f"got block {block}")

                        if block is None:
                            break

                        with benchmark_logger.trace("Process Block"):
                            process_block(block)
                        with benchmark_logger.trace("Mark Block Done"):
                            mark_block_done(block)

            if self.num_cache_workers is not None:
                workers = [
                    multiprocessing.Process(target=worker_loop)
                    for _ in range(self.num_cache_workers)
                ]

                for worker in workers:
                    worker.start()

                for worker in workers:
                    worker.join()

            else:
                worker_loop()

    def init_block_array(self):
        """
        Build the block done zarr for tracking completed blocks.
        """
        # prepare blocks done ds

        def get_dtype(write_roi, write_size):
            # need to factor in block offset, so use cantor number of last block
            # + 1 to be safe
            num_blocks = cantor_number(write_roi.shape / write_size + 1)

            for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
                if num_blocks <= np.iinfo(dtype).max:
                    return dtype
            raise ValueError(
                f"Number of blocks ({num_blocks}) is too large for available data types."
            )

        block_voxel_size = self.write_size

        try:
            prepare_ds(
                self.block_ds,
                shape=(self.write_roi.shape + block_voxel_size - 1) / block_voxel_size,
                offset=self.write_roi.offset,
                voxel_size=block_voxel_size,
                chunk_shape=self.write_size / block_voxel_size,
                dtype=get_dtype(self.write_roi, self.write_size),
                mode="a",
            )
        except PermissionError as e:
            # The dataset already exists but with different parameters.
            existing_block_ds = open_ds(self.block_ds, mode="r")
            error_msg = (
                f"Trying to overwrite existing {self.block_ds} array with incompatible data:\n"
                f"Shape (existing): {existing_block_ds.shape} vs (new) {self.write_roi.shape / block_voxel_size}\n"
                f"Chunk Shape (existing): {existing_block_ds._source_data.chunks} vs (new) {self.write_size / block_voxel_size}\n"
                f"Data Type (existing): {existing_block_ds.dtype} vs (new) {get_dtype(self.write_roi, self.write_size)}\n"
            )
            raise ValueError(error_msg) from e

    @contextmanager
    def task(
        self,
        upstream_tasks: daisy.Task | list[daisy.Task] | None = None,
        multiprocessing: bool = True,
    ) -> Iterator[daisy.Task]:
        """
        Builds a `daisy.Task` that puts together everything necessary to run a task
        blockwise.
        """
        benchmark_logger = self.get_benchmark_logger()

        with benchmark_logger.trace("init"):
            self.init_block_array()
            self.init()

            context = self.context_size
            if not isinstance(context, Coordinate):
                assert isinstance(context, tuple)
                context_low, context_high = context[0], context[1]
            else:
                context_low, context_high = context, context

        with ExitStack() as stack:
            if multiprocessing:
                process_func = self.worker_func()
            else:
                process_block_func = self.process_block_func()
                with benchmark_logger.trace("Process Block Setup"):
                    process_block = stack.enter_context(process_block_func)

                mark_block = self.mark_block_done_func()

                def process_func(block):
                    with benchmark_logger.trace("Process Block"):
                        process_block(block)
                    with benchmark_logger.trace("Mark Block Done"):
                        mark_block(block)

            task = daisy.Task(
                self.task_name,
                total_roi=self.write_roi.grow(context_low, context_high),
                read_roi=self.block_write_roi.grow(context_low, context_high),
                write_roi=self.block_write_roi,
                process_function=process_func,
                read_write_conflict=self.read_write_conflict,
                fit=self.fit,
                num_workers=self.num_workers,
                check_function=self.check_block_func(),
                max_retries=2,
                timeout=None,
                upstream_tasks=(
                    (
                        upstream_tasks
                        if isinstance(upstream_tasks, list)
                        else [upstream_tasks]
                    )
                    if upstream_tasks is not None
                    else None
                ),
            )

            yield task

    def get_benchmark_logger(self) -> BenchmarkLogger:
        benchmark_db_path = Path("volara_benchmark_logs/benchmark.db")
        return BenchmarkLogger(
            benchmark_db_path if benchmark_db_path.exists() else None,
            task=self.task_name,
        )

    def spoof(self, spoof_dir: Path):
        """
        Whether or not to spoof the data inputs to this task.
        """
        data = {}
        for name, field in self.__class__.model_fields.items():
            value = getattr(self, name)
            if hasattr(value, "spoof"):
                # If the value has a spoof method, call it to get a mock value
                data[name] = value.spoof(spoof_dir)
            else:
                data[name] = value
        return self.__class__(**data)

    def benchmark(self, multiprocessing: bool = True) -> dict:
        """
        A helper function for benchmarking and debugging a blockwise task or pipeline.

        Used as a "dry run" of `run_blockwise` without saving any outputs.

        - Will not skip blocks that have already been processed. You can benchmark a task
            that has already been run.
        - Will not save any run artifacts such as block done datasets, output datasets,
            graph nodes or edges, or look up tables. The only thing that will be saved are the
            worker logs and a timing report.
        """
        from volara.logging import set_log_basedir

        log_basedir = get_log_basedir()
        set_log_basedir("volara_benchmark_logs")
        benchmark_db_path = Path("volara_benchmark_logs/benchmark.db")
        if benchmark_db_path.exists():
            benchmark_db_path.unlink()
        benchmark_logger = BenchmarkLogger(task=None, db_path=benchmark_db_path)
        benchmark_logger._init_db()

        spoof_dir = Path("volara_benchmark_logs/spoof")
        debug_self = self.spoof(spoof_dir)

        try:
            with debug_self.task(multiprocessing=multiprocessing) as task:
                tasks = [task]
                if multiprocessing:
                    result = daisy.run_blockwise(tasks)  # noqa
                else:
                    server = daisy.SerialServer()
                    _cl_monitor = daisy.cl_monitor.CLMonitor(server)  # type: ignore[unresolved-attribute]
                    result = server.run_blockwise(tasks)

        except Exception as e:
            raise e

        finally:
            debug_self.drop()
            benchmark_logger.print_report()
            set_log_basedir(log_basedir)

        return result

    def run_blockwise(
        self,
        multiprocessing: bool = True,
    ):
        """
        Execute this task blockwise.
        """
        with self.task(multiprocessing=multiprocessing) as task:
            tasks = [task]
            if multiprocessing:
                result = daisy.run_blockwise(tasks)  # noqa
            else:
                server = daisy.SerialServer()
                _cl_monitor = daisy.cl_monitor.CLMonitor(server)  # type: ignore[unresolved-attribute]
                result = server.run_blockwise(tasks)
            return result

    def __add__(self, other: "BlockwiseTask | Pipeline") -> "Pipeline":
        """
        The task or pipeline (`task`) gets run in series after `self`.

        This means that every node in `self` without outgoing edges
        gets an edge to all nodes in `task` without incoming edges.
        """
        from .pipeline import Pipeline

        if isinstance(other, Pipeline):
            return Pipeline(self) + other
        elif isinstance(other, BlockwiseTask):
            return Pipeline(self) + Pipeline(other)
        else:
            raise NotImplementedError(
                f"We do not support other with type {type(other)}"
            )

    def __or__(self, other: "BlockwiseTask | Pipeline") -> "Pipeline":
        """
        The task or pipeline (`task`) gets run in parallel with `self`.

        Task graphs are merged, but no edges are added.
        """
        from .pipeline import Pipeline

        if isinstance(other, Pipeline):
            return Pipeline(self) | other
        elif isinstance(other, BlockwiseTask):
            return Pipeline(self) | Pipeline(other)
        else:
            raise NotImplementedError(
                f"We do not support other with type {type(other)}"
            )
