import logging
import multiprocessing
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import rmtree
from typing import Optional, Union

import daisy
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.math import cantor_number
from funlib.persistence import open_ds, prepare_ds

from volara.logging import LOG_BASEDIR

from ..utils import PydanticCoordinate, StrictBaseModel
from ..workers import Worker

logger = logging.getLogger(__name__)


class BlockwiseTask(ABC, StrictBaseModel):
    roi: Optional[tuple[PydanticCoordinate, PydanticCoordinate]] = None
    num_workers: Optional[int] = None
    num_cache_workers: Optional[int] = None
    worker_config: Optional[Worker] = None
    _out_array_dtype: np.dtype = np.dtype(np.uint8)

    fit: str
    read_write_conflict: bool

    # TODO: do we still want task_type as a property?

    # @property
    # @abstractmethod
    # def task_type(self) -> str:
    # pass

    @property
    @abstractmethod
    def task_name(self) -> str:
        pass

    @property
    @abstractmethod
    def write_roi(self) -> Roi:
        pass

    @property
    @abstractmethod
    def write_size(self) -> Coordinate:
        pass

    @property
    @abstractmethod
    def context_size(self) -> Union[Coordinate, tuple[Coordinate, Coordinate]]:
        pass

    def init(self):
        # TODO: override this in subclasses if necessary
        pass

    @abstractmethod
    def process_block_func(self):
        pass

    @abstractmethod
    def drop_artifacts(self):
        pass

    @property
    def block_write_roi(self) -> Roi:
        return Roi((0,) * self.write_size.dims, self.write_size)

    @property
    def meta_dir(self) -> Path:
        return LOG_BASEDIR / f"{self.task_name}-meta"

    @property
    def config_file(self) -> Path:
        return self.meta_dir / "config.json"

    @property
    def block_ds(self) -> Path:
        return self.meta_dir / "blocks_done.zarr"

    def process_roi(self, roi: Roi, context: Optional[Coordinate] = None):
        block = daisy.Block(
            roi, roi if context is None else roi.grow(context, context), roi
        )
        process_block = self.process_block_func()
        process_block(block)

    def drop(self, drop_outputs: bool = False) -> None:
        # reset the blocks_done ds so that the task is rerun
        if self.meta_dir.exists():
            rmtree(self.meta_dir)
        self.drop_artifacts()

    def check_block_func(self):
        def check_block(block):
            block_array = open_ds(self.block_ds, mode="r")
            offset = block.write_roi.offset
            voxel_size = block_array.voxel_size

            block_roi = Roi(offset, voxel_size)

            block_data = block_array[block_roi]

            return block_data == block.block_id[1] + 1

        return check_block

    def mark_block_done_func(self):
        def write_check_block(block):
            block_array = open_ds(str(self.block_ds[0]), self.block_ds[1], mode="a")
            write_roi = block.write_roi.intersect(block_array.roi)
            block_array[write_roi] = np.full(
                write_roi.shape // block_array.voxel_size,
                fill_value=block.block_id[1] + 1,
            )

            block.status = daisy.BlockStatus.SUCCESS

        return write_check_block

    def worker_func(self):
        if self.worker_config:
            config_file = self.config_file

            with open(config_file, "w") as f:
                f.write(self.model_dump_json())

            logging.info("Running block with config %s..." % config_file)

            cmd = [
                "e11-post",
                "blockwise-worker",
                "-c",
                str(config_file),
            ]

            queue = self.worker_config.queue

            if queue is not None:
                # todo: figure out how to consolidate worker directories since
                # we don't have access to worker ids yet here...

                # context = daisy.Context()
                # worker_id = context["worker_id"]

                # log_base = daisy.logging.get_worker_log_basename(
                # worker_id, context.get("task_id", None)
                # )

                # log_file = f"{log_base}.slurm.out"
                # log_error = f"{log_base}.slurm.err"

                log_base = f"./daisy_logs/{self.task_name}"

                log_file = f"{log_base}/slurm_worker_%j.log"
                log_error = f"{log_base}/slurm_worker_%j.err"

                cmd = self.worker_config.get_slurm_command(
                    command=" ".join(cmd),
                    execute=False,
                    expand=False,
                    queue=queue,
                    num_gpus=self.worker_config.num_gpus,
                    num_cpus=self.worker_config.num_cpus,
                    log_file=log_file,
                    error_file=log_error,
                )

            return lambda: subprocess.run(cmd)

        else:
            return self.process_blocks

    def process_blocks(self):
        with self.process_block_func() as process_block:

            def worker_loop():
                client = daisy.Client()
                mark_block_done = self.mark_block_done_func()

                while True:
                    logger.info("getting block")
                    with client.acquire_block() as block:
                        logger.info(f"got block {block}")

                        if block is None:
                            break

                        process_block(block)
                        mark_block_done(block)

            if self.num_cache_workers > 1:
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
        # prepare blocks done ds

        def cmin(a, b):
            return Coordinate([min(ai, bi) for ai, bi in zip(a, b)])

        def cmax(a, b):
            return Coordinate([max(ai, bi) for ai, bi in zip(a, b)])

        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def cgcd(a, *bs):
            while len(bs) > 0:
                b = bs[0]
                bs = bs[1:]
                a, b = cmax(a, b), cmin(a, b)
                a = Coordinate([gcd(ai, bi) for ai, bi in zip(a, b)])
            return abs(a)

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

        block_voxel_size = cgcd(
            self.write_roi.offset, self.write_size, self.write_roi.shape
        )

        prepare_ds(
            self.block_ds,
            shape=self.write_roi.shape // block_voxel_size,
            offset=self.write_roi.offset,
            voxel_size=block_voxel_size,
            chunk_shape=block_voxel_size,
            dtype=get_dtype(self.write_roi, self.write_size),
            mode="w",
        )

    def task(
        self, upstream_tasks: Optional[Union[daisy.Task, list[daisy.Task]]] = None
    ) -> daisy.Task:
        # create task
        context = self.context_size
        if not isinstance(context, Coordinate):
            assert isinstance(context, tuple)
            context_low, context_high = context[0], context[1]
        else:
            context_low, context_high = context, context

        if self.num_workers is not None:
            process_func = self.worker_func()
            num_workers = self.num_workers
        else:
            process_func = self.process_block_func().__enter__()
            num_workers = 1 # dummy value

        task = daisy.Task(
            self.task_name,
            total_roi=self.write_roi.grow(context_low, context_high),
            read_roi=self.block_write_roi.grow(context_low, context_high),
            write_roi=self.block_write_roi,
            process_function=process_func,
            read_write_conflict=self.read_write_conflict,
            fit=self.fit,
            num_workers=num_workers,
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

        return task

    def run_blockwise(
        self,
        upstream_tasks: Optional[list[daisy.Task]] = None,
        multiprocessing: bool = True,
    ):
        self.init_block_array()
        self.init()
        task = self.task(upstream_tasks)
        if multiprocessing:
            daisy.run_blockwise(task)
        else:
            server = daisy.SerialServer()
            cl_monitor = daisy.cl_monitor.CLMonitor(server)  # noqa
            server.run_blockwise([task])
