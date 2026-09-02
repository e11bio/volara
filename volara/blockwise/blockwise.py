import logging
import multiprocessing
import subprocess
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING, Any, Iterator, cast

import daisy
import numpy as np
from funlib.geometry import Coordinate, Roi

from volara.logging import get_log_basedir, set_log_basedir

from ..datasets import MaskDataset
from ..utils import PydanticRoi, StrictBaseModel
from ..workers import Worker
from .benchmark import BenchmarkLogger

if TYPE_CHECKING:
    from .pipeline import Pipeline

logger = logging.getLogger(__name__)

#: Group attribute recording what a seeded tracking store was written for.
_SEED_ATTR = "volara_block_done_seed"

#: Memoised result of the seeded-store capability probe (None = not yet run).
_daisy_seeded_store_ok: bool | None = None


def _require_daisy_accepts_seeded_store() -> None:
    """Refuse ``block_done_mask`` on a daisy that rejects caller-seeded stores.

    Measured, not inferred from a version: the probe seeds a one-cell tracking
    store in exactly the format ``_seed_block_done_mask`` writes and runs a
    one-block no-op task against it -- the same call path a real seeded run
    takes. daisy before ``a160769`` refuses such a store at scheduler init
    (``LayoutMismatch``: no ``daisy_task_hash``); from ``a160769`` it is
    accepted and validated by shape. The result is memoised per process.
    """
    global _daisy_seeded_store_ok
    if _daisy_seeded_store_ok is None:
        import tempfile

        import zarr

        with tempfile.TemporaryDirectory(prefix="volara-seed-probe-") as td:
            tracking = Path(td) / "blocks_done"
            group = zarr.open_group(str(tracking), mode="a")
            for name in ("done", "seed"):
                arr = group.create_array(
                    name, shape=(1,), chunks=(1,), dtype="uint8",
                    fill_value=0, compressors=[], overwrite=True,
                )
                arr[:] = np.ones(1, dtype=np.uint8)
            group.attrs[_SEED_ATTR] = {"probe": True}
            probe = daisy.Task(
                "volara-block-done-mask-probe",
                total_roi=Roi((0,), (8,)),
                read_roi=Roi((0,), (8,)),
                write_roi=Roi((0,), (8,)),
                process_function=lambda block: None,
                fit="shrink",
                max_workers=1,
                tracking_path=str(tracking),
                max_retries=0,
            )
            try:
                # ⚠️ multiprocessing=False. The refusal happens at tracking
                # init, before any server, so both settings return the same
                # verdict -- but the default stands up the distributed
                # scheduler, a TCP server and a worker for a no-op block.
                # Measured: 0.917s -> 0.006s, and it stops the probe printing
                # an Execution Summary and creating daisy_logs/ in the caller's
                # cwd on every process that touches a mask.
                daisy.run_blockwise(
                    [probe], progress=False, multiprocessing=False
                )
                _daisy_seeded_store_ok = True
            except RuntimeError as e:
                text = str(e)
                if "task layout" in text or "daisy_task_hash" in text:
                    _daisy_seeded_store_ok = False
                else:
                    raise
    if not _daisy_seeded_store_ok:
        raise RuntimeError(
            "block_done_mask needs a daisy that accepts a caller-seeded "
            "(hash-less) tracking store: funkelab/daisy a160769 "
            "('block_tracking: accept a caller-provided tracking store', "
            "v2.0 branch, 2026-08-31) or later. The installed daisy refused "
            "the probe store at scheduler init -- bump the daisy pin and "
            "relock before using block_done_mask."
        )


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
    block_timeout: float | None = None
    """
    Per-block timeout in seconds. ``None`` (the default) delegates to daisy's
    default (600s). daisy v2 always enforces a timeout -- there is no "unlimited"
    option -- so blocks should be sized to process well within it; a task whose
    single block legitimately runs long (e.g. the global mutex watershed) must set
    an explicit large value instead (see ``GraphMWS``). When a block exceeds the
    timeout daisy reclaims it and retries under ``max_retries``.
    """

    block_done_mask: MaskDataset | None = None
    """
    An optional per-block SKIP mask (a :class:`~volara.datasets.MaskDataset`): a
    uint8 zarr array over this task's block grid where 1 = skip the block
    (pre-mark it done) and 0 = run it. Before the run, volara seeds it into
    daisy's tracking store, so masked blocks are skipped without ever being
    scheduled; daisy still tracks the task and reports the skips in its summary.

    Grid contract: the mask's shape must equal :attr:`block_grid_shape`
    (``ceil(total_roi.shape / block)`` over the context-grown total ROI), and
    cell ``i`` in dim ``d`` covers the write window starting at
    ``block_grid_cell_origin()[d] + i * block[d]``. ⚠️ That origin is NOT
    ``write_roi.offset``: daisy indexes a block against the CONTEXT-GROWN total
    ROI, so the block written at ``write_roi.offset`` is cell
    ``context_low // block``. Use :meth:`block_grid_slices` to map a world ROI to
    cells rather than deriving the index yourself.

    Resume contract: seeded skip bits are recorded separately from real
    completions, so re-running with a different mask un-skips blocks the new
    mask no longer covers while keeping every block a prior run actually
    computed. A task whose geometry changed refuses to reuse the store.

    Requires a daisy that accepts a caller-seeded tracking store
    (funkelab/daisy ``a160769``, v2.0 branch, 2026-08-31); on an older daisy the
    seed refuses up front, naming that commit.
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
    def tracking_path(self) -> "Path":
        """Where daisy persists its per-block tracking for this task.

        Kept under ``meta_dir`` so ``drop()`` resets it with the logs; ``task()``
        hands exactly this path to ``daisy.Task(tracking_path=...)``.
        """
        return self.meta_dir / "blocks_done"

    def _context_pair(self) -> tuple[Coordinate, Coordinate]:
        """``(context_low, context_high)`` as ``task()`` resolves them."""
        context = self.context_size
        if not isinstance(context, Coordinate):
            assert isinstance(context, tuple)
            return context[0], context[1]
        return context, context

    @property
    def block_grid_shape(self) -> tuple[int, ...]:
        """Shape of daisy's per-block tracking grid for this task.

        ``ceil(total_roi.shape / block)`` where ``total_roi`` is the
        context-grown write ROI and ``block`` is ``block_write_roi.shape`` --
        the same numbers ``task()`` hands to ``daisy.Task``.
        """
        lo, hi = self._context_pair()
        total = self.write_roi.grow(lo, hi)
        block = self.block_write_roi.shape
        return tuple(-(-int(t) // int(b)) for t, b in zip(total.shape, block))

    def block_grid_cell_origin(self) -> Coordinate:
        """World coordinate that grid cell ``0`` starts at, per dim.

        ⚠️ NOT ``write_roi.offset``. daisy indexes a block as
        ``(block.write_roi.offset - total_roi.offset) // write_shape``
        (``block_tracking.rs::grid_coord``), and volara hands it the
        CONTEXT-GROWN write ROI as ``total_roi``. So the block whose write
        window starts at ``write_roi.offset`` is cell ``context_low // block``,
        not cell 0, and the grid origin sits ``(context_low // block) * block``
        below the write offset -- floor-divided, because daisy's index is an
        integer division and every block shifts by the same whole number of
        cells.

        Measured against daisy's own ``done`` array, write ROI ``(0,0)+(40,10)``
        with ``block = (10, 10)`` (4 blocks at write offsets 0/10/20/30):

        ==============  ======================  ===========================
        ``context_low``  daisy ``done``          cell of the block at 0
        ==============  ======================  ===========================
        0               ``[1,1,1,1]``           0
        9               ``[1,1,1,1,0,0]``       0
        10              ``[0,1,1,1,1,0]``       1
        25              ``[0,0,1,1,1,1,0,0,0]`` 2
        ==============  ======================  ===========================
        """
        lo, _ = self._context_pair()
        block = self.block_write_roi.shape
        return Coordinate(
            int(o) - (int(l) // int(b)) * int(b)
            for o, l, b in zip(self.write_roi.offset, lo, block)
        )

    def block_grid_slices(self, roi: Roi) -> tuple[slice, ...]:
        """Grid cells whose WRITE window intersects ``roi``, as per-dim slices.

        Cell ``i`` in dim ``d`` covers the write window starting at
        ``block_grid_cell_origin()[d] + i * block[d]`` -- see there for why that
        is not ``write_roi.offset``. Slices are clipped to
        :attr:`block_grid_shape`.
        """
        block = self.block_write_roi.shape
        anchor = self.block_grid_cell_origin()
        grid = self.block_grid_shape
        out = []
        for d, (a, b, g) in enumerate(zip(anchor, block, grid)):
            begin = int(roi.begin[d]) - int(a)
            end = int(roi.end[d]) - int(a)
            lo = max(begin // int(b), 0)
            hi = min(-(-end // int(b)), g)
            out.append(slice(lo, max(hi, lo)))
        return tuple(out)

    def process_roi(self, roi: Roi, context: Coordinate | None = None):
        """
        A helper function to process a given roi without needing to start a
        whole blockwise job.
        """
        read_roi = roi if context is None else roi.grow(context, context)
        block = daisy.Block(roi, read_roi, roi)
        process_block = self.process_block_func()
        process_block(block)

    def drop(self, drop_outputs: bool = True) -> None:
        """
        A helper function to drop any artifacts produced by a task
        and return to a state identical to before having executed the
        task.

        Args:
            drop_outputs: whether to also delete this task's outputs via
                :meth:`drop_artifacts`. ``True`` (the default) is the full reset
                described above, and is what ``drop()`` has always done. Pass
                ``False`` to clear only the meta directory -- the worker logs and
                the block-done cache -- which forces every block to be recomputed
                on the next run while leaving the existing outputs (datasets, dbs,
                luts) in place to be overwritten block by block.
        """
        # reset the blocks_done ds so that the task is rerun
        if self.meta_dir.exists():
            rmtree(self.meta_dir)
        if drop_outputs:
            self.drop_artifacts()

    def worker_func(self):
        """
        The function defining how workers are started.
        """
        worker_config = self.worker_config
        if worker_config is not None:
            config_file = self.config_file

            with open(config_file, "w") as f:
                f.write(self.model_dump_json())

            logging.info("Running block with config %s..." % config_file)

            def run_worker():
                cmd = worker_config.get_command(config_file, self.task_name)
                # daisy v2 runs this spawn function in a THREAD and expects it to
                # BLOCK for the worker's lifetime -- a spawn fn that returns early
                # is treated as a dead worker and respawned (up to
                # max_worker_restarts). get_command therefore builds a blocking
                # submission on every backend (local child process, sbatch --wait,
                # bsub -K); a fire-and-forget submit (bare sbatch/bsub) would trip
                # the v2 respawn loop and leak the real cluster job.
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
            self.meta_dir.mkdir(parents=True, exist_ok=True)
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

                def process_func(block):
                    with benchmark_logger.trace("Process Block"):
                        process_block(block)

            self._seed_block_done_mask()

            task = daisy.Task(
                self.task_name,
                total_roi=self.write_roi.grow(context_low, context_high),
                read_roi=self.block_write_roi.grow(context_low, context_high),
                write_roi=self.block_write_roi,
                process_function=process_func,
                read_write_conflict=self.read_write_conflict,
                fit=self.fit,
                max_workers=self.num_workers,
                tracking_path=str(self.tracking_path),
                max_retries=2,
                timeout=self.block_timeout,
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

    def _seed_layout(self) -> dict:
        """The task geometry a seeded tracking store is only valid for."""
        lo, hi = self._context_pair()
        total = self.write_roi.grow(lo, hi)
        return {
            "total_roi": [list(total.offset), list(total.shape)],
            "read_roi": [
                list(self.block_write_roi.grow(lo, hi).offset),
                list(self.block_write_roi.grow(lo, hi).shape),
            ],
            "write_roi": [
                list(self.block_write_roi.offset),
                list(self.block_write_roi.shape),
            ],
            "fit": str(self.fit),
            "grid_shape": list(self.block_grid_shape),
        }

    def _seed_block_done_mask(self) -> None:
        """Reconcile ``block_done_mask`` with daisy's tracking store before the run.

        ⚠️ Runs on EVERY task, including ``block_done_mask=None``. Seeding only
        when a mask is present leaves the previous run's skip bits standing in
        ``done``, and daisy honours them: the narrowest mask of all -- removing
        the field -- would otherwise be the one case that never un-skips.
        Measured before this was unconditional: a mask-less rerun through a
        seeded meta dir logged ``resumed -- 1/4 blocks skipped via done markers``
        and left the masked region unwritten with ``failed=0``. ``None`` is
        therefore an all-zero mask, and a store carrying no seed record is left
        untouched (so an unmasked task pays neither the probe nor a write).

        Three bits of state keep "seeded" and "completed" apart:

        ``done``
            what daisy reads. Written ``real | skip``.
        ``seed``
            the bits THIS volara put in ``done`` that no worker earned, i.e.
            ``skip & ~real``. Real completions are never recorded here, so a
            widen-then-narrow cycle cannot demote one into a re-run.
        ``_SEED_ATTR``
            the layout the store was seeded for, plus the mask digest. A store
            volara seeded never acquires daisy's own ``daisy_task_hash`` (volara
            creates the group first, and daisy only stamps a group it creates),
            so this attr is the ONLY layout identity such a store has -- which is
            why it must be checked on every run, not only masked ones.

        ⛔ The write order is a sandwich, and it is not decoration. The resume
        invariant is ``done & ~seed == real``; every intermediate state must keep
        ``seed`` a SUPERSET of ``done & ~real``, so an interruption can only cost
        a recompute. Narrowing ``seed`` before ``done`` drops that -- measured: a
        run masking cell 0, then a mask-less rerun killed in the window, leaves
        ``done=[1,1,1,1] seed=[0,0,0,0]`` and cell 0 is a permanent completion no
        worker ever ran, with ``failed=0``. Narrowing is exactly what this
        feature is for (``block_done_mask=None`` is the narrowest mask of all),
        so the widening write goes first, ``done`` in the middle, and the
        tightening write last. It does not take a SIGKILL: an ENOSPC on the
        ``done`` write leaves the same state, and the operator fixes the disk and
        reruns into a silent skip.
        """
        import hashlib

        import zarr

        tracking = self.tracking_path
        seeded = (tracking / "done" / "zarr.json").exists()
        if seeded:
            group = zarr.open_group(str(tracking), mode="r+")
            prior = cast("dict | None", dict(group.attrs).get(_SEED_ATTR))
        else:
            group, prior = None, None
        if self.block_done_mask is None and prior is None:
            # Nothing volara seeded and nothing to seed: daisy owns this store.
            return

        _require_daisy_accepts_seeded_store()
        expected = self.block_grid_shape
        if self.block_done_mask is None:
            skip = np.zeros(expected, dtype=np.uint8)
        else:
            skip = (self.block_done_mask.read_mask() != 0).astype(np.uint8)
            if tuple(skip.shape) != tuple(expected):
                raise ValueError(
                    f"block_done_mask shape {tuple(skip.shape)} does not match task "
                    f"{self.task_name!r}'s block grid {tuple(expected)} "
                    "(= ceil(context-grown total / block); cell i covers the write "
                    "window at block_grid_cell_origin() + i*block -- see "
                    "block_grid_slices())."
                )
        layout = self._seed_layout()

        if group is not None:
            # Layout BEFORE shape: a changed geometry usually changes the grid
            # shape too, and "your mask is the wrong shape" is the wrong story
            # then -- doubly so on the mask-less path, where there is no mask to
            # blame.
            #
            # ⚠️ Only for a store DAISY DOES NOT OWN. The stamp exists because a
            # volara-seeded group never gets daisy's `daisy_task_hash` and is
            # therefore accepted for any layout; where the hash IS present daisy
            # validates the geometry itself and is the authority. Honouring our
            # stamp there bricks the store: a masked run at the wrong geometry
            # writes the stamp, daisy then refuses and nothing runs, and every
            # later run of the ORIGINAL task refuses on our stale stamp -- with
            # "drop the meta dir" as the only way out, discarding real
            # completions after a mistake that computed nothing.
            daisy_owned = "daisy_task_hash" in dict(group.attrs)
            if prior is not None and not daisy_owned and prior.get("layout") != layout:
                raise RuntimeError(
                    f"task {self.task_name!r}: the seeded tracking store at "
                    f"{tracking} was written for a different task geometry; "
                    f"drop the meta dir ({self.meta_dir}) to reset tracking "
                    "rather than reusing done bits from another layout."
                )
            cur = self._read_grid_array(group, "done")
            if cur.shape != skip.shape:
                raise ValueError(
                    f"task {self.task_name!r}: the existing done array has shape "
                    f"{cur.shape} but this task's block grid is "
                    f"{tuple(skip.shape)}; drop the meta dir to reset tracking."
                )
            if prior is not None:
                prev_seed = self._read_grid_array(group, "seed")
                real = cur & ~prev_seed
            else:
                # daisy wrote this store; every bit in it was earned.
                real = cur
        else:
            group = zarr.open_group(str(tracking), mode="a")
            real = np.zeros_like(skip)

        record = {
            "layout": layout,
            "mask_sha256": hashlib.sha256(skip.tobytes()).hexdigest(),
        }
        seeded_now = skip & ~real
        prev = (
            self._read_grid_array(group, "seed")
            if "seed" in group
            else np.zeros_like(skip)
        )
        # WIDEN -> stamp -> done -> TIGHTEN. Each prefix leaves `seed` covering
        # every mask-only bit in `done`; see the docstring for what the other
        # order costs. The stamp precedes `done` for the same reason: without it
        # a fresh store's seeded bits read back as real.
        self._write_grid_array(group, "seed", prev | seeded_now)
        group.attrs[_SEED_ATTR] = record
        self._write_grid_array(group, "done", real | skip)
        self._write_grid_array(group, "seed", seeded_now)

    @staticmethod
    def _read_grid_array(group, name: str) -> "np.ndarray":
        """``group[name]`` as a uint8 array. Narrowing lives here because
        ``Group.__getitem__`` is typed ``AnyArray | Group``, so subscripting the
        result is unchecked at every call site otherwise."""
        arr = cast("Any", group[name])
        return np.asarray(arr[:], dtype=np.uint8)

    @staticmethod
    def _write_grid_array(group, name: str, values: "np.ndarray") -> None:
        """Write ``values`` to ``group[name]``, creating it in the raw
        single-chunk uint8 layout daisy reads ``done`` as."""
        if name in group:
            cast("Any", group[name])[:] = values
            return
        arr = group.create_array(
            name, shape=values.shape, chunks=values.shape, dtype="uint8",
            fill_value=0, compressors=[], overwrite=True,
        )
        arr[:] = values

    def get_benchmark_logger(self) -> BenchmarkLogger:
        _benchmark_db_path = Path("volara_benchmark_logs/benchmark.db")
        return BenchmarkLogger(
            None,
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
                    # daisy v2 Server.run_blockwise returns the {task_id: TaskState} map.
                    result = daisy.Server().run_blockwise(tasks)
                else:
                    result = daisy.run_blockwise(
                        tasks, multiprocessing=False, return_states=True
                    )

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

        Returns daisy's ``{task_id: TaskState}`` map for BOTH the multiprocessing and serial
        paths. Previously the multiprocessing path went through ``daisy.run_blockwise``, which
        collapses the states to a bool and, worse, reports ``True`` even when blocks failed
        (``TaskState.is_done()`` counts failed/orphaned blocks as "done"). Returning the states
        lets callers inspect ``failed_count`` / ``orphaned_count`` / ``is_done()`` and react to
        an incomplete run instead of silently accepting a partial output.
        """
        with self.task(multiprocessing=multiprocessing) as task:
            tasks = [task]
            if multiprocessing:
                # daisy v2's Server.run_blockwise returns the {task_id: TaskState}
                # map natively, so the old 1.x ThreadPool/IOLooper/progress-monitor
                # states workaround is no longer needed.
                return daisy.Server().run_blockwise(tasks)
            # Serial path: module-level run_blockwise returns a bool unless
            # return_states=True (see daisy._runner), so request the states map to
            # keep the same {task_id: TaskState} contract as the distributed path.
            return daisy.run_blockwise(tasks, multiprocessing=False, return_states=True)

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
