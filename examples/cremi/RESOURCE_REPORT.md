# CREMI mutex-watershed example: resource usage

Measured with volara `main` (daisy `>=2.0.1`) and daisy 2.0.1 from PyPI, Python 3.12, on 2026-10-27. Every volara `BlockwiseTask` has `resource_tracking=True` by default, so daisy measures every block. For each block it records:
- wall time;
- CPU time of the worker process, all threads included;
- peak RSS of the worker while the block ran;
- IO bytes through the worker's syscalls.

It also times every worker's startup, idle time and teardown.

## How to generate this report

1. Run the example. The CREMI sample (`sample_A+_20160601.zarr`) ships with the repository.

   ```sh
   git clone https://github.com/e11bio/volara.git
   cd volara/examples/cremi
   uv run --python 3.12 --extra docs python cremi.py
   ```

   After the pipeline finishes, daisy prints an Execution Summary and its Resource Utilization panel. It also appends that run's report to `runs.jsonl` in each task's tracking store, under `volara_logs/` in the working directory. (`--python 3.12` because numba does not support 3.14 yet; `--extra docs` provides matplotlib for the example's GIFs.)

2. Build the report from those records with [`cremi_report.py`](cremi_report.py), which sits next to `cremi.py`:

   ```sh
   uv run --python 3.12 --extra docs python cremi_report.py   # reads ./volara_logs
   ```

   It reads the last line of each task's `runs.jsonl` with `daisy.TaskReport.from_dict`. It prints the per-task table below, then daisy's panel again via `daisy.RunReport(...).render()`. Inside a script you can also call `task.report()` on any `daisy.Task`.

3. To measure again, remove `volara_logs/` and the outputs (`fragments`, `segments`, `fragments.db`, `lut.npz` in the zarr), or call `pipeline.drop()`. Otherwise every block is skipped as already done, and the new report counts the blocks as `skipped`.

## The run

| | |
|---|---|
| Input | raw 125×1250×1250 uint8, affinities 7 channels (1.37 GB uncompressed, 776 MB on disk) |
| Blocks | 16×313×313 voxels (the raw chunk shape), 128 per blockwise task |
| Workers | 4 per blockwise task, 1 for graph-mws; all four tasks scheduled in one daisy pass |
| Outcome | 385 / 385 blocks completed; 0 failed, 0 orphaned, every worker exited cleanly |
| Pipeline wall time | 214 s |

**Machine:** a 96-core workstation that two unrelated training jobs shared during this run (load average ≈ 33). Wall times and block times are inflated by that contention. CPU-seconds, memory and IO depend on the work, not the load, so they are representative. Rerun on an idle machine for clean timings.

## Per task

| task | blocks | workers | block time | CPU | CPU / block time | mean block | peak RSS (max block) | IO read | IO write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| extract-frags | 128 | 4 | 782 s | 947 s | 1.21 | 6.11 s | 3.25 GB | 4.9 GB | 41 MB |
| aff-agglom | 128 | 4 | 218 s | 239 s | 1.09 | 1.71 s | 2.16 GB | 4.5 GB | 23 MB |
| graph-mws | 1 | 1 | 3.5 s | 3.5 s | 1.00 | 3.48 s | 0.55 GB | 17 MB | 11 MB |
| relabel | 128 | 4 | 9.5 s | 132 s | **13.9** | 0.074 s | 0.47 GB | 22 MB | 18 MB |
| **total** | 385 | 13 | 1013 s | 1321 s | 1.30 | | | 9.5 GB | 92 MB |

Workers took 1.9–3.4 s to start (interpreter, imports, task setup) and 0.4–0.7 s to shut down. Scheduling overhead per block (round trip minus the block's own wall time) was under 1.5 ms. Block times stayed level through the run.

## What it shows

- **extract-frags sets the pipeline's length.** It accounts for 72% of the CPU and 77% of block time. Its four workers each computed for about 195 s of the 214 s run, idling only about 1 ms between blocks.
- **aff-agglom waits on extract-frags.** Each of its workers sat idle for about 104 s, roughly half the run, at about 3.2 s per block. It can only take a block once the extract-frags blocks around it are done. Its own blocks are cheap (1.7 s).
- **relabel burns about 14 CPU-seconds per second of block work.** Its kernel, `replace_values_sorted` in `volara/segment_utils.py`, is `@numba.njit(parallel=True)`. With no limit set, numba starts one thread per core, so on 96 cores each of the 4 workers runs a 96-thread pool, and those threads spend most of the block spinning. Timing the kernel alone on one relabel block:

  | numba threads | wall | CPU |
  |---:|---:|---:|
  | 96 (default) | 6.5 ms | 602 ms |
  | 4 | 3.2 ms | 27 ms |
  | 1 | 12.2 ms | 13 ms |

  Capping numba's threads per worker would remove this waste, for example with `numba.set_num_threads(max(1, os.cpu_count() // num_workers))` at worker start. extract-frags and aff-agglom (1.1–1.2 CPU per block-second) probably pay a smaller version of the same cost.
- **Memory per worker is modest:** 3.25 GB at most for extract-frags, 2.2 GB for aff-agglom, and about 0.5 GB for graph-mws and relabel. With 4 workers per task, the blockwise steps need roughly 13 GB at their busiest. The driver process is separate: the example loads the whole volume to draw its input and output GIFs.
- **Reads are about 12× the store size:** 9.5 GB read from a 776 MB store. Each block reads its write region plus context, and that pulls whole 7×32×160×160 affinity chunks that neighbouring blocks read again. The count is bytes through `read()` calls, page-cache hits included, not disk traffic.
- **The machine is mostly unused.** Over the whole run, block time (1013 s) divided by wall time (214 s) averages fewer than 5 blocks in flight on 96 cores. Raising `num_workers` on extract-frags and aff-agglom is the obvious speed-up, once numba's threads are capped so that more workers don't oversubscribe the cores. The example keeps 4 workers so that it fits a CI runner.
