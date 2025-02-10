# Volara

E11 Bio is excited to release its first software package, Volara - An open source python library that faciliates the application of common blockwise operations for image processing of arbitrarily large volumetric microscopy datasets.

When working with large n-dimensional datasets, efficient and scalable processing is necessary. Complex image processing pipelines have generally been challenging to use by non-experts, thus limiting the accessibility of cutting edge methods. To address these issues, we developed Volara, a Python library designed to make blockwise processing of massive volumes easy, robust and repeatable.

Volara was inititally developed for the task of neuron segmentation in connectomics datasets, and therefore contains the necessary logic to extract segmentations from affinity graphs. However, at its core, Volara aims to create block-wise task abstractions and is thus easily extendable for other image processing pipelines. All the block management, parallelization, and resulting aggregation complexity of dividing tasks into manageable blocks is transparently handled. Volara also provides many nice-to-have features such as detailed progress bars, organized logging, and visualization of completed blocks.

### Key Features

#### 1. **Common Tasks**

Volara comes with built in support for common operations:

- **Supports next gen file formats**: Volara supports [zarr](https://zarr.readthedocs.io/en/stable/) and [ome-zarr](https://github.com/ome/ome-zarr-py) style datasets
- **Supports lazy operations** - By using [dask](https://www.dask.org/), Volara can perform many  operations such as thresholding, normalizing, slicing and dtype conversions on-the-fly
- **Model Prediction**: Run [PyTorch](https://pytorch.org/) machine learning models on arbitrarily large volumes.

#### 2. **Microscopy-Specific Operations**

Volara comes equipped with a suite of operations tailored to machine learning and computational tasks for microscopy:

- **Affinity Processing**:
  - Supervoxel extraction using [Mutex Watershed](https://arxiv.org/abs/1904.12654).
  - Compute within and across block aggregated edge costs between supervoxels.
  - Global graph optimization using Mutex Watershed.
  - Relabel fragments into segments based on a globally optimized lookup table.
- **Other Tasks**:
  - Perform local registration of a moving image to a fixed image
  - Take an argmax over multi-channel volumes.

#### 3. **Flexible Graph Support**

Volara supports graph storage in databases like [SQLite](https://www.sqlite.org/) for quick and simple setups, or [PostgreSQL](https://www.postgresql.org/) for better performance in larger-scale projects.

#### 4. **Parallelized Blockwise Processing**

Volara uses [daisy](https://github.com/funkelab/daisy) under the hood for efficient block-wise processing and task scheduling. It has support for running jobs both locally and on a cluster (e.g [SLURM](https://slurm.schedmd.com/documentation.html)). This is handled by a simple configurable worker config which Volara then uses  to distribute the workload while ensuring efficient resource utilization.

#### 5. **Progress Tracking and Visualization**

Volara has a built-in progress bar which provides an estimated time to completion and detailed information about any failed blocks. Additionally, Volara tracks which blocks have been processed and allows for easy visualization of block progress overlaid on the volumes being processed.

#### 6. **Robustness to Failure**

During processing of large volumes, it is common for blocks to fail for various reasons. If a specific block fails, it can be retried until before being marked as failed. If a specific worker dies, it can be restarted before the task is considered failed. If a job fails or is interrupted, on its next execution, Volara will quickly skip all previously completed blocks to continue finishing the volume. Throughout this entire process Volara maintains robust logs allowing for easier debugging of errors anywhere in the pipeline.

#### 7. **Chainable tasks**

Inspired by [Daisy](https://github.com/funkelab/daisy) and [Luigi](https://luigi.readthedocs.io/en/stable/central_scheduler.html), Volara provides flexibility for chaining tasks together by representing each task as a node in a directed acyclic graph (DAG). When the final downstream task is called it will request any upstream task which cascades up to the starting task of the DAG. This task then begins processing, and when there is enough block context available the next task will start, until all tasks have been completed. This offers an extra layer of parallelization and is useful for long running tasks which require the output of previous tasks.

#### 8. **Plugin System for Custom Tasks**

Volara has a built in plugin system that makes it easy to 
define custom blockwise tasks. With little overhead, a custom task can leverage all of Volara's features, including cluster job processing, progress tracking, task scheduling, and visualization.
