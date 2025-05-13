# %% [markdown]
# # Building a custom task
# This notebook provides a behind the scenes look at the Volara framework, how it works, and how you
# can build upon it to process your own data in whatever way you would like. This is also a good
# place to start if you are trying to debug a job going wrong.

# %% [markdown]
# ## Daisy
# `volara` is essentially a convenience wrapper around `daisy`, a python library for blockwise
# processing. `daisy` is a powerful library that provides the basics of blockwise processing.
# [See here](https://funkelab.github.io/daisy/) for details about `daisy`. An understanding of
# `daisy` is not required to use `volara`, but will make unstanding and developing using `volara`
# easier.
#
# `volara` provides:
# - A simple interface for defining blockwise operations, that then generates the necessary daisy code
# - A set of common operations that are often used in volumetric data processing
# - Free nice to have features for any task using the `volara` framework:
#   - Completed block tracking that is both fast, and easily visualized
#   - Support for running workers on remote machines, making it easy to utilize
#     slurm, lsf, or any other cluster type. As long as you can create a tcp connection
#     to the machine you want to run the worker on, it should be possible to support any `volara` task.
#
# Other tutorials go into more detail about how to use the existing operations, but this tutorial
# will focus on how to use `volara` for your own custom operations.

# %% [markdown]
# ## BlockwiseTask
# The `BlockwiseTask` class is the main entry point for defining a blockwise operation.
# It is a pydantic class and an ABC that defines the minimum requirements for a blockwise task.
#
# See [the documentation](https://e11bio.github.io/volara/api.html#volara.blockwise.BlockwiseTask) for details
# about the BlockwiseTask class.
#
# If you subclass `BlockwiseTask`, you must provide at minimum the following fields, abstract methods and properties:
#
# ### fields:
# - task_type: A string that identifies the type of task. This is used to deserialize a yaml file
#   into a task. This is only necessary to override if you want to run a worker on a separate process
#   or machine. If you are just running locally, you can ignore this.
# - fit: See [daisy docs](https://funkelab.github.io/daisy/api.html#daisy.Task) for info on the "fit"
#   field.
# - read_write_conflict: See [daisy docs](https://funkelab.github.io/daisy/api.html#daisy.Task) for info
#   on the "read_write_conflict" field.
#
# ### properties:
# - task_name: A string that uniquely identifies specific instances of your task. This is used
#   in the file path to write logs, keep track of progress, and for communication between the
#   client/server model in `daisy`.
# - write_roi: The total output ROI (Region Of Interest) of the task. This is often just ROI of your
#   input array, but can be different for some tasks. Note that this is expected in *World Units*.
# - write_size: The write size of each block processed as part of a task. Note that this is expected
#   in *World Units*.
# - context_size: The amount of context needed to process each block for a task. It can be provided
#   as a single tuple of context that is added above and below every block, or as a pair of lower and
#   upper context.
#
# ### methods:
# - drop_artifacts: A helper function to reset anything produced by a task to a clean state equivalent
#   to not having run the task at all
# - process_block_func: A constructor for a function that will take a single block as input and process it.
#   Note that this constructor should be implemented as a context manager, that yields a `process_block` function:
#   ```python
#   @contextmanager
#   def process_block_func(self, block: Block, ...) -> Callable[[Block], None]:
#       # do any setup that is needed for the worker
#       def process_block(block: Block, ...) -> None:
#           # do something with the block
#       yield process_block
#       # do any cleanup that is needed for the worker
#   ```

# %%
import multiprocessing as mp

mp.set_start_method("fork", force=True)

import dask

dask.config.set(scheduler="single-threaded")

# %% [markdown]
# ## Example: Argmax
# Lets build the simplest possible argmax task using volara.

# %%
# `BlockwiseTask` base class is necessary to use the `volara` framework
import logging

# `shutil` is used to remove the artifacts of a task
import shutil

# contextmanager decorator for the process_block_func method
from contextlib import contextmanager

# `numpy` for generating data
import numpy as np

# `daisy` splits the task into blocks for processing and passes the blocks to the workers
# Note that blocks only contain read_roi, write_roi, and a unique identifier
from daisy import Block

# `Coordinate` and `Roi` are used to define points and regions in 3D space
from funlib.geometry import Coordinate, Roi

# `prepare_ds` and `open_ds` are helper methods to interface with zarr arrays
# with offsets, voxel_sizes, units, and axis types such as "channel", "time", and "space"
from funlib.persistence import open_ds, prepare_ds

from volara.blockwise import BlockwiseTask

logging.basicConfig(level=logging.INFO)


class Argmax(BlockwiseTask):
    """
    A super simple argmax task
    """

    # task_type is used to identify the task type. This is only needed if you are
    # running the task on a remote machine.
    task_type: str = "argmax"

    # simple task settings
    fit: str = "shrink"
    read_write_conflict: bool = False

    # There are no inputs, so this is just a constant string
    @property
    def task_name(self) -> str:
        return "simple-argmax"

    # We will make a 10x10x10 array with 3 channels. The channels are not included
    # Roi since they are not spatially relevant
    @property
    def write_roi(self) -> Roi:
        return Roi((0, 0, 0), (10, 10, 10))

    # We will write chunks of size 5x5x5 at a time. So we will have 8 blocks
    @property
    def write_size(self) -> Coordinate:
        return Coordinate((5, 5, 5))

    # No context is needed for argmax
    @property
    def context_size(self) -> Coordinate:
        return Coordinate((0, 0, 0))

    # We will initialize some input data, and create the output array.
    # Most tasks will need an init to define the output of a task. The inputs
    # will usually be passed in as a parameter to the task.
    def init(self):
        in_array = prepare_ds(
            f"{self.task_name}/data.zarr/in_array",
            shape=(3, 10, 10, 10),
            chunk_shape=(3, 5, 5, 5),
            offset=(0, 0, 0),
        )
        np.random.seed(0)
        in_array[:] = np.random.randint(0, 10, size=in_array.shape)

        prepare_ds(
            f"{self.task_name}/data.zarr/out_array",
            shape=(10, 10, 10),
            chunk_shape=(5, 5, 5),
            offset=(0, 0, 0),
        )

    # make sure that both the input and output arrays are removed if this task
    # is dropped.
    def drop_artifacts(self):
        shutil.rmtree(f"{self.task_name}/data.zarr/in_array")
        shutil.rmtree(f"{self.task_name}/data.zarr/out_array")

    # Input and output arrays are opened in the context manager, the process_block
    # function only needs to read and write to those arrays.
    @contextmanager
    def process_block_func(self):
        in_array = open_ds(
            f"{self.task_name}/data.zarr/in_array",
            mode="r+",
        )
        out_array = open_ds(
            f"{self.task_name}/data.zarr/out_array",
            mode="r+",
        )

        def process_block(block: Block) -> None:
            in_data = in_array[block.read_roi]
            out_data = in_data.argmax(axis=0)
            out_array[block.write_roi] = out_data

        yield process_block


# %% [markdown]
# ### Running the task

# %%
argmax_task = Argmax()
argmax_task.run_blockwise(multiprocessing=False)

# %% [markdown]
# ### Inspecting the results
# %%
import zarr

print(zarr.open(f"{argmax_task.task_name}/data.zarr/in_array")[:, :, 0, 0])
print(zarr.open(f"{argmax_task.task_name}/data.zarr/out_array")[:, 0, 0])

# %% [markdown]
# ### What do we get for free?

# %% [markdown]
# #### Block done tracking
# %%
# Rerunning the same task. All blocks get skipped
argmax_task.run_blockwise(multiprocessing=False)

# %% [markdown]
# #### drop task

# %%
# Call `argmax_task.drop()` to reset the task. This calls `drop_artifacts`
# but also removes any logs and block completion tracking.
argmax_task.drop()

# %% [markdown]
# #### Multiprocessing

# %%
# We can run the same job with mulitple workers.
argmax_task = Argmax(num_workers=2)
argmax_task.run_blockwise(multiprocessing=True)

# %% [markdown]
# #### Running on a remote machine
#
# This task is not quite ready to be run on a remote machine, but it is very close.
# To run on a remote machine, you need to register the task with `volara` so that we can
# deserialize the config files that are passed to the worker, and execute the correct code.
# This has to be done automatically based on the environment, so you need put your task in a
# pip installable python package. The basic structure of the package is:
# ```
# package-root/
# ├── volara-argmax-plugin/
# │   ├── __init__.py
# │   └── argmax.py
# └── pyproject.toml
# ```
# The `pyproject.toml` must include the following lines:
# ```toml
# [project.entry-points."volara.blockwise_tasks"]
# argmax = "volara_argmax_plugin.argmax:Argmax"
# ```
# This will register the task with `volara` so that it can be deserialized and run on a remote machine.
