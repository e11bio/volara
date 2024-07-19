from contextlib import contextmanager
from typing import Optional, Union

import daisy
import pytest
from funlib.geometry import Coordinate, Roi

from volara.blockwise import BlockwiseTask
from volara.logging import set_log_basedir


import random

import matplotlib.pyplot as plt
import numpy as np
from funlib.geometry import Coordinate
from funlib.persistence import prepare_ds, Array
from scipy.ndimage.measurements import label
from skimage import data
from skimage.filters import gaussian

from volara.tmp import seg_to_affgraph


@pytest.fixture()
def cell_array(tmp_path) -> Array:
    cell_data = (data.cells3d().transpose((1, 0, 2, 3)) / 256).astype(np.uint8)

    # Handle metadata
    # This is a single sample image, so we don't need an
    # offset to line it up with other samples.
    offset = Coordinate(0, 0, 0)
    # voxel size in nanometers
    voxel_size = Coordinate(290, 260, 260)
    # By convention we add a '^' to non spatial axes
    axis_names = ["c^", "z", "y", "x"]
    # units for each spatial axis
    units = ["nm", "nm", "nm"]

    # Creates the zarr array with appropriate metadata
    cell_array = prepare_ds(
        tmp_path / "cells3d.zarr/raw",
        cell_data.shape,
        offset=offset,
        voxel_size=voxel_size,
        axis_names=axis_names,
        units=units,
        mode="w",
        dtype=np.uint8,
    )

    # Saves the cell data to the zarr array
    cell_array[:] = cell_data
    return cell_array


@pytest.fixture()
def mask_array(tmp_path, cell_array: Array) -> Array:
    # generate and save some psuedo gt data
    mask_array = prepare_ds(
        tmp_path / "cells3d.zarr/mask",
        cell_array.shape[1:],
        offset=cell_array.offset,
        voxel_size=cell_array.voxel_size,
        axis_names=cell_array.axis_names[1:],
        units=cell_array.units,
        mode="w",
        dtype=np.uint8,
    )
    cell_mask = np.clip(gaussian(cell_array[1] / 255.0, sigma=1), 0, 255) * 255 > 30
    not_membrane_mask = (
        np.clip(gaussian(cell_array[0] / 255.0, sigma=1), 0, 255) * 255 < 10
    )
    mask_array[:] = cell_mask * not_membrane_mask
    return mask_array


@pytest.fixture()
def labels_array(tmp_path, mask_array: Array) -> Array:
    # generate labels via connected components
    # generate and save some psuedo gt data
    labels_array = prepare_ds(
        tmp_path / "cells3d.zarr/labels",
        mask_array.shape,
        offset=mask_array.offset,
        voxel_size=mask_array.voxel_size,
        axis_names=mask_array.axis_names[1:],
        units=mask_array.units,
        mode="w",
        dtype=np.uint8,
    )
    labels_array[:] = label(mask_array[:])[0]
    return labels_array


@pytest.fixture()
def affs_array(tmp_path, cell_array: Array) -> Array:
    # generate affinity graph
    affs_array = prepare_ds(
        tmp_path / "cells3d.zarr/affs",
        (3,) + cell_array.shape[1:],
        offset=cell_array.offset,
        voxel_size=cell_array.voxel_size,
        axis_names=["neighborhood^"] + cell_array.axis_names[1:],
        units=cell_array.units,
        mode="w",
        dtype=np.uint8,
    )
    affs_array[:] = (
        seg_to_affgraph(labels_array[:], nhood=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 255
    )
    return affs_array


def test_dummy_blockwise(tmpdir):
    class DummyTask(BlockwiseTask):
        task_type: str = "dummy"
        fit: str = "shrink"
        read_write_conflict: bool = False
        task_name: str = "dummy"

        @property
        def write_roi(self):
            return Roi((0, 0), (200, 200))

        @property
        def write_size(self) -> Coordinate:
            return Coordinate(100, 100)

        @property
        def context_size(self) -> Coordinate:
            return Coordinate(0, 0)

        @contextmanager
        def process_block_func(self):
            def process_block(block):
                pass

            yield process_block

        def drop_artifacts(self):
            pass

        def task(
            self, upstream_tasks: Optional[Union[daisy.Task, list[daisy.Task]]] = None
        ) -> daisy.Task:
            # create task
            task = daisy.Task(
                self.task_type,
                total_roi=self.write_roi,
                read_roi=Roi((0, 0), (10, 10)),
                write_roi=Roi((0, 0), (10, 10)),
                process_function=self.worker_func(),
                read_write_conflict=False,
                fit="shrink",
                num_workers=1,
                check_function=self.check_block_func(),
            )

            return task

        def init(self):
            pass

    config = DummyTask()
    config.run_blockwise(multiprocessing=False)


def test_aff_agglom():
    pass


def test_argmax():
    pass


def test_distance_agglom():
    pass


def test_extract_frags():
    pass


def test_global_mws():
    pass


def test_lut():
    pass


def test_predict():
    pass


def test_psuedo_affs():
    pass


def test_seeded_extract_frags():
    pass
