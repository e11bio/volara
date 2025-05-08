from pathlib import Path

import numpy as np
import pytest
from funlib.geometry import Coordinate
from funlib.persistence import Array, prepare_ds
from scipy.ndimage import label
from skimage import data
from skimage.filters import gaussian

from volara.blockwise import AffAgglom
from volara.datasets import Affs, Labels
from volara.dbs import SQLite
from volara.tmp import seg_to_affgraph


@pytest.fixture()
def cell(tmp_path) -> tuple[Array, Path]:
    cell_data = (data.cells3d().transpose((1, 0, 2, 3)) / 256).astype(np.uint8)
    cell_path = tmp_path / "cells3d.zarr/raw"

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
        cell_path,
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
    return cell_array, cell_path


@pytest.fixture()
def mask(tmp_path, cell: tuple[Array, Path]) -> tuple[Array, Path]:
    cell_array, _cell_path = cell
    mask_path = tmp_path / "cells3d.zarr/mask"
    # generate and save some psuedo gt data
    mask_array = prepare_ds(
        mask_path,
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
    return mask_array, mask_path


@pytest.fixture()
def labels(tmp_path, mask: tuple[Array, Path]) -> tuple[Array, Path]:
    mask_array, _mask_array_path = mask
    labels_path = tmp_path / "cells3d.zarr/labels"
    # generate labels via connected components
    # generate and save some psuedo gt data
    labels_array = prepare_ds(
        labels_path,
        mask_array.shape,
        offset=mask_array.offset,
        voxel_size=mask_array.voxel_size,
        axis_names=mask_array.axis_names,
        units=mask_array.units,
        mode="w",
        dtype=np.uint8,
    )
    labels_array[:] = label(mask_array[:])[0]
    return labels_array, labels_path


@pytest.fixture()
def affs(tmp_path, labels: tuple[Array, Path]) -> tuple[Array, Path]:
    labels_array, _ = labels
    affs_path = tmp_path / "cells3d.zarr/affs"
    # generate affinity graph
    affs_array = prepare_ds(
        affs_path,
        (3,) + labels_array.shape,
        offset=labels_array.offset,
        voxel_size=labels_array.voxel_size,
        axis_names=["neighborhood^"] + labels_array.axis_names,
        units=labels_array.units,
        mode="w",
        dtype=np.uint8,
    )
    affs_array[:] = (
        seg_to_affgraph(labels_array[:], nhood=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 255
    )
    return affs_array, affs_path


@pytest.mark.skip(reason="pytest quitting for some reason")
def test_aff_agglom(affs, labels, tmp_path):
    affs_array, affs_path = affs
    labels_array, labels_path = labels
    db = SQLite(
        path=tmp_path / "db.sqlite",
        node_attrs={"xy_aff": "float", "z_aff": "float"},
    )
    affs_config = AffAgglom(
        db=db,
        affs_data=Affs(store=affs_path, neighborhood=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        frags_data=Labels(store=labels_path),
        block_size=(20, 20, 20),
        context=(2, 2, 2),
        scores={"xy_aff": [(1, 0, 0), (0, 1, 0)], "z_aff": [(0, 0, 1)]},
    )
    affs_config.run_blockwise(multiprocessing=False)

    g = db.open("r").read_graph()
    assert g.number_of_edges() == 0


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
