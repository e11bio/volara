import json
import textwrap
from pathlib import Path

import daisy
import numpy as np
import pytest
import torch
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.blockwise import (
    LUT,
    AffAgglom,
    Argmax,
    DistanceAgglom,
    ExtractFrags,
    GlobalMWS,
    Predict,
    SeededExtractFrags,
)
from volara.datasets import Affs, Labels, Raw, Dataset
from volara.dbs import SQLite
from volara.models import Checkpoint

BLOCK = daisy.Block(
    total_roi=Roi((0, 0), (10, 10)),
    read_roi=Roi((0, 0), (10, 10)),
    write_roi=Roi((0, 0), (10, 10)),
)


def build_zarr(
    tmpdir: Path,
    name: str,
    data: np.ndarray,
    spatial_dims: int,
    neighborhood: list[Coordinate] | None = None,
) -> Dataset:
    arr = prepare_ds(
        tmpdir / "test_data.zarr" / name,
        shape=data.shape,
        voxel_size=Coordinate((1,) * spatial_dims),
        dtype=data.dtype,
        mode="w",
    )
    arr[:] = data
    dataset_type = {
        "raw": Raw,
        "probs": Raw,
        "affs": Affs,
        "frags": Labels,
        "segments": Labels,
        "labels": Labels,
    }[name]

    kwargs = {}
    if neighborhood is not None:
        kwargs["neighborhood"] = neighborhood
    return dataset_type(store=tmpdir / "test_data.zarr" / name, **kwargs)


def build_db(tmpdir: Path) -> SQLite:
    data_dir = tmpdir / "test_data.zarr"
    if not data_dir.exists():
        data_dir.mkdir()
    db_config = SQLite(
        path=tmpdir / "test_data.zarr" / "db.sqlite",
        node_attrs={"raw_intensity": 1},
        edge_attrs={
            "y_aff": "float",
        },
        ndim=2,
    )
    db_config.init()
    return db_config


def test_aff_agglom(tmpdir):
    tmpdir = Path(tmpdir)
    db_config = build_db(tmpdir)

    # create fragment labels array
    frags_data = np.zeros((10, 10), dtype=np.uint32)
    frags_data[:, :] = np.arange(1, 11)[:, None]  # horizontal stripes
    frags = build_zarr(tmpdir, "frags", frags_data, 2)

    # add fragment nodes to db
    db = db_config.open("r+")
    empty_graph = db.read_graph()
    for i in range(1, 11):
        empty_graph.add_node(i, position=(i, 5), size=1, raw_intensity=(i,))
    db.write_graph(empty_graph)

    # create affs data
    affs_data = np.zeros((1, 10, 10), dtype=np.uint32)
    affs_data[0, ::2, :] = np.ones((5, 1))  # vertical affs 1 in every other row
    affs = build_zarr(tmpdir, "affs", affs_data, 2, neighborhood=[Coordinate(1, 0)])

    # affs config
    aff_agglom_config = AffAgglom(
        db=db_config,
        frags_data=frags,
        affs_data=affs,
        block_size=Coordinate(10, 10),
        context=Coordinate(0, 0),
        scores={"y_aff": [Coordinate(1, 0)]},
    )

    with aff_agglom_config.process_block_func() as process_block:
        process_block(BLOCK)

    # Check that we got the expected results
    db = db_config.open("r")
    g = db.read_graph(BLOCK.write_roi)
    assert g.number_of_nodes() == 10
    assert g.number_of_edges() == 9
    for u, v, data in g.edges(data=True):
        if u % 2 == 0:
            assert data["y_aff"] == 0.0
        else:
            assert data["y_aff"] == 1.0


def test_argmax(tmpdir):
    tmpdir = Path(tmpdir)

    # create raw intensities array
    probs_data = np.arange(1, 201, dtype=np.float32).reshape(2, 10, 10)
    probs = build_zarr(tmpdir, "probs", probs_data, 2)
    labels = build_zarr(tmpdir, "labels", np.zeros((10, 10), dtype=np.uint32), 2)

    # affs config
    aff_agglom_config = Argmax(
        probs_data=probs,
        sem_data=labels,
        block_size=Coordinate(10, 10),
    )

    with aff_agglom_config.process_block_func() as process_block:
        process_block(BLOCK)

    assert np.isclose(labels.array("r")[:], np.ones((10, 10))).all()


def test_create_lut(tmpdir):
    pass


def test_distance_agglom(tmpdir):
    pass


def test_distance_agglom_simple(tmpdir):
    pass


def test_dummy(tmpdir):
    pass


def test_extract_frags(tmpdir):
    pass


def test_lut(tmpdir):
    pass


def test_predict(tmpdir):
    pass


def test_seeded_extract_frags(tmpdir):
    pass


def test_threshold(tmpdir):
    pass

