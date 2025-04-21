from pathlib import Path

import daisy
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.datasets import Affs, Dataset, Labels, Raw
from volara.dbs import SQLite

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
