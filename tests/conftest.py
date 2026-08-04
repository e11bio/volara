from pathlib import Path

import daisy
import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import prepare_ds

from volara.dbs import SQLite
from volara.logging import set_log_basedir
from volara.segment_utils import seg_to_affgraph


@pytest.fixture(autouse=True)
def logdir(tmp_path):
    set_log_basedir(tmp_path / "volara_logs")


@pytest.fixture()
def zarr_2d(tmp_path) -> tuple[Path, np.ndarray]:
    """10x10 float32 array, voxel_size=(1,1), values in [0,1]."""
    data = np.random.default_rng(42).random((10, 10), dtype=np.float32)
    path = tmp_path / "test.zarr" / "raw"
    arr = prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )
    arr[:] = data
    return path, data


@pytest.fixture()
def labels_2d(tmp_path) -> tuple[Path, np.ndarray]:
    """10x10 uint64 with 4 labeled regions (2x2 grid of 5x5 blocks)."""
    data = np.zeros((10, 10), dtype=np.uint64)
    data[:5, :5] = 1
    data[:5, 5:] = 2
    data[5:, :5] = 3
    data[5:, 5:] = 4
    path = tmp_path / "test.zarr" / "labels"
    arr = prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )
    arr[:] = data
    return path, data


@pytest.fixture()
def affs_2d(tmp_path, labels_2d) -> tuple[Path, np.ndarray]:
    """(2,10,10) affinities from labels_2d using seg_to_affgraph."""
    labels_path, labels_data = labels_2d
    nhood = [[1, 0], [0, 1]]
    affs_data = seg_to_affgraph(labels_data, nhood=nhood).astype(np.float32)
    path = tmp_path / "test.zarr" / "affs"
    arr = prepare_ds(
        path,
        shape=affs_data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=affs_data.dtype,
        mode="w",
    )
    arr[:] = affs_data
    arr._source_data.attrs["neighborhood"] = nhood
    return path, affs_data


@pytest.fixture()
def frags_2d(tmp_path) -> tuple[Path, np.ndarray]:
    """10x10 uint64 with 10 horizontal stripe fragments (1..10)."""
    data = np.zeros((10, 10), dtype=np.uint64)
    data[:, :] = np.arange(1, 11)[:, None]
    path = tmp_path / "test.zarr" / "frags"
    arr = prepare_ds(
        path,
        shape=data.shape,
        voxel_size=Coordinate(1, 1),
        dtype=data.dtype,
        mode="w",
    )
    arr[:] = data
    return path, data


@pytest.fixture()
def sqlite_db_2d(tmp_path) -> SQLite:
    """SQLite DB with ndim=2, ready for use."""
    db_config = SQLite(
        path=tmp_path / "test.zarr" / "db.sqlite",
        node_attrs={"raw_intensity": 1},
        edge_attrs={"y_aff": "float"},
        ndim=2,
    )
    db_config.init()
    return db_config


@pytest.fixture()
def block_2d() -> daisy.Block:
    """daisy.Block covering Roi((0,0),(10,10))."""
    return daisy.Block(
        total_roi=Roi((0, 0), (10, 10)),
        read_roi=Roi((0, 0), (10, 10)),
        write_roi=Roi((0, 0), (10, 10)),
    )


# ---------------------------------------------------------------------------
# bulk_write helpers, shared by test_extract_frags.py and test_aff_agglom.py
# ---------------------------------------------------------------------------

BULK_NEIGHBORHOOD = [[1, 0], [0, 1]]
BULK_SCORES = {"aff": [Coordinate(1, 0), Coordinate(0, 1)]}


@pytest.fixture()
def blocky_affs(tmp_path) -> Path:
    """Affinities from a 4x4 grid of 5x5 labels over a 20x20 volume.

    Sixteen regions gives several fragments per block and plenty of adjacencies

    The label-derived affinities are deliberately perturbed with noise. Raw
    `seg_to_affgraph` output is exactly 0 on every inter-region boundary, which
    makes every edge weight exactly 0.0 - and then comparing bulk against
    non-bulk weights cannot tell any two blocks apart, which defeats the point of
    the comparison. Varying the boundary values makes a partial-boundary mean
    differ from a whole-boundary one. The perturbation stays well inside the -0.5
    bias, so mws still recovers the 16 regions.
    """
    labels = np.zeros((20, 20), dtype=np.uint64)
    label = 1
    for z in range(0, 20, 5):
        for y in range(0, 20, 5):
            labels[z : z + 5, y : y + 5] = label
            label += 1

    clean = seg_to_affgraph(labels, nhood=BULK_NEIGHBORHOOD).astype(np.float32)
    noise = np.random.default_rng(0).random(clean.shape).astype(np.float32)
    affs = (clean * 0.75 + noise * 0.25).astype(np.float32)

    path = tmp_path / "test.zarr" / "bulk_affs"
    arr = prepare_ds(
        path,
        shape=affs.shape,
        voxel_size=Coordinate(1, 1),
        dtype=affs.dtype,
        mode="w",
    )
    arr[:] = affs
    arr._source_data.attrs["neighborhood"] = BULK_NEIGHBORHOOD
    return path


class BulkHarness:
    """Runs extract-frags / aff-agglom serially and reads back the DB rows.

    Tasks are run through `run_blockwise(multiprocessing=False)`, because bulk
    mode needs the orchestrator-side context that `task()` enters: that is what
    drops and rebuilds the indexes, and on SQLite it is also what switches the
    file from rollback-journal to WAL. Driving `process_block_func` alone leaves
    that switch to happen while a write connection is already open, which SQLite
    rejects with "database is locked".
    """

    def __init__(self, tmp_path, affs_path):
        self.tmp_path = tmp_path
        self.affs_path = affs_path

    def db(self, tag) -> SQLite:
        return SQLite(
            path=self.tmp_path / f"bulk_{tag}.sqlite",
            node_attrs={"size": "int"},
            edge_attrs={"aff": "float"},
            ndim=2,
        )

    def frags_path(self, tag) -> Path:
        return self.tmp_path / "test.zarr" / f"bulk_frags_{tag}"

    @staticmethod
    def _run(task):
        """Run serially, asserting every block succeeded.

        Without this a task that failed every block would leave an empty DB and
        every comparison in the callers would pass
        """
        state = task.run_blockwise(multiprocessing=False)[task.task_name]
        assert state.failed_count == 0, f"{task.task_name}: {state.failed_count} failed"
        assert state.orphaned_count == 0
        assert state.completed_count > 0

    def extract_frags(self, db, tag, bulk_write, block_size, context) -> np.ndarray:
        from volara.blockwise import ExtractFrags
        from volara.datasets import Affs, Labels

        task = ExtractFrags(
            db=db,
            affs_data=Affs(store=self.affs_path),
            frags_data=Labels(store=self.frags_path(tag)),
            block_size=block_size,
            context=context,
            bias=[-0.5, -0.5],
            # noise_eps left unset: both runs must see identical affinities for
            # their fragments, and hence their node rows, to be comparable.
            bulk_write=bulk_write,
        )
        self._run(task)
        return task.frags_data.array("r")[:]

    def aff_agglom(self, db, tag, bulk_write, block_size, context) -> None:
        from volara.blockwise import AffAgglom
        from volara.datasets import Affs, Labels

        self._run(
            AffAgglom(
                db=db,
                frags_data=Labels(store=self.frags_path(tag)),
                affs_data=Affs(store=self.affs_path),
                block_size=block_size,
                context=context,
                scores=BULK_SCORES,
                bulk_write=bulk_write,
            )
        )

    def pipeline(self, tag, bulk_write, block_size, context):
        """extract-frags then aff-agglom into one DB

        AffAgglom needs the nodes ExtractFrags writes, and on SQLite it also needs
        the journal-mode switch ExtractFrags' bulk context already performed.
        """
        db = self.db(tag)
        frags = self.extract_frags(db, tag, bulk_write, block_size, context)
        self.aff_agglom(db, tag, bulk_write, block_size, context)
        return db, frags

    @staticmethod
    def nodes(db) -> dict:
        graph = db.open("r").read_graph()
        return {
            int(node): (tuple(int(p) for p in attrs["position"]), int(attrs["size"]))
            for node, attrs in graph.nodes(data=True)
        }

    @staticmethod
    def edges(db) -> dict:
        graph = db.open("r").read_graph()
        return {
            (min(int(u), int(v)), max(int(u), int(v))): dict(data)
            for u, v, data in graph.edges(data=True)
        }


@pytest.fixture()
def bulk(tmp_path, blocky_affs) -> BulkHarness:
    return BulkHarness(tmp_path, blocky_affs)
