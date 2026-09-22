"""A remote log basedir must survive as a URL, not decay into a local path.

``pathlib.Path("s3://bucket/x")`` is ``PosixPath("s3:/bucket/x")`` -- one slash gone,
absolute turned relative, and no error anywhere. Every path volara derives from the
basedir (``meta_dir``, ``config_file``, ``block_ds``) inherits that, so a run configured
to log beside its outputs in a bucket instead writes a directory literally named ``s3:``
under whatever the launch directory happened to be. Callers cannot even detect it: the
value they set comes back looking almost right.

``upath.UPath`` keeps the protocol. These tests pin the two halves of that: the URL
survives ``set_log_basedir`` and the joins built on top of it, and a local basedir still
resolves to an ordinary ``pathlib.Path`` so nothing downstream that type-checks or
stringifies a path had its footing moved.

The remote assertions are pure string/path algebra and the round-trip runs on fsspec's
in-memory filesystem, so nothing here touches the network.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import pytest
from funlib.geometry import Coordinate, Roi
from upath import UPath

from volara.blockwise.blockwise import BlockwiseTask
from volara.logging import get_log_basedir, set_log_basedir
from volara.workers import LocalWorker


class DummyTask(BlockwiseTask):
    """The smallest concrete task -- same shape as tests/test_blockwise_base.py."""

    task_type: Literal["dummy"] = "dummy"
    label: str = "dummy-task"

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return self.label

    @property
    def write_roi(self) -> Roi:
        return Roi((0, 0), (10, 10))

    @property
    def write_size(self) -> Coordinate:
        return Coordinate(10, 10)

    @property
    def context_size(self) -> Coordinate:
        return Coordinate(0, 0)

    def drop_artifacts(self):
        pass

    @contextmanager
    def process_block_func(self):
        def process_block(block):
            pass

        yield process_block


@pytest.fixture(autouse=True)
def restore_basedir():
    """``LOG_BASEDIR`` is a module global; do not leak one test's value into the next."""
    previous = get_log_basedir()
    yield
    set_log_basedir(previous)


def test_an_s3_basedir_survives_as_a_url():
    """The whole point: ``s3://bucket/x``, not ``s3:/bucket/x``.

    pytest tests/test_upath_basedir.py::test_an_s3_basedir_survives_as_a_url
    """
    set_log_basedir("s3://bucket/x")

    basedir = get_log_basedir()
    assert str(basedir) == "s3://bucket/x", (
        "the uri lost its protocol; pathlib coercion turns it into the relative local "
        f"path 's3:/bucket/x' -- got {str(basedir)!r}"
    )
    assert isinstance(basedir, UPath), type(basedir)


def test_a_task_derives_its_meta_dir_inside_the_bucket():
    """``meta_dir`` is where the block-done cache and worker logs land.

    It is built by joining onto the basedir, so it is the join -- not just the basedir --
    that has to keep the protocol.

    pytest tests/test_upath_basedir.py::test_a_task_derives_its_meta_dir_inside_the_bucket
    """
    set_log_basedir("s3://bucket/x")
    task = DummyTask()

    assert str(task.meta_dir) == "s3://bucket/x/dummy-task-meta"
    assert str(task.config_file) == "s3://bucket/x/dummy-task-meta/config.json"
    assert str(task.block_ds) == "s3://bucket/x/dummy-task-meta/blocks_done.zarr"


def test_a_remote_meta_dir_round_trips_write_and_drop():
    """mkdir, write the worker config, and drop it again -- all off the local filesystem.

    ``memory://`` is fsspec's in-memory filesystem: a real non-local UPath backend with
    no network. This is what proves the writes go through the path's filesystem rather
    than through ``open()`` and ``shutil.rmtree``, which only ever see local paths.

    pytest tests/test_upath_basedir.py::test_a_remote_meta_dir_round_trips_write_and_drop
    """
    set_log_basedir("memory://volara-test-logs")
    task = DummyTask(worker_config=LocalWorker())

    assert str(task.meta_dir) == "memory://volara-test-logs/dummy-task-meta"

    task.meta_dir.mkdir(parents=True, exist_ok=True)
    assert task.meta_dir.exists()

    # the real config-write path: worker_func() serializes the task for the worker
    task.worker_func()
    assert task.config_file.exists()
    assert task.config_file.read_text().startswith("{")

    task.drop(drop_outputs=False)
    assert not task.meta_dir.exists(), "drop() left the remote meta dir behind"
    assert not task.config_file.exists()


def test_a_local_basedir_is_still_an_ordinary_path(tmp_path):
    """Nothing moved for local runs: a local UPath *is* a ``pathlib.Path``.

    Everything downstream -- funlib's ``prepare_ds``/``open_ds``, ``sqlite3.connect``,
    ``subprocess``, ``isinstance(..., Path)`` checks -- keeps working only because
    ``UPath`` of a local path returns a ``PosixUPath``, which subclasses ``pathlib.Path``.

    pytest tests/test_upath_basedir.py::test_a_local_basedir_is_still_an_ordinary_path
    """
    set_log_basedir(tmp_path / "logs")

    basedir = get_log_basedir()
    assert isinstance(basedir, Path), (
        f"a local basedir resolved to {type(basedir).__name__}, which is not a "
        "pathlib.Path; every consumer that type-checks or os.fspath()es it breaks"
    )
    assert basedir == tmp_path / "logs"

    task = DummyTask()
    assert task.meta_dir == tmp_path / "logs" / "dummy-task-meta"
    assert isinstance(task.meta_dir, Path)


def test_a_local_meta_dir_still_drops(tmp_path):
    """``drop()`` no longer calls ``shutil.rmtree``; the local effect is unchanged.

    pytest tests/test_upath_basedir.py::test_a_local_meta_dir_still_drops
    """
    set_log_basedir(tmp_path / "logs")
    task = DummyTask()
    task.meta_dir.mkdir(parents=True)
    (task.meta_dir / "nested" / "deeper").mkdir(parents=True)
    (task.meta_dir / "nested" / "a_file").write_text("x")

    task.drop(drop_outputs=False)
    assert not task.meta_dir.exists()


def test_none_is_still_not_a_log_basedir():
    """``None`` has to keep raising -- it is rejected by the coercion, not by a check.

    pytest tests/test_upath_basedir.py::test_none_is_still_not_a_log_basedir
    """
    with pytest.raises(TypeError):
        set_log_basedir(None)  # type: ignore[arg-type]
