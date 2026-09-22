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

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, cast

import daisy
import fsspec
import pytest
from click.testing import CliRunner
from funlib.geometry import Coordinate, Roi
from upath import UPath

from volara.blockwise.blockwise import BlockwiseTask
from volara.cli import cli
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


@pytest.fixture(autouse=True)
def clean_memory_fs():
    """``MemoryFileSystem``'s store is a CLASS attribute shared by every instance.

    A test that leaves files behind (because it failed, or because a later assertion
    never ran) would otherwise hand the next test a half-populated bucket and make the
    failure look like it belongs there.
    """
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs[:] = [""]
    yield
    fs.store.clear()
    fs.pseudo_dirs[:] = [""]


def _one_block(task: BlockwiseTask) -> daisy.Block:
    return daisy.Block(
        total_roi=task.write_roi,
        read_roi=task.write_roi,
        write_roi=task.write_roi,
        block_id=0,
    )


def test_an_s3_basedir_survives_as_a_url():
    """The whole point: ``s3://bucket/x``, not ``s3:/bucket/x``.

    pytest tests/test_upath_basedir.py::test_an_s3_basedir_survives_as_a_url
    """
    # ``UPath("s3://...")`` needs the s3 backend even for pure path algebra; it is an
    # optional dependency, so this test skips (not fails) where it is absent.
    pytest.importorskip("s3fs")
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
    pytest.importorskip("s3fs")
    set_log_basedir("s3://bucket/x")
    task = DummyTask()

    assert str(task.meta_dir) == "s3://bucket/x/dummy-task-meta"
    assert str(task.config_file) == "s3://bucket/x/dummy-task-meta/config.json"
    assert str(task.block_ds) == "s3://bucket/x/dummy-task-meta/blocks_done.zarr"


def test_a_remote_meta_dir_round_trips_the_whole_task_lifecycle():
    """Every meta-dir operation a real run performs, against a non-local filesystem.

    ``memory://`` is fsspec's in-memory filesystem: a real non-local UPath backend with
    no network. Going through the *whole* lifecycle is the point -- a test that only
    mkdirs, writes the config and drops exercises the three operations that already
    worked and misses the two that did not (``init_block_array`` handing a remote
    ``UPath`` to ``prepare_ds``, and ``check_block`` reading ``store.root``, which only
    zarr's ``LocalStore`` has).

    pytest tests/test_upath_basedir.py::test_a_remote_meta_dir_round_trips_the_whole_task_lifecycle
    """
    # zarr < 3.4 resolves a non-local ``UPath`` handed to it by funlib as a LOCAL store
    # (``LocalStore('file://memory://...')``, CI's lowest-direct job on 3.11 pinned zarr
    # 3.1.6 and failed exactly there; 3.4.0 passes). That is a real floor for remote
    # basedirs, recorded in the PR; below it this test cannot mean anything.
    pytest.importorskip("zarr", minversion="3.4")
    set_log_basedir("memory://volara-test-logs")
    task = DummyTask(worker_config=LocalWorker())

    assert str(task.meta_dir) == "memory://volara-test-logs/dummy-task-meta"

    # 1. the block-done store: built by prepare_ds, read back by open_ds
    task.init_block_array()
    assert task.block_ds.exists()

    # 2. the resume check: mark a block done, then see it as done
    block = _one_block(task)
    check_block = task.check_block_func()
    mark_block_done = task.mark_block_done_func()
    assert not check_block(block)
    mark_block_done(block)
    assert check_block(block), (
        "a block marked done in the bucket does not read back as done, so every "
        "resumed run would recompute every block"
    )

    # 3. the config a spawned worker reads: worker_func() writes it
    task.worker_func()
    assert task.config_file.exists()
    assert task.config_file.read_text().startswith("{")

    # 4. --rerun
    task.drop(drop_outputs=False)
    assert not task.meta_dir.exists(), "drop() left the remote meta dir behind"
    assert not task.config_file.exists()
    assert not task.block_ds.exists()


def test_the_worker_cli_accepts_a_remote_config_path():
    """``volara-cli blockwise-worker -c <url>`` is how the config reaches a worker.

    The driver writes the config beside the basedir and passes ``str(config_path)`` to
    the worker command, so with a remote basedir the CLI is handed a URL. A
    ``click.Path(exists=True)`` gate stats that locally and kills the worker with a usage
    error before volara is even imported.

    pytest tests/test_upath_basedir.py::test_the_worker_cli_accepts_a_remote_config_path
    """
    config_file = UPath("memory://volara-test-logs/dummy-task-meta/config.json")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    # valid JSON, not a task volara knows: enough to prove the file was READ, while
    # stopping short of process_blocks(), which needs a daisy server.
    config_file.write_text(json.dumps({"task_type": "not-a-real-task"}))

    result = CliRunner().invoke(cli, ["blockwise-worker", "-c", str(config_file)])

    assert result.exit_code != 0
    assert "does not exist" not in result.output, (
        "click rejected the remote config path on a local stat; no worker can ever read "
        f"a config from a bucket -- {result.output.strip()!r}"
    )
    # got past the option gate and all the way into validating the file's contents
    assert result.exception is not None
    assert "not-a-real-task" in str(result.exception)


def test_the_worker_cli_still_refuses_a_missing_local_config(tmp_path):
    """Dropping click's ``exists=True`` must not soften the local failure.

    The existence check moved from the option parser to the read. A worker pointed at a
    config that is not there has to die saying so -- a worker that starts on an empty or
    half-read config would take blocks and mark them done.

    pytest tests/test_upath_basedir.py::test_the_worker_cli_still_refuses_a_missing_local_config
    """
    missing = tmp_path / "nowhere" / "config.json"

    result = CliRunner().invoke(cli, ["blockwise-worker", "-c", str(missing)])

    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError), result.exception
    assert str(missing) in str(result.exception)

    # a directory is not a config either
    result = CliRunner().invoke(cli, ["blockwise-worker", "-c", str(tmp_path)])
    assert result.exit_code != 0
    assert isinstance(result.exception, IsADirectoryError), result.exception


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
    # ``cast`` rather than an ignore comment: the point is a caller that lies about the
    # type, and ty does not honour mypy-style ``# type: ignore[rule]`` codes.
    with pytest.raises(TypeError):
        set_log_basedir(cast(str, None))
