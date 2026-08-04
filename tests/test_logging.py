"""The log basedir has to reach a worker, and volara does not own the mechanism.

volara spawns each worker as a fresh ``volara-cli`` PROCESS, so nothing that lives only in
a module-global reaches it.  ``BlockwiseTask.process_blocks`` recovers the driver's basedir
from ``daisy.Client().context["logdir"]`` (``volara/blockwise/blockwise.py:260``) -- i.e.
volara depends on **daisy** to carry it across the boundary, in the ``DAISY_CONTEXT``
environment variable.

That dependency is load-bearing well beyond log placement: ``BlockwiseTask.meta_dir`` -- and
therefore ``block_ds`` (``blocks_done.zarr``) -- lives under the basedir.  A driver and a
worker that disagree about it address *different* done-marker stores, the worker's
``open_ds(..., mode="a")`` creates a zarr Group where an Array is expected, and the run
orphans **every** block while the driver log stays clean.

It has already broken once: daisy v2 dropped ``logdir`` from the worker context and replaced
it with a fallback that reads the process-global basedir -- correct for a *forked* worker,
wrong for a spawned one.  Fixed upstream in funkelab/daisy@505e7be.

There was no test in volara crossing a process boundary at all, which is why a change to
someone else's contract could do that silently.  These tests pin the contract volara
actually relies on, so the next change to it fails here instead of in a full-volume run.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import daisy
import pytest

from volara.logging import get_log_basedir, set_log_basedir


def _context(**kwargs):
    """A worker context as the driver builds it."""
    return daisy.Context(hostname="localhost", port=1234, task_id="t", worker_id=0, **kwargs)


def test_volara_set_log_basedir_reaches_the_daisy_context(tmp_path):
    """The driver's basedir must appear in the context daisy hands to workers.

    This is the assumption ``blockwise.py:260`` depends on. It is daisy's to provide, and
    volara has no fallback if it stops -- so it is worth asserting from this side.

    pytest tests/test_logging.py::test_volara_set_log_basedir_reaches_the_daisy_context
    """
    basedir = tmp_path / "driver_basedir"
    set_log_basedir(basedir)
    assert get_log_basedir() == basedir

    # daisy stores the Path itself in the context object and stringifies it in to_env(),
    # so compare as paths here and as strings in the serialized test below.
    assert Path(_context()["logdir"]) == basedir, (
        "daisy's worker context does not carry the driver's log basedir; every spawned "
        "worker will resolve a different meta_dir and address a different blocks_done store"
    )


def test_the_basedir_survives_serialization_to_the_environment(tmp_path, monkeypatch):
    """``logdir`` must survive the round-trip that actually crosses the boundary.

    Being in the context object is not enough -- it reaches a worker only through
    ``DAISY_CONTEXT``, so the encoded form is what matters.

    pytest tests/test_logging.py::test_the_basedir_survives_serialization_to_the_environment
    """
    basedir = tmp_path / "driver_basedir"
    set_log_basedir(basedir)

    encoded = _context().to_env()
    assert "logdir" in encoded, encoded

    # from_env() is the only decoder daisy 1.x exposes, so go through the env var -- which
    # is the real transport anyway.
    monkeypatch.setenv(daisy.Context.ENV_VARIABLE, encoded)
    assert Path(daisy.Context.from_env()["logdir"]) == basedir


def test_a_spawned_process_recovers_the_drivers_basedir(tmp_path):
    """A real child process must resolve the basedir its parent set.

    The genuine cross-boundary check, and the one that was missing: a fresh interpreter
    reading only ``DAISY_CONTEXT`` from its environment, exactly as ``volara-cli
    blockwise-worker`` does.

    pytest tests/test_logging.py::test_a_spawned_process_recovers_the_drivers_basedir
    """
    basedir = tmp_path / "driver_basedir"
    set_log_basedir(basedir)

    env = dict(os.environ)
    env[daisy.Context.ENV_VARIABLE] = _context().to_env()
    # Deliberately NOT inheriting the parent's module state: a fresh import, like a worker.
    child = subprocess.run(
        [sys.executable, "-c", "import daisy, json;"
         " print(json.dumps(daisy.Context.from_env()['logdir']))"],
        capture_output=True, text=True, env=env, check=True,
    )
    assert json.loads(child.stdout.strip()) == str(basedir), (
        f"child resolved {child.stdout.strip()!r}, driver set {str(basedir)!r}"
    )


def test_a_worker_without_a_context_does_not_silently_guess(tmp_path, monkeypatch):
    """With no ``DAISY_CONTEXT``, a worker must fail rather than invent a basedir.

    Silently falling back to a default is what turns a misconfiguration into an
    orphaned run with a clean driver log, so the loud failure is the desirable behaviour
    and worth pinning.

    pytest tests/test_logging.py::test_a_worker_without_a_context_does_not_silently_guess
    """
    monkeypatch.delenv(daisy.Context.ENV_VARIABLE, raising=False)
    with pytest.raises(Exception):
        daisy.Context.from_env()
