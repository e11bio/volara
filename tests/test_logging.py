"""The log basedir has to reach a worker, and volara does not own the mechanism.

volara spawns each worker as a fresh ``volara-cli`` PROCESS, so nothing that lives only in
a module-global reaches it.  ``BlockwiseTask.process_blocks`` recovers the driver's basedir
from ``daisy.Client().context["logdir"]`` (``volara/blockwise/blockwise.py:284``) -- i.e.
volara depends on **daisy** to carry it across the boundary, in the ``DAISY_CONTEXT``
environment variable.

That dependency is load-bearing well beyond log placement: ``BlockwiseTask.meta_dir`` -- and
therefore ``block_ds`` (``blocks_done.zarr``) -- lives under the basedir.  A driver and a
worker that disagree about it address *different* done-marker stores, the worker's
``open_ds(..., mode="a")`` creates a zarr Group where an Array is expected, and the run
orphans **every** block while the driver log stays clean.

WHERE daisy carries it moved in daisy 2.0, and these tests pin the new place.  daisy 1.x
put ``logdir`` on every ``Context`` at construction; daisy 2.0 injects it only when
``DAISY_CONTEXT`` is built for a spawned worker (``context_with_logdir()`` in
``daisy/_worker_processes.py``), reading the driver's ``daisy.logging`` global -- which
``volara.logging.set_log_basedir`` forwards to.  A bare ``daisy.Context(...)`` never
carries ``logdir`` under v2, so asserting the 1.x form means asserting a contract daisy
no longer offers anywhere.  What volara actually relies on is the spawn path, and that is
what is pinned here.

The end-to-end form of the same contract -- a real spawned ``volara-cli`` worker marking
blocks done in the store the driver prepared -- is already exercised by every
multiprocessing test in this suite: ``tests/conftest.py``'s autouse ``logdir`` fixture
points each test at a private basedir, so a worker that fell back to the default would
address a different ``blocks_done`` store and fail those tests loudly.
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
    """A worker context as daisy's server builds it -- ``logdir``-less under v2."""
    return daisy.Context(hostname="localhost", port=1234, task_id="t", worker_id=0, **kwargs)


def _spawn_context():
    """The context exactly as daisy hands it to a spawned worker.

    ``context_with_logdir`` is what daisy applies when it builds ``DAISY_CONTEXT`` for a
    worker subprocess.  It is private, and importing it here is deliberate: volara's
    worker reads what that function wrote, so if daisy renames or reshapes it, volara
    wants to find out from this file rather than from a full-volume run.
    """
    from daisy._worker_processes import context_with_logdir

    return context_with_logdir(_context())


def test_volara_set_log_basedir_reaches_the_daisy_context(tmp_path):
    """The driver's basedir must appear in the context daisy hands to workers.

    This is the assumption ``blockwise.py:284`` depends on.  Under daisy 2.0 it holds at
    the spawn boundary, not at construction -- both halves are asserted, so this cannot
    drift back to the 1.x form that no longer exists.

    pytest tests/test_logging.py::test_volara_set_log_basedir_reaches_the_daisy_context
    """
    basedir = tmp_path / "driver_basedir"
    set_log_basedir(basedir)
    assert get_log_basedir() == basedir

    # v2 semantics: a bare Context carries no logdir; only the spawn path adds it.
    assert "logdir" not in _context()

    assert Path(_spawn_context()["logdir"]) == basedir, (
        "daisy's spawn-path context does not carry the driver's log basedir; every "
        "spawned worker will resolve a different meta_dir and address a different "
        "blocks_done store"
    )


def test_the_basedir_survives_serialization_to_the_environment(tmp_path, monkeypatch):
    """``logdir`` must survive the round-trip that actually crosses the boundary.

    Being in the spawn-path context object is not enough -- it reaches a worker only
    through ``DAISY_CONTEXT``, so the encoded form is what matters.

    pytest tests/test_logging.py::test_the_basedir_survives_serialization_to_the_environment
    """
    basedir = tmp_path / "driver_basedir"
    set_log_basedir(basedir)

    encoded = _spawn_context().to_env()
    assert "logdir" in encoded, encoded

    # from_env() is the decoder the worker side uses, so go through the env var -- which
    # is the real transport anyway.
    monkeypatch.setenv(daisy.Context.ENV_VARIABLE, encoded)
    assert Path(daisy.Context.from_env()["logdir"]) == basedir


def test_a_spawned_process_recovers_the_drivers_basedir(tmp_path):
    """A real child process must resolve the basedir its parent set.

    The genuine cross-boundary check: a fresh interpreter reading only ``DAISY_CONTEXT``
    from its environment, exactly as ``volara-cli blockwise-worker`` does -- with that
    variable holding what daisy's spawn path puts there.

    pytest tests/test_logging.py::test_a_spawned_process_recovers_the_drivers_basedir
    """
    basedir = tmp_path / "driver_basedir"
    set_log_basedir(basedir)

    env = dict(os.environ)
    env[daisy.Context.ENV_VARIABLE] = _spawn_context().to_env()
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
