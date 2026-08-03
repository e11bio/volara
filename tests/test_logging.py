"""The log basedir has to survive a process boundary.

volara spawns each worker as a fresh ``volara-cli`` PROCESS, so anything that only
lives in a module-global is invisible to it. That matters beyond log placement:
``BlockwiseTask.meta_dir`` -- and therefore ``block_ds`` (``blocks_done.zarr``) --
lives under the basedir, so a driver and a worker that disagree about it address
different done-marker stores and every block is orphaned with no driver-visible cause.
"""

import subprocess
import sys

from volara.logging import get_log_basedir, set_log_basedir

_CHILD = "import volara.logging as L; print(L.get_log_basedir())"


def test_basedir_survives_a_spawned_process(tmp_path):
    """A child process must resolve the basedir its parent set.

    pytest tests/test_logging.py::test_basedir_survives_a_spawned_process
    """
    basedir = tmp_path / "driver_basedir"
    set_log_basedir(basedir)
    assert get_log_basedir() == basedir

    child = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, check=True
    )
    assert child.stdout.strip() == str(basedir), (
        f"child resolved {child.stdout.strip()!r}, parent set {str(basedir)!r} -- a "
        "spawned worker would address a different blocks_done store"
    )


def test_default_is_unchanged_when_nobody_sets_it(tmp_path, monkeypatch):
    """With no basedir set anywhere, the default still applies.

    pytest tests/test_logging.py::test_default_is_unchanged_when_nobody_sets_it
    """
    monkeypatch.delenv("VOLARA_LOG_BASEDIR", raising=False)
    monkeypatch.chdir(tmp_path)
    child = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, check=True
    )
    assert child.stdout.strip() == "volara_logs"
