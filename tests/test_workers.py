"""Worker command construction.

The worker command is built in the DRIVER but executed on a scheduler node, in a shell the
driver does not control -- so anything it resolves by name is resolved against the worker's
environment, not the driver's.
"""

import sys
from pathlib import Path

from volara import workers


# ------------------------------------------------- volara-cli resolution (PATH-proof) ---
def test_worker_command_pins_the_cli_beside_this_interpreter(monkeypatch):
    """A bare ``volara-cli`` resolves against the WORKER's PATH, not the driver's. A
    scheduler starts the job in a fresh login shell, so it can pick up a different
    environment -- typically a base conda env with a stale volara missing task
    entry-points -- and the worker then fails task-union validation while the driver
    waits forever for a worker that can never succeed."""
    # LocalWorker.get_command reads the daisy worker context; supply one so this test
    # needs no running daisy server.
    monkeypatch.setenv("DAISY_CONTEXT", "worker_id=0:task_id=t")
    expected = str(Path(sys.executable).parent / "volara-cli")
    cmd = workers.LocalWorker().get_command(Path("/tmp/c.json"), "t")
    if Path(expected).is_file():
        assert cmd[0] == expected, cmd
    else:  # no sibling CLI in this environment -> documented PATH fallback
        assert cmd[0] == "volara-cli", cmd
    assert cmd[1:] == ["blockwise-worker", "-c", "/tmp/c.json"], cmd


def test_cli_falls_back_to_the_bare_name_when_there_is_no_sibling(monkeypatch, tmp_path):
    """Unusual layouts (zipapp, shim on PATH, scripts installed elsewhere) must keep
    working exactly as before rather than pointing at a path that does not exist."""
    monkeypatch.setattr(sys, "executable", str(tmp_path / "nonesuch" / "python"))
    assert workers.Worker.volara_cli() == "volara-cli"


def test_cli_is_absolute_when_a_sibling_exists(monkeypatch, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "volara-cli").write_text("#!/usr/bin/env bash\n")
    monkeypatch.setattr(sys, "executable", str(fake_bin / "python"))
    assert workers.Worker.volara_cli() == str(fake_bin / "volara-cli")
