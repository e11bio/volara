"""Worker teardown: cluster worker jobs must be reaped when ``run_blockwise`` returns.

Regression test for the worker-teardown leak. volara fire-and-forgets ``sbatch`` (no --wait,
the sbatch client exits the instant the job is queued), so daisy's per-worker ``terminate()``
kills only the already-dead sbatch client and can NEVER scancel the real slurm job. Any worker
that did not cleanly self-exit on ``block is None`` (e.g. one that finished connecting after its
run_blockwise server had already shut down) leaked until walltime and piled up across stages
(the observed ~99-node worker storm; operators reaped it by hand with ``scancel --name``).

The fix is a name-scoped teardown sweep: workers are submitted with ``--job-name=<task_name>``
and ``Worker.cleanup(task_name)`` runs in ``run_blockwise``'s ``finally`` -- after daisy has
stopped its pools and every block is terminal, so a name-scoped scancel cannot touch in-flight
work. ``SlurmWorker.cleanup`` runs ``scancel --name <task_name>``; the base/local worker is a
no-op. (This sweep is a safety net on TOP of the per-worker reap in ``SlurmWorker.run``, which
lives in the daisy-v2 migration branch.)

These tests need NO real cluster: they prepend fake ``sbatch``/``scancel``/``squeue`` executables
to ``PATH`` (each records its argv to a file), so the reap path runs for real and only the slurm
binaries are fake. No test here requires a real slurm install; a ``slurm`` marker is registered in
pyproject.toml for any future test that genuinely does (run those with ``-m slurm``).

Guarded on the daisy v2 surface (``daisy.v2``): the migrated ``volara.workers`` targets daisy v2
(``get_command`` calls ``daisy.Context.from_env`` / ``daisy.logging.get_worker_log_basename``),
which is a Rust extension not buildable in every env. Runtime validation is deferred to a
v2-built CI environment.
"""

import os
import stat

import pytest

pytest.importorskip("daisy.v2")

from volara import workers  # noqa: E402  (after importorskip guard)

_SHIM = """#!/usr/bin/env bash
# Record this invocation (program name + args), one per line, then behave like the real tool.
printf '%s' "{prog}" >> "$SLURM_SHIM_CALLS"
for a in "$@"; do printf '\\t%s' "$a" >> "$SLURM_SHIM_CALLS"; done
printf '\\n' >> "$SLURM_SHIM_CALLS"
{body}
"""

_BODIES = {
    # is_sbatch_available() runs `sbatch --version`; a real submit prints a job id.
    "sbatch": 'if [ "$1" = "--version" ]; then echo "slurm-wlm 21.08.0"; fi; echo "12345"; exit 0',
    "scancel": "exit 0",
    "squeue": "exit 0",
}


@pytest.fixture()
def slurm_shims(tmp_path, monkeypatch):
    """Install fake sbatch/scancel/squeue on PATH; return a reader for the recorded calls."""
    bindir = tmp_path / "slurm_shim_bin"
    bindir.mkdir()
    calls_file = tmp_path / "slurm_shim_calls.tsv"

    for prog, body in _BODIES.items():
        script = bindir / prog
        script.write_text(_SHIM.format(prog=prog, body=body))
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    monkeypatch.setenv("SLURM_SHIM_CALLS", str(calls_file))
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")

    def read_calls():
        if not calls_file.exists():
            return []
        return [
            line.split("\t")
            for line in calls_file.read_text().splitlines()
            if line.strip()
        ]

    return read_calls


def test_cleanup_scancels_by_task_name(slurm_shims):
    """The teardown sweep must scancel every worker job by the task's name."""
    task = "combined_volume-gel2-slabreg_composed_flatten_warp"
    workers.SlurmWorker(queue="gpu-q").cleanup(task)
    calls = slurm_shims()
    assert ["scancel", "--name", task] in calls, calls
    # scancel is scoped by --name (never a blanket cancel of unrelated jobs).
    scancels = [c for c in calls if c[0] == "scancel"]
    assert len(scancels) == 1 and scancels[0][1] == "--name", scancels


def test_base_worker_cleanup_is_noop(slurm_shims):
    """Local/base workers have no cluster jobs to reap: cleanup must not scancel anything."""
    workers.LocalWorker().cleanup("anytask")
    assert [c for c in slurm_shims() if c[0] == "scancel"] == []


def test_get_slurm_command_names_job_after_task(slurm_shims):
    """The sweep is name-scoped, so submitted workers MUST carry --job-name=<task_name>."""
    w = workers.SlurmWorker(queue="gpu-q")
    cmd = w.get_slurm_command(
        command="volara-cli blockwise-worker -c c.json",
        queue="gpu-q",
        job_name="mytask",
        expand=False,
    )
    assert "--job-name=mytask" in cmd, cmd


def test_get_command_wires_task_name_into_job_name(slurm_shims, monkeypatch, tmp_path):
    """End-to-end: get_command (what daisy calls to build the sbatch line) names the worker
    job after its task, which is exactly what cleanup's scancel --name relies on."""
    import daisy

    # Provide the daisy worker context get_command reads, without a running server.
    monkeypatch.setenv("DAISY_CONTEXT", "worker_id=0:task_id=mytask")
    # daisy v2 moved get_worker_log_basename to daisy.logging (see the migration branch).
    monkeypatch.setattr(daisy.logging, "get_worker_log_basename", lambda wid, tid: tmp_path)

    cmd = workers.SlurmWorker(queue="gpu-q").get_command(tmp_path / "c.json", "mytask")
    assert "--job-name=mytask" in cmd, cmd
