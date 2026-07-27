"""Workers must satisfy daisy v2's blocking-spawn contract.

daisy v2's worker accounting (``max_workers``, the start budget, abandonment) tracks the
spawn-function call: while the spawned worker command is running, that worker slot counts
as alive; when it returns, the slot is considered dead and eligible for replacement (see
daisy's MIGRATION.md, "blocking-spawn contract"). volara previously fire-and-forgot
``sbatch`` (the client exits the instant the job is queued), so daisy could never reap
the real slurm job and any worker that did not cleanly self-exit leaked until walltime,
piling up across stages (the observed ~99-node worker storm, reaped by hand with
``scancel --name``).

The fix is to make every backend's submission command ITSELF block for the worker's
lifetime -- ``run_worker`` stays a plain ``subprocess.run(cmd)``: a child process
locally, ``srun`` on slurm (which also ties the job to the client: cancelling the driver
cancels the steps), and ``bsub -K`` on LSF.

These tests need NO real cluster: fake ``srun``/``bsub`` executables are prepended to
``PATH`` so the availability probes succeed and command construction runs for real. A
``slurm`` marker is registered in pyproject.toml for any future test that genuinely
needs a slurm install (run those with ``-m slurm``); none currently do.
"""

import os
import stat

import pytest

from volara import workers

_BODIES = {
    # is_srun_available() probes `srun --version`.
    "srun": 'echo "slurm-wlm 21.08.0"; exit 0',
    # is_bsub_available() probes `bsub -V` (prints to stderr on real LSF).
    "bsub": 'echo "IBM Spectrum LSF 10.1"; exit 0',
}


@pytest.fixture()
def cluster_shims(tmp_path, monkeypatch):
    """Install fake srun/bsub on PATH so availability probes pass without a cluster."""
    bindir = tmp_path / "cluster_shim_bin"
    bindir.mkdir()
    for prog, body in _BODIES.items():
        script = bindir / prog
        script.write_text(f"#!/usr/bin/env bash\n{body}\n")
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


def test_slurm_command_is_blocking_srun(cluster_shims):
    """Slurm workers submit via srun: it blocks for the job's lifetime (the spawn
    contract) and ties the job to the client, unlike fire-and-forget sbatch."""
    w = workers.SlurmWorker(queue="gpu-q", num_gpus=1, num_cpus=4)
    cmd = w.get_slurm_command(
        command=["volara-cli", "blockwise-worker", "-c", "c.json"],
        job_name="mytask",
        queue="gpu-q",
        num_gpus=1,
        num_cpus=4,
    )
    assert cmd[0] == "srun", cmd
    assert "--job-name=mytask" in cmd, cmd
    assert "--partition=gpu-q" in cmd, cmd
    assert "--gpus=1" in cmd and "--cpus-per-task=4" in cmd, cmd
    # the worker command rides along as argv (srun has no sbatch-style --wrap)
    assert cmd[-4:] == ["volara-cli", "blockwise-worker", "-c", "c.json"], cmd
    assert not any(a == "sbatch" or a.startswith("--wrap") for a in cmd), cmd


def test_lsf_command_is_blocking_bsub(cluster_shims):
    """LSF workers submit via bsub -K (submit and wait for completion)."""
    w = workers.LSFWorker(queue="gpu-q")
    cmd = w.get_lsf_command(
        command=["volara-cli", "blockwise-worker", "-c", "c.json"],
        job_name="mytask",
        queue="gpu-q",
    )
    assert cmd[:2] == ["bsub", "-K"], cmd
    assert "-J" in cmd and cmd[cmd.index("-J") + 1] == "mytask", cmd
    assert cmd[-4:] == ["volara-cli", "blockwise-worker", "-c", "c.json"], cmd


def test_get_command_names_job_after_task(cluster_shims, monkeypatch, tmp_path):
    """End-to-end: get_command (what daisy's spawn function runs) names the slurm
    worker job after its task, so workers are identifiable in squeue."""
    import daisy

    # Provide the daisy worker context get_command reads, without a running server.
    monkeypatch.setenv("DAISY_CONTEXT", "worker_id=0:task_id=mytask")
    monkeypatch.setattr(
        daisy.logging, "get_worker_log_basename", lambda wid, tid: tmp_path
    )

    cmd = workers.SlurmWorker(queue="gpu-q").get_command(tmp_path / "c.json", "mytask")
    assert cmd[0] == "srun" and "--job-name=mytask" in cmd, cmd
