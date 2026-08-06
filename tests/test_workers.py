"""Workers must satisfy daisy v2's blocking-spawn contract.

daisy v2's worker accounting (``max_workers``, the start budget, abandonment) tracks the
spawn-function call: while ``Worker.run`` is running, that worker slot counts as alive;
when it returns, the slot is considered dead and eligible for replacement (see daisy's
MIGRATION.md, "blocking-spawn contract"). volara previously fire-and-forgot ``sbatch``
(the client exits the instant the job is queued), so daisy could never reap the real
slurm job and any worker that did not cleanly self-exit leaked until walltime, piling up
across stages (the observed ~99-node worker storm, reaped by hand with ``scancel --name``).

The fix is to make every backend's submission command itself block for the worker's
lifetime: a plain child process locally, ``srun`` on slurm (which also ties the job to
the client: cancelling the driver cancels the steps), and ``bsub -K`` on LSF.

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
    # is_sbatch_available() probes `sbatch --version`.
    "sbatch": 'echo "slurm-wlm 21.08.0"; exit 0',
    # scancel is invoked by SlurmWorker.run's teardown.
    "scancel": 'echo "$@" >> "$SCANCEL_LOG"; exit 0',
    # is_bsub_available() probes `bsub -V` (prints to stderr on real LSF).
    "bsub": 'echo "IBM Spectrum LSF 10.1"; exit 0',
}


@pytest.fixture()
def cluster_shims(tmp_path, monkeypatch):
    """Install fake sbatch/scancel/bsub on PATH so the availability probes and the
    teardown run without a cluster. ``SCANCEL_LOG`` collects what teardown cancelled."""
    bindir = tmp_path / "cluster_shim_bin"
    bindir.mkdir()
    monkeypatch.setenv("SCANCEL_LOG", str(tmp_path / "scancel.log"))
    for prog, body in _BODIES.items():
        script = bindir / prog
        script.write_text(f"#!/usr/bin/env bash\n{body}\n")
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


def test_slurm_command_is_blocking_sbatch_wait(cluster_shims):
    """Slurm workers submit via ``sbatch --wait``: it blocks for the job's lifetime
    (the spawn contract) AND submits an independent job, so workers reach the
    requested partition even when the driver is itself a slurm job.

    ``srun`` also blocks, but inside an allocation it makes a job STEP -- confining
    every worker to the driver's own nodes and silently ignoring ``--partition``.

    Asserted as full-argv equality, not membership: a repeated flag or a repeated
    worker command is invisible to ``in``/suffix checks, and a doubled worker command
    is exactly what click rejects with "Got unexpected extra arguments" -- killing
    every block of every stage.
    """
    w = workers.SlurmWorker(queue="gpu-q", num_gpus=1, num_cpus=4)
    cmd = w.get_slurm_command(
        command=["volara-cli", "blockwise-worker", "-c", "c.json"],
        job_name="mytask",
        queue="gpu-q",
        num_gpus=1,
        num_cpus=4,
    )
    assert cmd == [
        "sbatch",
        "--wait",
        "--job-name=mytask",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--gpus=1",
        "--mem=15564",
        "--partition=gpu-q",
        "--output=%x_%j.log",
        "--error=%x_%j.err",
        "--wrap=volara-cli blockwise-worker -c c.json",
    ], cmd
    assert cmd.count("--wrap=volara-cli blockwise-worker -c c.json") == 1, cmd
    assert "srun" not in cmd, cmd


def test_ntasks_is_exactly_one(cluster_shims):
    """REGRESSION. Without an explicit ``--ntasks`` slurm derives it from the
    allocation, so ONE submission expands into that many identical clones sharing a
    single DAISY_CONTEXT (hence one worker_id, the race daisy warns about) while every
    later worker blocks on "step creation still disabled". Measured on a 32-CPU driver
    with num_cpus=4: one step, Tasks=8, and 623 retries."""
    cmd = workers.SlurmWorker(queue="q", num_cpus=4).get_slurm_command(
        command=["volara-cli"], num_cpus=4
    )
    assert cmd.count("--ntasks=1") == 1, cmd


def test_wrap_quotes_arguments_containing_spaces(cluster_shims):
    """--wrap is a shell string, so a path with a space must survive as one arg."""
    cmd = workers.SlurmWorker(queue="q").get_slurm_command(
        command=["volara-cli", "-c", "/a path/c.json"]
    )
    assert cmd[-1] == "--wrap=volara-cli -c '/a path/c.json'", cmd


def test_time_limit_emitted_only_when_set(cluster_shims):
    """An independent worker job outlives its client, and a SIGKILLed driver runs no
    teardown at all -- so ``--time`` is the last bound. Do not assume the cluster has
    one: partitions may be DefaultTime=NONE / MaxTime=UNLIMITED."""
    assert not any(
        a.startswith("--time=")
        for a in workers.SlurmWorker(queue="q").get_slurm_command(command=["v"])
    )
    w = workers.SlurmWorker(queue="q", time_limit="4:00:00")
    assert "--time=4:00:00" in w.get_slurm_command(command=["v"], time_limit="4:00:00")


def test_lsf_command_is_blocking_bsub(cluster_shims):
    """LSF workers submit via bsub -K (submit and wait for completion).

    Full-argv equality for the same reason as the slurm case above.
    """
    w = workers.LSFWorker(queue="gpu-q")
    cmd = w.get_lsf_command(
        command=["volara-cli", "blockwise-worker", "-c", "c.json"],
        job_name="mytask",
        queue="gpu-q",
    )
    assert cmd == [
        "bsub",
        "-K",
        "-J",
        "mytask",
        "-n",
        "1",
        "-q",
        "gpu-q",
        "volara-cli",
        "blockwise-worker",
        "-c",
        "c.json",
    ], cmd
    assert cmd.count("blockwise-worker") == 1, cmd
    assert cmd.count("-J") == 1, cmd


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
    assert cmd[0] == "sbatch" and cmd[1] == "--wait", cmd
    assert "--job-name=mytask" in cmd, cmd


def test_get_command_threads_the_time_limit(cluster_shims, monkeypatch, tmp_path):
    """time_limit is a field on the worker, so it must reach the emitted command."""
    import daisy

    monkeypatch.setenv("DAISY_CONTEXT", "worker_id=0:task_id=mytask")
    monkeypatch.setattr(daisy.logging, "get_worker_log_basename", lambda wid, tid: tmp_path)

    cmd = workers.SlurmWorker(queue="gpu-q", time_limit="2:00:00").get_command(
        tmp_path / "c.json", "mytask"
    )
    assert "--time=2:00:00" in cmd, cmd


# ------------------------------------------------------- teardown (scancel by id) ---
def _fake_sbatch(tmp_path, body):
    """Rewrite the sbatch shim so a test can script the job's behaviour."""
    s = tmp_path / "cluster_shim_bin" / "sbatch"
    s.write_text("#!/usr/bin/env bash\n" + body + "\n")
    s.chmod(0o755)


def test_run_blocks_and_returns_the_jobs_exit_status(cluster_shims, tmp_path):
    """``sbatch --wait`` returns only when the job ends, and with the job's own exit
    code -- that IS the blocking-spawn contract, with no polling wrapper. Confirmed
    against a real scheduler too: rc=7 propagated after a 171 s wait."""
    _fake_sbatch(tmp_path, 'echo "Submitted batch job 4242"; sleep 0.2; exit 7')
    assert workers.SlurmWorker(queue="q").run(["sbatch", "--wait"]).returncode == 7


def test_run_cancels_its_job_by_id_on_exit(cluster_shims, tmp_path):
    """The job MUST be cancelled by the client because it does not die with it --
    measured on a real cluster, NEITHER SIGTERM NOR SIGKILL of ``sbatch --wait`` ends
    the job. Cancel BY ID: task names are not unique across concurrent runs, so a
    ``scancel --name`` reap would kill a different run's workers."""
    _fake_sbatch(tmp_path, 'echo "Submitted batch job 4242"; exit 0')
    workers.SlurmWorker(queue="q").run(["sbatch", "--wait"])
    assert (tmp_path / "scancel.log").read_text().split() == ["4242"]


def test_run_cancels_the_job_even_when_the_wait_raises(cluster_shims, tmp_path, monkeypatch):
    """The path that matters: an exception or Ctrl-C mid-run must not strand the job."""
    import subprocess as sp

    _fake_sbatch(tmp_path, 'echo "Submitted batch job 99"; sleep 30')

    class _Boom(sp.Popen):
        def wait(self, *a, **k):
            raise KeyboardInterrupt

    monkeypatch.setattr(workers.sp, "Popen", _Boom)
    with pytest.raises(KeyboardInterrupt):
        workers.SlurmWorker(queue="q").run(["sbatch", "--wait"])
    assert (tmp_path / "scancel.log").read_text().split() == ["99"], "job was stranded"


def test_unparseable_sbatch_output_warns_rather_than_crashing(cluster_shims, tmp_path, caplog):
    """If the job id cannot be read the worker still runs -- but say so loudly, because
    that worker can no longer be cancelled and will run to its time limit."""
    _fake_sbatch(tmp_path, 'echo "something unexpected"; exit 0')
    with caplog.at_level("WARNING"):
        assert workers.SlurmWorker(queue="q").run(["sbatch", "--wait"]).returncode == 0
    assert "could not parse a slurm job id" in caplog.text
    assert not (tmp_path / "scancel.log").exists()


def test_worker_run_default_is_plain_subprocess_run(cluster_shims):
    """Non-scheduler backends keep the old behaviour: no job id, no teardown."""
    assert workers.LocalWorker().run(["true"]).returncode == 0
