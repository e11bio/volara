import logging
import re
import shlex
import subprocess as sp
from abc import ABC
from pathlib import Path

import daisy

from .utils import StrictBaseModel

logger = logging.getLogger(__name__)


class Worker(StrictBaseModel, ABC):
    queue: str | None = None
    num_gpus: int = 0
    num_cpus: int = 1

    def get_command(self, config_path: Path, task_name: str) -> list[str]:
        cmd = [
            "volara-cli",
            "blockwise-worker",
            "-c",
            str(config_path),
        ]
        return cmd

    def run(self, cmd: list[str]) -> sp.CompletedProcess:
        """Execute one worker submission and BLOCK until that worker exits.

        daisy v2 runs the spawn function in a thread and reads its return as the
        worker's death (MIGRATION.md, "blocking-spawn contract"), so this must not
        return early. Backends override it when submitting and owning the process are
        not the same thing: a scheduler job outlives the client that submitted it, so
        it must be cleaned up explicitly (see :meth:`SlurmWorker.run`)."""
        return sp.run(cmd)


class SlurmWorker(Worker):
    queue: str
    num_gpus: int = 0
    num_cpus: int = 1

    time_limit: str | None = None
    """Walltime for each worker job (``--time``), e.g. ``"4:00:00"``.

    Strongly recommended. A worker is an independent slurm job, so it outlives its
    submitting client; :meth:`run` cancels it on every ordinary exit path, but a
    SIGKILLed driver runs no cleanup at all and then this is the ONLY bound. Do not
    assume the cluster provides one -- a partition may be ``DefaultTime=NONE`` /
    ``MaxTime=UNLIMITED``, in which case an escaped worker runs forever.
    """

    def get_command(self, config_path: Path, task_name: str) -> list[str]:
        cmd = super().get_command(config_path, task_name)

        context = daisy.Context.from_env()
        worker_id = context["worker_id"]
        task_id = context["task_id"]

        worker_log_basename = daisy.logging.get_worker_log_basename(worker_id, task_id)

        log_file = worker_log_basename / "slurm_worker.log"
        log_error = worker_log_basename / "slurm_worker.err"

        return self.get_slurm_command(
            command=cmd,
            job_name=task_name,  # name the worker after its task (identifiable in squeue)
            queue=self.queue,
            num_gpus=self.num_gpus,
            num_cpus=self.num_cpus,
            log_file=log_file,
            error_file=log_error,
            time_limit=self.time_limit,
        )

    _JOB_ID = re.compile(r"Submitted batch job (\d+)")

    def run(self, cmd: list[str]) -> sp.CompletedProcess:
        """Submit one worker with ``sbatch --wait`` and block until the job ends,
        cancelling it on the way out.

        ``--wait`` gives the blocking-spawn contract natively -- it returns only when
        the job terminates, and with the job's own exit status -- so no polling wrapper
        is needed. What it does NOT give is any tie between the job and this client:
        an independent slurm job survives its submitter, and MEASURED, neither SIGTERM
        nor SIGKILL of ``sbatch --wait`` cancels it. So the job id is read off sbatch's
        first line of stdout and ``scancel``-ed in ``finally`` -- on normal exit (a
        no-op, the job is already gone), on exception, and on ``KeyboardInterrupt``.

        Cancelling BY ID, not by ``--name``: task names are not unique across
        concurrent runs, so a name-based reap would cancel another run's workers.

        The one path this cannot cover is a SIGKILLed driver, which runs no cleanup at
        all; :attr:`time_limit` is the bound there."""
        job_id: str | None = None
        proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=None, text=True)
        try:
            # sbatch prints "Submitted batch job N" immediately, then --wait holds the
            # pipe open for the job's lifetime -- so read the one line, do NOT drain.
            first = proc.stdout.readline() if proc.stdout else ""
            match = self._JOB_ID.search(first or "")
            if match:
                job_id = match.group(1)
                logger.debug("submitted slurm worker job %s", job_id)
            else:
                logger.warning(
                    "could not parse a slurm job id from sbatch output %r; this worker "
                    "cannot be cancelled on teardown and will run until its time limit",
                    first,
                )
            returncode = proc.wait()
            return sp.CompletedProcess(cmd, returncode, stdout=first)
        finally:
            if proc.poll() is None:
                proc.kill()
            if job_id is not None:
                # Already-finished jobs are the normal case; scancel is a no-op there.
                sp.run(["scancel", job_id], capture_output=True, check=False)

    def is_sbatch_available(self) -> bool:
        try:
            _result = sp.run(
                ["sbatch", "--version"], capture_output=True, text=True, check=True
            )
            # successful, return True
            return True
        except sp.CalledProcessError as e:
            # errors in the subprocess
            raise RuntimeError(f"sbatch failed to execute: {e}") from e
        except FileNotFoundError:
            # sbatch is not found in the system's PATH
            raise EnvironmentError(
                "sbatch is not installed or not in PATH. Either install slurm on your cluster, or run locally."
            )

    def get_slurm_command(
        self,
        command: list[str],
        num_cpus: int = 1,
        num_gpus: int = 0,
        memory: int = 15564,
        constraint: str = "",
        queue: str = "",
        job_name: str = "",
        log_file: str | Path | None = None,
        error_file: str | Path | None = None,
        flags: list[str] | None = None,
        time_limit: str | None = None,
    ) -> list[str]:
        """
        Build the ``sbatch --wait`` line that runs one worker as its own slurm job.

        ``--wait`` is what makes this satisfy daisy v2's spawn contract (MIGRATION.md,
        "blocking-spawn contract"): it returns only when the job terminates, and with
        the job's exit status. A BARE ``sbatch`` returns at queue time, which is the
        old bug -- daisy saw every worker as instantly dead, respawned, and leaked the
        real job. ``--wait`` fixes that without a polling wrapper.

        Why not ``srun``: ``srun`` blocks too, and additionally ties the job to the
        client. But inside an existing allocation it starts a job STEP rather than
        submitting a new job, so a driver that is itself a slurm job -- the normal way
        to obtain a GPU node -- can never place workers anywhere but its own
        allocation, and ``queue``/``--partition`` is silently ignored for them.
        Measured: a 32-CPU driver requesting 32 workers got ONE step of 8 task clones
        sharing a single ``DAISY_CONTEXT``, and 623 "step creation still disabled"
        retries from the workers that never started. That is not a tuning problem, it
        is the whole fan-out, so ``sbatch --wait`` is the only slurm path.

        The tie to the client is replaced by explicit cleanup: :meth:`SlurmWorker.run`
        cancels the job by id on every ordinary exit path, and :attr:`time_limit`
        bounds the one path it cannot cover (a SIGKILLed driver).

        Args:
            command (list[str]): The worker command to run inside the slurm job.
            num_cpus (int, optional): Number of CPU cores per task. Defaults to 1.
            num_gpus (int, optional): Number of GPUs required. Defaults to 0.
            memory (int, optional): Memory allocation (in MB) for the job. Defaults
                to 15564.
            constraint (str, optional): Constraint specification for job
                execution. Defaults to "".
            queue (str, optional): Name of the Slurm partition (queue) to submit the
                job. Defaults to "".
            job_name (str, optional): Name assigned to the Slurm job. Defaults to "".
            log_file (str | Path | None, optional): Path for standard output logging.
                Defaults to None.
            error_file (str | Path | None, optional): Path for standard error logging.
                Defaults to None.
            flags (list[str] | None, optional): Additional sbatch flags as a
                list. Defaults to None.
            time_limit (str | None, optional): Walltime per worker job (``--time``),
                e.g. "4:00:00". Defaults to None (no limit requested -- see
                :attr:`SlurmWorker.time_limit` for why you want one).

        Returns:
            list[str]: The sbatch command, ready for :meth:`SlurmWorker.run`.
        """

        # TODO: raises exception on failure. Maybe handle this gracefully?
        self.is_sbatch_available()

        run_command = ["sbatch", "--wait"]

        if job_name:
            run_command.append(f"--job-name={job_name}")
        # ONE task per worker. Without it slurm derives ntasks from the allocation,
        # which made a single submission expand into N identical clones sharing one
        # DAISY_CONTEXT (hence one worker_id -- the race daisy warns about).
        run_command.append("--ntasks=1")
        run_command.append(f"--cpus-per-task={num_cpus}")
        if num_gpus > 0:
            run_command.append(f"--gpus={num_gpus}")
        run_command.append(f"--mem={memory}")
        if queue:
            run_command.append(f"--partition={queue}")
        if constraint and constraint != "None":
            run_command.append(f"--constraint={constraint}")

        run_command.append(f"--output={log_file}" if log_file else "--output=%x_%j.log")
        run_command.append(
            f"--error={error_file}" if error_file else "--error=%x_%j.err"
        )

        if time_limit:
            run_command.append(f"--time={time_limit}")

        if flags:
            run_command.extend(flags)

        # sbatch runs a batch SCRIPT, so the worker command is handed over as one
        # shell string. shlex.join keeps arguments containing spaces intact.
        run_command.append(f"--wrap={shlex.join(command)}")

        return run_command


class LSFWorker(Worker):
    queue: str
    num_gpus: int = 0
    num_cpus: int = 1

    def get_command(self, config_path: Path, task_name: str) -> list[str]:
        cmd = super().get_command(config_path, task_name)

        context = daisy.Context.from_env()
        worker_id = context["worker_id"]
        task_id = context["task_id"]

        worker_log_basename = daisy.logging.get_worker_log_basename(worker_id, task_id)
        if not worker_log_basename.exists():
            worker_log_basename.mkdir(parents=True, exist_ok=True)

        log_file = worker_log_basename / "lsf_worker.log"
        log_error = worker_log_basename / "lsf_worker.err"

        return self.get_lsf_command(
            command=cmd,
            job_name=task_name,  # name the worker after its task (identifiable in bjobs)
            queue=self.queue,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            log_file=log_file,
            error_file=log_error,
        )

    def is_bsub_available(self) -> bool:
        try:
            _result = sp.run(["bsub", "-V"], capture_output=True, text=True, check=True)
            # successful, return True
            return True
        except sp.CalledProcessError as e:
            # errors in the subprocess
            raise RuntimeError(f"bsub failed to execute: {e}") from e
        except FileNotFoundError:
            # bsub is not found in the system's PATH
            raise EnvironmentError(
                "bsub is not installed or not in PATH. Either install bsub on your cluster, or run locally."
            )

    def get_lsf_command(
        self,
        command: list[str],
        num_cpus: int = 1,
        num_gpus: int = 0,
        queue: str = "",
        job_name: str = "",
        log_file: str | None = None,
        error_file: str | None = None,
    ) -> list[str]:
        """
        Build the ``bsub -K`` line that runs one worker as an LSF job.

        ``-K`` makes bsub submit AND block until the job finishes, which is daisy v2's
        spawn contract (MIGRATION.md, "blocking-spawn contract") -- a plain ``bsub``
        returns at queue time, so daisy saw every worker as instantly dead and the
        real job leaked until walltime (same fire-and-forget bug as a bare slurm
        ``sbatch``, fixed there with ``sbatch --wait``).

        Args:
            command (list[str]): The command to be executed within the LSF job.
            num_cpus (int, optional): Number of CPU cores per task. Defaults to 1.
            num_gpus (int, optional): Number of GPUs required. Defaults to 0.
            queue (str, optional): Name of the LSF queue to submit the job.
                Defaults to "".
            job_name (str, optional): Name assigned to the LSF job. Defaults to "".
            log_file (str | None, optional): Path for standard output logging.
                Defaults to None.
            error_file (str | None, optional): Path for standard error logging.
                Defaults to None.
        """
        self.is_bsub_available()

        log = ["-o", str(log_file)] if log_file is not None else []
        error = ["-e", str(error_file)] if error_file is not None else []

        # -K: submit and wait for the job to complete (the blocking-spawn contract)
        run_command = ["bsub", "-K"]

        if job_name:
            run_command.extend(["-J", job_name])
        run_command.extend(["-n", str(num_cpus)])
        if num_gpus > 0:
            run_command.extend(["-num-gpus", str(num_gpus)])
        if queue:
            run_command.extend(["-q", str(queue)])

        run_command.extend(log)
        run_command.extend(error)

        run_command += command

        return run_command


class LocalWorker(Worker):
    def get_command(self, config_path: Path, task_name: str) -> list[str]:
        cmd = super().get_command(config_path, task_name)

        context = daisy.Context.from_env()
        worker_id = context["worker_id"]
        task_id = context["task_id"]

        worker_log_basename = daisy.logging.get_worker_log_basename(worker_id, task_id)

        _log_file = worker_log_basename / "out.log"
        _log_error = worker_log_basename / "out.err"

        # TODO: update command to use log files, test that they exist
        # current tests only show that "worker_id" and "task_id" can be retrieved
        return cmd
