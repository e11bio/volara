import logging
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


class SlurmWorker(Worker):
    queue: str
    num_gpus: int = 0
    num_cpus: int = 1
    memory: int = 15564

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
            memory=self.memory,
            log_file=log_file,
            error_file=log_error,
        )

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

        KNOWN TRADE-OFF, accepted deliberately: what ``srun`` gave for free and this
        does not is any tie between the job and its client. A worker is now an
        independent slurm job, and MEASURED on a real cluster, neither ``SIGTERM`` nor
        ``SIGKILL`` of the ``sbatch --wait`` client ends it -- only ``scancel`` does.
        So a driver that dies abnormally leaves its workers running until the
        partition's walltime, and there is no knob here to bound that: ``flags`` is not
        threaded through :meth:`SlurmWorker.get_command`, so a configured worker cannot
        yet request ``--time``. On a partition with ``DefaultTime=NONE`` /
        ``MaxTime=UNLIMITED`` -- the ones measured here -- an escaped worker runs
        forever, and must be reaped with ``scancel``.

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

        Returns:
            list[str]: The sbatch command, ready for ``subprocess.run``.
        """

        # TODO: raises exception on failure. Maybe handle this gracefully?
        self.is_sbatch_available()

        run_command = ["sbatch", "--wait"]

        if job_name:
            run_command.append(f"--job-name={job_name}")
        # ONE task per worker, pinned explicitly. This is the failure the srun path
        # hit: as a step inside an allocation, ntasks came from the allocation, so a
        # single submission expanded into N identical clones sharing one
        # DAISY_CONTEXT (hence one worker_id -- the race daisy warns about). sbatch
        # defaults to 1, so this pins the invariant rather than fixing a live bug.
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
