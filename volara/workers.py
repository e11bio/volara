import logging
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
        )

    def is_srun_available(self) -> bool:
        try:
            _result = sp.run(
                ["srun", "--version"], capture_output=True, text=True, check=True
            )
            # successful, return True
            return True
        except sp.CalledProcessError as e:
            # errors in the subprocess
            raise RuntimeError(f"srun failed to execute: {e}") from e
        except FileNotFoundError:
            # srun is not found in the system's PATH
            raise EnvironmentError(
                "srun is not installed or not in PATH. Either install slurm on your cluster, or run locally."
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
        Build the ``srun`` line that runs one worker as a slurm job.

        ``srun`` rather than ``sbatch``: it submits AND blocks for the job's lifetime,
        which is daisy v2's spawn contract (MIGRATION.md, "blocking-spawn contract") --
        ``sbatch`` returns at queue time, so daisy saw every worker as instantly dead,
        could never reap the real job (terminating the long-gone sbatch client), and
        leaked workers until walltime. ``srun`` also ties the job to the client:
        cancelling/killing the driver cancels the job steps with it, so worker jobs
        cannot outlive the run. Note that ``srun`` inside an existing slurm allocation
        starts a step WITHIN that allocation rather than submitting a new job --
        drivers that themselves run as slurm jobs must request resources for their
        workers up front.

        Args:
            command (list[str]): The worker command to run inside the slurm job.
            command (list[str]): The worker command to run inside the slurm job.
            num_cpus (int, optional): Number of CPU cores per task. Defaults to 1.
            num_gpus (int, optional): Number of GPUs required. Defaults to 0.
            memory (int, optional): Memory allocation (in MB) for the job. Defaults
                to 15564.
                to 15564.
            constraint (str, optional): Constraint specification for job
                execution. Defaults to "".
            queue (str, optional): Name of the Slurm partition (queue) to submit the
                job. Defaults to "".
            job_name (str, optional): Name assigned to the Slurm job. Defaults to "".
            log_file (str | Path | None, optional): Path for standard output logging.
            log_file (str | Path | None, optional): Path for standard output logging.
                Defaults to None.
            error_file (str | Path | None, optional): Path for standard error logging.
            error_file (str | Path | None, optional): Path for standard error logging.
                Defaults to None.
            flags (list[str] | None, optional): Additional srun flags as a
            flags (list[str] | None, optional): Additional srun flags as a
                list. Defaults to None.

        Returns:
            list[str]: The srun command, ready for ``subprocess.run``.
        """

        # TODO: raises exception on failure. Maybe handle this gracefully?
        self.is_srun_available()

        run_command = ["srun"]

        if job_name:
            run_command.append(f"--job-name={job_name}")
        self.is_srun_available()

        run_command = ["srun"]

        if job_name:
            run_command.append(f"--job-name={job_name}")
        run_command.append(f"--cpus-per-task={num_cpus}")
        if num_gpus > 0:
            run_command.append(f"--gpus={num_gpus}")
        if num_gpus > 0:
            run_command.append(f"--gpus={num_gpus}")
        run_command.append(f"--mem={memory}")
        if queue:
            run_command.append(f"--partition={queue}")
        if constraint and constraint != "None":
            run_command.append(f"--constraint={constraint}")
        if constraint and constraint != "None":
            run_command.append(f"--constraint={constraint}")

        run_command.append(f"--output={log_file}" if log_file else "--output=%x_%j.log")
        run_command.append(
            f"--error={error_file}" if error_file else "--error=%x_%j.err"
        )

        run_command.append(f"--output={log_file}" if log_file else "--output=%x_%j.log")
        run_command.append(
            f"--error={error_file}" if error_file else "--error=%x_%j.err"
        )

        if flags:
            run_command.extend(flags)

        # srun takes the worker command directly (no sbatch-style --wrap)
        run_command.extend(command)
        # srun takes the worker command directly (no sbatch-style --wrap)
        run_command.extend(command)

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
        job_name: str = "",
        log_file: str | None = None,
        error_file: str | None = None,
    ) -> list[str]:
        """
        Build the ``bsub -K`` line that runs one worker as an LSF job.

        ``-K`` makes bsub submit AND block until the job finishes, which is daisy v2's
        spawn contract (MIGRATION.md, "blocking-spawn contract") -- a plain ``bsub``
        returns at queue time, so daisy saw every worker as instantly dead and the
        real job leaked until walltime (same fire-and-forget bug as slurm's
        ``sbatch``, fixed there with ``srun``).

        Args:
            command (list[str]): The command to be executed within the LSF job.
            command (list[str]): The command to be executed within the LSF job.
            num_cpus (int, optional): Number of CPU cores per task. Defaults to 1.
            num_gpus (int, optional): Number of GPUs required. Defaults to 0.
            queue (str, optional): Name of the LSF queue to submit the job.
                Defaults to "".
            job_name (str, optional): Name assigned to the LSF job. Defaults to "".
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
        # -K: submit and wait for the job to complete (the blocking-spawn contract)
        run_command = ["bsub", "-K"]

        if job_name:
            run_command.extend(["-J", job_name])
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
