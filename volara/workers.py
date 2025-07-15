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

        worker_log_basename = daisy.get_worker_log_basename(worker_id, task_id)

        log_file = worker_log_basename / "slurm_worker.log"
        log_error = worker_log_basename / "slurm_worker.err"

        return self.get_slurm_command(
            command=" ".join(cmd),
            execute=False,
            expand=False,
            queue=self.queue,
            num_gpus=self.num_gpus,
            num_cpus=self.num_cpus,
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
                "sbatch is not installed or not in PATH. Either install sbatch on your cluster, or run locally."
            )

    def get_slurm_command(
        self,
        command: str,
        num_cpus: int = 1,
        num_gpus: int = 0,
        memory: int = 15564,
        constraint: str = "",
        queue: str = "",
        execute: bool = False,
        expand: bool = True,
        job_name: str = "",
        array_size: int = 1,
        array_limit: int | None = None,
        log_file: str | None = None,
        error_file: str | None = None,
        flags: list[str] | None = None,
    ) -> list[str]:
        """
        Prepares and optionally executes a command on a slurm cluster,

        Args:
            command (str): The command to be executed within the Slurm job.
            num_cpus (int, optional): Number of CPU cores per task. Defaults to 1.
            num_gpus (int, optional): Number of GPUs required. Defaults to 0.
            memory (int, optional): Memory allocation (in MB) for the job. Defaults
                to 25600.
            constraint (str, optional): Constraint specification for job
                execution. Defaults to "".
            queue (str, optional): Name of the Slurm partition (queue) to submit the
                job. Defaults to "".
            execute (bool, optional): Whether to execute the command or just return
                the command. Defaults to False.
            expand (bool, optional): Returns a string if True, and a list if False.
                Defaults to True.
            job_name (str, optional): Name assigned to the Slurm job. Defaults to "".
            array_size (int, optional): If greater than 1, submits a job array of
                this size. Defaults to 1.
            array_limit (int | None, optional): Limits the number of
                simultaneously running tasks in the job array. Defaults to None.
            log_file (str | None, optional): Path for standard output logging.
                Defaults to None.
            error_file (str | None, optional): Path for standard error logging.
                Defaults to None.
            flags (list[str] | None, optional): Additional sbatch flags as a
                list. Defaults to None.

        Returns:
            str | list[str] | None: Depending on `execute` and `expand`,
                returns the job ID (if executed), the constructed command as a string or
                list, or None if an error occurred.
        """

        # TODO: raises exception on failure. Maybe handle this gracefully?
        self.is_sbatch_available()

        if execute:
            logging.info(
                f"Scheduling job on {num_cpus} CPUs, {num_gpus} GPUs with {memory} MB on queue {queue}"
            )

        log = f"--output={log_file}" if log_file else "--output=%x_%j.log"
        error = f"--error={error_file}" if error_file else "--error=%x_%j.err"

        use_gpus = f"--gpus={num_gpus}" if num_gpus > 0 else ""
        use_constraint = (
            f"--constraint={constraint}" if constraint and constraint != "None" else ""
        )
        job_name_cmd = f"--job-name={job_name}" if job_name else ""

        # Initialize the command list with the base sbatch command
        run_command = ["sbatch"]

        # Job array handling
        if array_size > 1:
            array_cmd = (
                f"--array=1-{array_size}{f'%{array_limit}' if array_limit else ''}"
            )
            run_command.append(array_cmd)

        # Append job name, CPU, GPU, memory, queue, and constraint constraints to the command
        if job_name_cmd:
            run_command.append(job_name_cmd)
        run_command.append(f"--cpus-per-task={num_cpus}")
        if use_gpus:
            run_command.append(use_gpus)
        run_command.append(f"--mem={memory}")
        if queue:
            run_command.append(f"--partition={queue}")
        if use_constraint:
            run_command.append(use_constraint)

        # Append log and error file paths to the command
        run_command.append(log)
        run_command.append(error)

        # Append additional flags if provided
        if flags:
            run_command.extend(flags)

        # Append the command to be executed within the job
        run_command.append(f"--wrap={command}")

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

        worker_log_basename = daisy.get_worker_log_basename(worker_id, task_id)
        if not worker_log_basename.exists():
            worker_log_basename.mkdir(parents=True, exist_ok=True)

        log_file = worker_log_basename / "lsf_worker.log"
        log_error = worker_log_basename / "lsf_worker.err"

        return self.get_lsf_command(
            command=cmd,
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
        log_file: str | None = None,
        error_file: str | None = None,
    ) -> list[str]:
        """
        Prepares and optionally executes a command on an LSF cluster,

        Args:
            command (str): The command to be executed within the LSF job.
            num_cpus (int, optional): Number of CPU cores per task. Defaults to 1.
            num_gpus (int, optional): Number of GPUs required. Defaults to 0.
            queue (str, optional): Name of the LSF queue to submit the job.
                Defaults to "".
            log_file (str | None, optional): Path for standard output logging.
                Defaults to None.
            error_file (str | None, optional): Path for standard error logging.
                Defaults to None.
        """
        self.is_bsub_available()

        log = ["-o", str(log_file)] if log_file is not None else []
        error = ["-e", str(error_file)] if error_file is not None else []

        run_command = ["bsub"]

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

        worker_log_basename = daisy.get_worker_log_basename(worker_id, task_id)

        _log_file = worker_log_basename / "out.log"
        _log_error = worker_log_basename / "out.err"

        # TODO: update command to use log files, test that they exist
        # current tests only show that "worker_id" and "task_id" can be retrieved
        return cmd
