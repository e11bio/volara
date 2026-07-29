import os
from pathlib import Path

import daisy

# The log basedir must survive a process boundary. Workers are spawned as fresh `volara-cli`
# PROCESSES that re-import this module, so a module-global alone silently reverts them to the
# default while the driver uses something else -- and since BlockwiseTask.meta_dir lives under it,
# driver and worker then read/write DIFFERENT blocks_done stores. The worker's `open_ds(..., "a")`
# on its nonexistent path creates a zarr GROUP, so every block then fails with "Expected a zarr
# Array ... got Group" and the run orphans every block with no driver-visible cause. Backing the
# value with an env var makes subprocesses inherit it.
LOG_BASEDIR_ENV = "VOLARA_LOG_BASEDIR"

# default log dir (env wins, so a spawned worker matches the driver that set it)
LOG_BASEDIR = Path(os.environ.get(LOG_BASEDIR_ENV, "./volara_logs"))
daisy.logging.set_log_basedir(LOG_BASEDIR)


def set_log_basedir(path: Path | str):
    """Set the base directory for logging (indivudal worker logs and detailed
    task summaries). If set to ``None``, all logging will be shown on the
    command line (which can get very messy).

    Default is ``./volara_logs``.
    """
    path = Path(path)

    global LOG_BASEDIR

    if path is not None:
        LOG_BASEDIR = Path(path)
        # Export so spawned workers (fresh processes) resolve the SAME basedir as this driver.
        os.environ[LOG_BASEDIR_ENV] = str(LOG_BASEDIR)
    else:
        raise NotImplementedError("None is not a valid log directory")
        LOG_BASEDIR = None

    daisy.logging.set_log_basedir(LOG_BASEDIR)


def get_log_basedir():
    """Get the base directory for logging (indivudal worker logs and detailed
    task summaries).

    Default is ``./volara_logs``.
    """
    global LOG_BASEDIR
    return LOG_BASEDIR
