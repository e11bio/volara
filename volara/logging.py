from pathlib import Path

import daisy

# default log dir
LOG_BASEDIR = Path("./volara_logs")
daisy.logging.set_log_basedir(LOG_BASEDIR)


def set_log_basedir(path: Path):
    """Set the base directory for logging (indivudal worker logs and detailed
    task summaries). If set to ``None``, all logging will be shown on the
    command line (which can get very messy).

    Default is ``./volara_logs``.
    """

    global LOG_BASEDIR

    if path is not None:
        LOG_BASEDIR = Path(path)
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
