from pathlib import Path

import daisy

# default log dir
LOG_BASEDIR = Path("./volara_logs")
daisy.logging.set_log_basedir(LOG_BASEDIR)


def set_log_basedir(path: Path | str):
    """Set the base directory for logging (indivudal worker logs and detailed
    task summaries). ``None`` is not a valid log directory and raises
    ``TypeError``.

    Default is ``./volara_logs``.
    """
    # Note: this coercion is what rejects ``None`` - ``Path(None)`` raises
    # ``TypeError``, so the value below is always a real path.
    path = Path(path)

    global LOG_BASEDIR

    LOG_BASEDIR = path

    daisy.logging.set_log_basedir(LOG_BASEDIR)


def get_log_basedir():
    """Get the base directory for logging (indivudal worker logs and detailed
    task summaries).

    Default is ``./volara_logs``.
    """
    global LOG_BASEDIR
    return LOG_BASEDIR
