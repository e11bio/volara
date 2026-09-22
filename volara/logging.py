import daisy
from upath import UPath

# default log dir
LOG_BASEDIR = UPath("./volara_logs")
# daisy coerces whatever it is given through ``pathlib.Path``, so hand it the
# unmangled URL string and let any mangling be daisy's alone.
daisy.logging.set_log_basedir(str(LOG_BASEDIR))


def set_log_basedir(path: UPath | str):
    """Set the base directory for logging (indivudal worker logs and detailed
    task summaries). ``None`` is not a valid log directory and raises
    ``TypeError``.

    A remote URL (``s3://bucket/logs``) survives here: ``UPath`` keeps the
    protocol, where ``pathlib.Path`` would collapse ``s3://`` to the relative
    local path ``s3:/``.

    Default is ``./volara_logs``.
    """
    # Note: this coercion is what rejects ``None`` - ``UPath(None)`` raises
    # ``TypeError``, so the value below is always a real path.
    path = UPath(path)

    global LOG_BASEDIR

    LOG_BASEDIR = path

    daisy.logging.set_log_basedir(str(LOG_BASEDIR))


def get_log_basedir() -> UPath:
    """Get the base directory for logging (indivudal worker logs and detailed
    task summaries).

    Default is ``./volara_logs``.
    """
    global LOG_BASEDIR
    return LOG_BASEDIR
