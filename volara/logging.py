from pathlib import Path

import daisy

from .paths import get_remote_root, local_path, set_remote_root  # noqa: F401 -- re-exported

# default log dir
LOG_BASEDIR = Path("./volara_logs")
daisy.logging.set_log_basedir(LOG_BASEDIR)


def set_log_basedir(path) -> Path:
    """Set the base directory for logging (indivudal worker logs and detailed
    task summaries). ``None`` is not a valid log directory and raises
    ``TypeError``.

    This is the one path volara hands to daisy, and everything derived from it
    (``meta_dir``, ``config_file``, ``tracking_path``) is local by contract:
    daisy's worker ``.out``/``.err`` files and its mmap'd block-tracking store
    cannot live in a bucket. So the value is cast with :func:`volara.paths.local_path`:
    a local path is used as given; a URL or remote ``UPath`` is mapped to a local
    directory under :func:`volara.paths.get_remote_root` with a warning. The local
    path is returned, and it is what daisy carries to every spawned worker.

    Default is ``./volara_logs``.
    """
    path = local_path(path)

    global LOG_BASEDIR

    LOG_BASEDIR = path

    daisy.logging.set_log_basedir(LOG_BASEDIR)
    return LOG_BASEDIR


def get_log_basedir() -> Path:
    """Get the base directory for logging (indivudal worker logs and detailed
    task summaries).

    Default is ``./volara_logs``.
    """
    global LOG_BASEDIR
    return LOG_BASEDIR
