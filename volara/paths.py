"""One path type for volara: ``upath.UPath``.

Every path-valued config field (``Dataset.store``, ``LUT.path``, ``SQLite.path``,
``OmeNormalize.metadata``, ...) is a :data:`PydanticUPath`. A ``str``, a
``pathlib.Path`` or a ``UPath`` is accepted and becomes a ``UPath``; a local
one is a ``pathlib.Path`` subclass, a remote one (``s3://``, ``gs://``,
``memory://``, ...) carries its protocol and an fsspec filesystem. It
serialises as the string it was built from, so saved configs are unchanged.

Two consumers cannot use a remote location: daisy (worker log files, the
mmap'd block-tracking store) and SQLite. Both go through :func:`local_path`,
the one explicit cast from a ``UPath`` to a local ``pathlib.Path``: a URL is
mapped to a deterministic local directory under :func:`get_remote_root` and
a ``UserWarning`` names the mapping, rather than ``pathlib.Path("s3://b/x")``
silently becoming the relative directory ``s3:/b/x``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote, urlsplit

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from upath import UPath

LOCAL_PROTOCOLS = ("", "file", "local")


def is_local(path: UPath | Path) -> bool:
    """Whether ``path`` is on the local filesystem (a plain ``Path`` or a local ``UPath``)."""
    return getattr(path, "protocol", "") in LOCAL_PROTOCOLS


def to_upath(value: Any, **storage_options) -> UPath:
    """Coerce a ``str``, ``os.PathLike`` or ``UPath`` to a ``UPath``.

    ``storage_options`` are the fsspec filesystem options (credentials, a custom
    endpoint, ...). Given any, the path is rebuilt with them, replacing what the
    value carried.
    """
    if isinstance(value, UPath) and not storage_options:
        return value
    if isinstance(value, (UPath, str, os.PathLike)):
        return UPath(
            os.fspath(value) if isinstance(value, Path) else str(value),
            **storage_options,
        )
    raise TypeError(f"expected a path or URL, got {type(value).__name__}: {value!r}")


class _UPathSchema:
    """Pydantic schema for a ``UPath`` field: validates from ``str``/``Path``/``UPath``
    (JSON: a string), serialises as ``str``. Replaces universal-pathlib's own
    schema, whose JSON form is a ``{protocol, path, storage_options}`` object."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        from_any = core_schema.no_info_plain_validator_function(to_upath)
        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema([core_schema.str_schema(), from_any]),
            python_schema=from_any,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "string"}


PydanticUPath = Annotated[UPath, _UPathSchema]
"""A ``UPath`` config field: validates from ``str``/``Path``/``UPath``, serialises as ``str``."""


def zarr_store(path: UPath | Path):
    """The store object zarr accepts for ``path``: the path itself when local,
    an fsspec-backed ``FsspecStore`` otherwise (zarr does not take a remote
    ``UPath`` object, and a bare URL string would drop the storage options)."""
    if is_local(path):
        return Path(str(path))
    assert isinstance(path, UPath)
    from zarr.storage import FsspecStore

    return FsspecStore(fs=_async_fs(path.fs), path=path.path.rstrip("/"))


def _async_fs(fs):
    """``fs`` in the asynchronous form zarr's ``FsspecStore`` requires.

    A ``UPath`` hands back its filesystem in synchronous mode. An async-capable
    filesystem (s3fs, gcsfs, ...) is re-created in async mode with the same
    storage options; a genuinely synchronous one (memory, local, sftp, ...) is
    wrapped. zarr >= 3.2 does this inside ``FsspecStore.from_upath``; this is
    the same logic so that zarr 3.1 (the newest on Python 3.11) works too.
    """
    if getattr(fs, "async_impl", False):
        if getattr(fs, "asynchronous", False):
            return fs
        return type(fs)(
            *fs.storage_args, **{**fs.storage_options, "asynchronous": True}
        )
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    return AsyncFileSystemWrapper(fs, asynchronous=True)


def rm(path: UPath | Path, recursive: bool = True) -> None:
    """Delete ``path`` wherever it lives; a missing path is a no-op."""
    if is_local(path):
        from shutil import rmtree

        p = Path(str(path))
        if p.is_dir() and not p.is_symlink():
            rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()
        return
    assert isinstance(path, UPath)
    try:
        path.fs.rm(path.path, recursive=recursive)
    except FileNotFoundError:
        pass
    invalidate = getattr(path.fs, "invalidate_cache", None)
    if invalidate is not None:
        invalidate(path.path)


# --- the cast to a local path ----------------------------------------------

# Where a URL that a local-only consumer receives is mapped on the local
# filesystem. Resolved against the CWD at the moment of mapping.
_REMOTE_ROOT = Path("./volara_logs")
_warned_remote: set[str] = set()


def set_remote_root(path: Path | str) -> Path:
    """Set the local directory under which :func:`local_path` maps URLs.
    Default is ``./volara_logs``. Returns the new root."""
    global _REMOTE_ROOT
    _REMOTE_ROOT = Path(os.fspath(path))
    return _REMOTE_ROOT


def get_remote_root() -> Path:
    return _REMOTE_ROOT


def local_path(
    path: Any, what: str = "volara's worker logs and block tracking"
) -> Path:
    """The one explicit cast from a path volara holds to a local ``pathlib.Path``.

    ``what`` names the consumer in the warning. A local ``str``, ``Path`` or
    ``UPath`` is used as given, relative or not. A URL, as a string or as a
    remote ``UPath``, maps to ``<remote root>/<protocol>/<path>``, absolute,
    and a ``UserWarning`` names the mapping once per URL; ``file://`` URLs
    become the local path they name. ``None`` raises ``TypeError``.

    Without this, ``pathlib.Path("s3://bucket/x")`` silently yields the
    RELATIVE directory ``s3:/bucket/x``, and a driver and a spawned worker
    launched from different directories disagree about where it is.
    """
    if path is None:
        raise TypeError(f"None is not a valid path for {what}")
    if isinstance(path, os.PathLike) and is_local(path):
        return Path(os.fspath(path))
    s = str(path)
    if "://" not in s:
        return Path(s)
    parts = urlsplit(s)
    if parts.scheme in ("file", "local"):
        return Path(unquote(parts.netloc + parts.path))
    rest = unquote((parts.netloc + parts.path).strip("/"))
    target = (get_remote_root() / parts.scheme / rest).resolve()
    if s not in _warned_remote:
        _warned_remote.add(s)
        warnings.warn(
            f"{what} live on the local filesystem only; {s!r} is mapped to "
            f"{str(target)!r} (volara.paths.set_remote_root changes the root).",
            UserWarning,
            stacklevel=3,
        )
    return target
