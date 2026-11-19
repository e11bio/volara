"""
Node attributes stored as plain arrays beside a graph.

A graph model writes one vector per node. Those vectors are too wide to be
columns of the graph they describe and are written once, so they live in a
directory of ``.npy`` files keyed by node id::

    <path>/meta.json        {"version": 1, "num_nodes": N,
                             "attrs": {name: {"dtype": "<f4", "size": 0 | k}},
                             "graph_digest": str | null, "source": str | null}
    <path>/ids.npy          uint64 (N,), node ids in stored order
    <path>/order.npy        int64 (N,), stable argsort of ids
    <path>/node_<attr>.npy  (N,) for size 0, (N, k) for size k

Rows of every ``node_<attr>.npy`` are aligned with ``ids.npy``. ``order.npy``
lets a reader find an id by binary search without sorting the store.
"""

import json
import os
import shutil
import struct
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .paths import PydanticUPath, local_path
from .utils import StrictBaseModel

FORMAT_VERSION = 1
ID_DTYPE = np.dtype("<u8")
ORDER_DTYPE = np.dtype("<i8")

# Every array file starts with a header of this many bytes, padded with
# spaces as the npy format allows, so the shape can be rewritten in place
# once the number of rows is known.
_HEADER_BYTES = 128
_MAGIC = b"\x93NUMPY\x01\x00"


def _header(dtype: np.dtype, shape: tuple[int, ...]) -> bytes:
    described = repr(
        {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        }
    )
    room = _HEADER_BYTES - len(_MAGIC) - 2 - 1
    if len(described) > room:
        raise ValueError(f"array header does not fit in {_HEADER_BYTES} bytes")
    text = described + " " * (room - len(described)) + "\n"
    return _MAGIC + struct.pack("<H", len(text)) + text.encode("latin1")


def _missing(dtype: np.dtype) -> Any:
    return np.nan if dtype.kind == "f" else 0


class NodeArraysWriter:
    """
    Appends nodes to a :class:`NodeArrays` store that does not exist yet.

    Use it as a context manager. Nodes are appended to a scratch directory
    beside the target; leaving the block without an error builds the id
    index and moves the directory into place, so a reader never sees a store
    that is half written. Leaving it with an error removes the scratch
    directory and leaves any earlier store untouched.
    """

    def __init__(
        self,
        path: Path,
        attrs: Mapping[str, tuple[Any, int]],
        graph_digest: str | None = None,
        source: str | None = None,
    ):
        self.path = path
        self.attrs = {
            name: (np.dtype(dtype).newbyteorder("<"), int(size))
            for name, (dtype, size) in attrs.items()
        }
        for name, (dtype, size) in self.attrs.items():
            if dtype.kind not in "fiub":
                raise ValueError(f"attribute {name!r} has a non-numeric dtype {dtype}")
            if size < 0:
                raise ValueError(f"attribute {name!r} has a negative size")
        self.graph_digest = graph_digest
        self.source = source
        self.num_nodes = 0
        self._scratch = path.with_name(f"{path.name}.writing-{uuid.uuid4().hex}")
        self._files: dict[str, Any] = {}

    def _shape(self, name: str, rows: int) -> tuple[int, ...]:
        size = self.attrs[name][1]
        return (rows, size) if size else (rows,)

    def __enter__(self) -> "NodeArraysWriter":
        self._scratch.mkdir(parents=True)
        self._files["ids"] = (self._scratch / "ids.npy").open("wb")
        self._files["ids"].write(_header(ID_DTYPE, (0,)))
        for name, (dtype, _) in self.attrs.items():
            handle = (self._scratch / f"node_{name}.npy").open("wb")
            handle.write(_header(dtype, self._shape(name, 0)))
            self._files[name] = handle
        return self

    def append(self, ids: np.ndarray, **arrays: np.ndarray) -> None:
        """Append nodes: their ids and one array per attribute of the store,
        each with one row per id."""
        if set(arrays) != set(self.attrs):
            raise ValueError(
                f"expected the attributes {sorted(self.attrs)}, got {sorted(arrays)}"
            )
        ids = np.ascontiguousarray(ids, dtype=ID_DTYPE).reshape(-1)
        for name, values in arrays.items():
            dtype, _ = self.attrs[name]
            values = np.ascontiguousarray(values, dtype=dtype)
            if values.shape != self._shape(name, len(ids)):
                raise ValueError(
                    f"attribute {name!r} has shape {values.shape}, "
                    f"expected {self._shape(name, len(ids))}"
                )
            self._files[name].write(values.tobytes())
        self._files["ids"].write(ids.tobytes())
        self.num_nodes += len(ids)

    def _finish(self) -> None:
        for name, handle in self._files.items():
            dtype = ID_DTYPE if name == "ids" else self.attrs[name][0]
            shape = (
                (self.num_nodes,)
                if name == "ids"
                else self._shape(name, self.num_nodes)
            )
            handle.seek(0)
            handle.write(_header(dtype, shape))
            handle.close()
        ids = np.load(self._scratch / "ids.npy", mmap_mode="r")
        order = np.argsort(ids, kind="stable").astype(ORDER_DTYPE)
        in_order = ids[order]
        if len(in_order) > 1 and (in_order[1:] == in_order[:-1]).any():
            raise ValueError("a node id was written more than once")
        del ids
        with (self._scratch / "order.npy").open("wb") as handle:
            handle.write(_header(ORDER_DTYPE, (self.num_nodes,)))
            handle.write(order.tobytes())
        meta = {
            "version": FORMAT_VERSION,
            "num_nodes": self.num_nodes,
            "attrs": {
                name: {"dtype": dtype.str, "size": size}
                for name, (dtype, size) in self.attrs.items()
            },
            "graph_digest": self.graph_digest,
            "source": self.source,
        }
        (self._scratch / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        if self.path.exists():
            shutil.rmtree(self.path)
        os.replace(self._scratch, self.path)

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc_type is None:
                self._finish()
        finally:
            for handle in self._files.values():
                handle.close()
            if self._scratch.exists():
                shutil.rmtree(self._scratch)


class NodeArraysReader:
    """
    A :class:`NodeArrays` store opened for reading. The arrays are
    memory-mapped, so opening costs nothing and a handle per worker shares
    the page cache with every other worker.
    """

    def __init__(self, path: Path):
        self.path = path
        if not (path / "meta.json").exists():
            raise FileNotFoundError(f"no node arrays at {path}")
        self.meta: dict[str, Any] = json.loads((path / "meta.json").read_text())
        if self.meta["version"] != FORMAT_VERSION:
            raise RuntimeError(
                f"{path} is node array format version {self.meta['version']}, "
                f"this library reads version {FORMAT_VERSION}"
            )
        self.ids: np.ndarray = np.load(path / "ids.npy", mmap_mode="r")
        self.order: np.ndarray = np.load(path / "order.npy", mmap_mode="r")
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return int(self.meta["num_nodes"])

    @property
    def attrs(self) -> dict[str, dict[str, Any]]:
        """``{name: {"dtype": str, "size": 0 | k}}`` of the stored attributes."""
        return self.meta["attrs"]

    def array(self, attr: str) -> np.ndarray:
        """The memory-mapped array of one attribute, aligned with :attr:`ids`."""
        if attr not in self.attrs:
            raise KeyError(
                f"attribute {attr!r} is not in {self.path}; it has {sorted(self.attrs)}"
            )
        if attr not in self._arrays:
            self._arrays[attr] = np.load(self.path / f"node_{attr}.npy", mmap_mode="r")
        return self._arrays[attr]

    def rows(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(found, rows)`` for ``ids``: whether each id is stored, and its
        row (0 where it is not)."""
        ids = np.ascontiguousarray(ids, dtype=ID_DTYPE).reshape(-1)
        if len(self) == 0 or len(ids) == 0:
            return np.zeros(len(ids), dtype=bool), np.zeros(len(ids), dtype=np.int64)
        at = np.searchsorted(self.ids, ids, sorter=self.order)
        at[at == len(self)] = 0
        rows = np.asarray(self.order[at])
        found = np.asarray(self.ids[rows]) == ids
        rows[~found] = 0
        return found, rows

    def read(
        self, ids: np.ndarray, attrs: Sequence[str] | None = None
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        The attributes of the nodes ``ids``.

        Returns ``(found, values)``. ``found`` is a boolean mask over ``ids``.
        ``values`` has one array per attribute, aligned with ``ids``; a row of
        an id that is not stored is NaN for a float attribute and 0 otherwise.
        All attributes are returned when none are named.
        """
        names = list(self.attrs) if attrs is None else list(attrs)
        found, rows = self.rows(ids)
        hit = np.flatnonzero(found)
        # read the rows in file order, then put them back in the asked order
        by_row = hit[np.argsort(rows[hit], kind="stable")]
        values: dict[str, np.ndarray] = {}
        for name in names:
            stored = self.array(name)
            out = np.full(
                (len(found), *stored.shape[1:]),
                _missing(stored.dtype),
                dtype=stored.dtype,
            )
            out[by_row] = stored[rows[by_row]]
            values[name] = out
        return found, values


class NodeArrays(StrictBaseModel):
    """
    Node attributes stored as plain arrays beside a graph: one row per node,
    found by node id. This is where a model's per-node output (an embedding)
    is kept, and what :class:`~volara.blockwise.EdgeSimilarity` reads.

    The directory holds ``meta.json``, ``ids.npy`` (node ids in stored order),
    ``order.npy`` (the stable argsort of the ids) and one
    ``node_<attr>.npy`` per attribute, of shape ``(N,)`` or ``(N, k)``.

    A store is written once and never changed. The arrays are memory-mapped,
    so only local paths are supported.
    """

    path: PydanticUPath
    """
    The directory of the store. A URL here is cast to a local path under
    :func:`volara.paths.get_remote_root` with a warning (see
    :func:`volara.paths.local_path`).
    """

    @property
    def local(self) -> Path:
        """The local directory behind `path`."""
        return local_path(self.path, what="node arrays")

    @property
    def name(self) -> str:
        return self.local.stem

    def exists(self) -> bool:
        return (self.local / "meta.json").exists()

    def drop(self) -> None:
        if self.local.exists():
            shutil.rmtree(self.local)

    def writer(
        self,
        attrs: Mapping[str, tuple[Any, int]],
        graph_digest: str | None = None,
        source: str | None = None,
    ) -> NodeArraysWriter:
        """
        A writer that appends nodes block by block and puts the store in
        place when its ``with`` block ends.

        Args:
            attrs: ``{name: (dtype, size)}``; size 0 is one value per node,
                size k a vector of k values.
            graph_digest: the digest of the graph whose nodes these are.
            source: where the values came from, e.g. the id of a model.
        """
        return NodeArraysWriter(
            self.local, attrs, graph_digest=graph_digest, source=source
        )

    def save(
        self,
        ids: np.ndarray,
        arrays: Mapping[str, np.ndarray],
        graph_digest: str | None = None,
        source: str | None = None,
    ) -> None:
        """
        Write the whole store at once: ``ids`` and one array per attribute,
        each with one row per id. An existing store is replaced only when the
        new one is complete.
        """
        attrs = {}
        for name, values in arrays.items():
            values = np.asarray(values)
            if values.ndim not in (1, 2):
                raise ValueError(f"attribute {name!r} is not a (N,) or (N, k) array")
            attrs[name] = (values.dtype, values.shape[1] if values.ndim == 2 else 0)
        with self.writer(attrs, graph_digest=graph_digest, source=source) as writer:
            writer.append(ids, **dict(arrays))

    def open(self) -> NodeArraysReader:
        """Open the store for reading, memory-mapped."""
        return NodeArraysReader(self.local)

    def read(
        self, ids: np.ndarray, attrs: Sequence[str] | None = None
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """``(found, values)`` for the nodes ``ids``; see
        :meth:`NodeArraysReader.read`. Open the store once with :meth:`open`
        when reading many times."""
        return self.open().read(ids, attrs)
