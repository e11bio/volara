"""On-disk hierarchy state for :class:`GraphCluster`.

Layout::

    work_dir/
      manifest.json                 # roi, levels, weights, method, ... (must match)
      L0/<i>_<j>_<k>/
        lut.npz                     # in: block input node ids -> out: component ids
        graph.npz                   # contracted nodes + surviving signed edges
        boundary.npz                # original id -> component id for nodes that
                                    #   still have edges deferred to an ancestor
        deferred/from_<key>.npz     # edges first introduced in THIS block,
                                    #   written by the leaf that read them
        stats.json
        done                        # written last; marks the block complete
      L1/...

``work_dir`` is any ``UPath``: a local directory or a URL on an fsspec
filesystem (``s3://bucket/run/``, ...). Every file is written whole: locally
to a temporary name renamed into place, remotely as a single object write
(atomic on object stores, which have no rename). The ``done`` marker is the
last thing written, so a block is either complete or absent. Files in other
blocks' ``deferred/`` directories are rewritten identically on a re-run, which
keeps a restart idempotent. "Directories" are only key prefixes on an object
store, so nothing here relies on a directory existing: listings are globs.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from upath import UPath

from volara.paths import is_local, rm, to_upath

from .methods import SignedEdges
from .octree import Index, Octree


def _write_bytes(path: UPath, data: bytes) -> None:
    """Write ``data`` to ``path`` so that readers see the old file or the
    whole new one, never a partial write."""
    if is_local(path):
        local = Path(str(path))
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, local)
    else:
        path.write_bytes(data)  # one PUT: atomic on object stores


def _atomic_savez(path: UPath, **arrays) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    _write_bytes(path, buffer.getvalue())


def _atomic_write_text(path: UPath, text: str) -> None:
    _write_bytes(path, text.encode())


def _load_npz(path: UPath) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(path.read_bytes())) as d:
        return {k: d[k] for k in d.files}


class HierarchyStore:
    def __init__(self, work_dir: Any):
        self.work_dir: UPath = to_upath(work_dir)

    # -- manifest -----------------------------------------------------------

    @property
    def manifest_path(self) -> UPath:
        return self.work_dir / "manifest.json"

    def ensure_manifest(self, manifest: dict) -> None:
        """Write ``manifest`` if absent, otherwise require it to match."""
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text())
            if existing != manifest:
                raise RuntimeError(
                    f"{self.work_dir} holds intermediate results for a different "
                    f"configuration.\n  stored:  {existing}\n  current: {manifest}\n"
                    "Drop the task (GraphCluster.drop()) or use another work_dir."
                )
            return
        _atomic_write_text(self.manifest_path, json.dumps(manifest, sort_keys=True))

    # -- paths --------------------------------------------------------------

    def level_dir(self, level: int) -> UPath:
        return self.work_dir / f"L{level}"

    def block_dir(self, level: int, index: Index) -> UPath:
        return self.level_dir(level) / "_".join(str(int(i)) for i in index)

    def level_files(self, level: int, name: str) -> Iterator[UPath]:
        """Every block's ``name`` file at ``level``, in block-key order."""
        yield from sorted(self.level_dir(level).glob(f"*/{name}"), key=str)

    # -- completion ---------------------------------------------------------

    def is_done(self, level: int, index: Index) -> bool:
        return (self.block_dir(level, index) / "done").exists()

    def mark_done(self, level: int, index: Index) -> None:
        _atomic_write_text(self.block_dir(level, index) / "done", "")

    def clear_block(self, level: int, index: Index) -> None:
        """Remove a block's own outputs (not deferred files others wrote into it)."""
        d = self.block_dir(level, index)
        for name in ("done", "lut.npz", "graph.npz", "boundary.npz", "stats.json"):
            rm(d / name, recursive=False)

    # -- per-block artifacts -------------------------------------------------

    def write_lut(self, level: int, index: Index, in_ids, out_ids) -> None:
        _atomic_savez(
            self.block_dir(level, index) / "lut.npz",
            in_ids=np.asarray(in_ids, dtype=np.int64),
            out_ids=np.asarray(out_ids, dtype=np.int64),
        )

    def read_lut(self, level: int, index: Index) -> tuple[np.ndarray, np.ndarray]:
        d = _load_npz(self.block_dir(level, index) / "lut.npz")
        return d["in_ids"], d["out_ids"]

    def write_graph(self, level: int, index: Index, nodes, edges: SignedEdges) -> None:
        _atomic_savez(
            self.block_dir(level, index) / "graph.npz",
            nodes=np.asarray(nodes, dtype=np.int64),
            **edges.arrays(),
        )

    def read_graph(self, level: int, index: Index) -> tuple[np.ndarray, SignedEdges]:
        d = _load_npz(self.block_dir(level, index) / "graph.npz")
        return d["nodes"], SignedEdges.from_arrays(d)

    def write_boundary(
        self, level: int, index: Index, orig, comp, max_level, max_attractive_level
    ) -> None:
        """Original node -> current component for nodes with an edge deferred
        to an ancestor; ``max_level`` is the highest level any such edge is
        introduced at, ``max_attractive_level`` the same over attractive edges
        only (what decides whether the component can still merge)."""
        _atomic_savez(
            self.block_dir(level, index) / "boundary.npz",
            orig=np.asarray(orig, dtype=np.int64),
            comp=np.asarray(comp, dtype=np.int64),
            max_level=np.asarray(max_level, dtype=np.int64),
            max_attractive_level=np.asarray(max_attractive_level, dtype=np.int64),
        )

    def read_boundary(
        self, level: int, index: Index
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        d = _load_npz(self.block_dir(level, index) / "boundary.npz")
        return d["orig"], d["comp"], d["max_level"], d["max_attractive_level"]

    def write_deferred(
        self,
        target_level: int,
        target_index: Index,
        source_level: int,
        source_index: Index,
        edges: SignedEdges,
    ) -> None:
        path = (
            self.block_dir(target_level, target_index)
            / "deferred"
            / f"from_{Octree.key(source_level, source_index)}.npz"
        )
        _atomic_savez(path, **edges.arrays())

    def read_deferred(self, level: int, index: Index) -> SignedEdges:
        d = self.block_dir(level, index) / "deferred"
        parts = [
            SignedEdges.from_arrays(_load_npz(p))
            for p in sorted(d.glob("from_*.npz"), key=str)
        ]
        return SignedEdges.concat(parts) if parts else SignedEdges.empty()

    def write_stats(self, level: int, index: Index, stats: dict) -> None:
        _atomic_write_text(
            self.block_dir(level, index) / "stats.json",
            json.dumps(stats, sort_keys=True),
        )

    def read_stats(self, level: int, index: Index) -> dict:
        return json.loads((self.block_dir(level, index) / "stats.json").read_text())

    def level_stats(self, level: int) -> list[dict]:
        return [
            json.loads(p.read_text()) for p in self.level_files(level, "stats.json")
        ]

    # -- composition --------------------------------------------------------

    def level_lut(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """Concatenation of every block LUT at ``level`` (disjoint inputs)."""
        ins, outs = [], []
        for p in self.level_files(level, "lut.npz"):
            arrays = _load_npz(p)
            ins.append(arrays["in_ids"])
            outs.append(arrays["out_ids"])
        if not ins:
            z = np.zeros(0, dtype=np.int64)
            return z, z
        return np.concatenate(ins), np.concatenate(outs)

    # -- cleanup ------------------------------------------------------------

    def drop_level(self, level: int) -> None:
        rm(self.level_dir(level))

    def drop(self) -> None:
        rm(self.work_dir)
