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

Every file is written to a temporary name and renamed into place, and the
``done`` marker is the last thing written, so a block is either complete or
absent. Files in other blocks' ``deferred/`` directories are rewritten
identically on a re-run, which keeps a restart idempotent.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np

from .methods import SignedEdges
from .octree import Index, Octree


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class HierarchyStore:
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)

    # -- manifest -----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
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

    def level_dir(self, level: int) -> Path:
        return self.work_dir / f"L{level}"

    def block_dir(self, level: int, index: Index) -> Path:
        return self.level_dir(level) / "_".join(str(int(i)) for i in index)

    def block_dirs(self, level: int) -> Iterator[Path]:
        level_dir = self.level_dir(level)
        if not level_dir.exists():
            return
        for p in sorted(level_dir.iterdir()):
            if p.is_dir():
                yield p

    # -- completion ---------------------------------------------------------

    def is_done(self, level: int, index: Index) -> bool:
        return (self.block_dir(level, index) / "done").exists()

    def mark_done(self, level: int, index: Index) -> None:
        _atomic_write_text(self.block_dir(level, index) / "done", "")

    def clear_block(self, level: int, index: Index) -> None:
        """Remove a block's own outputs (not deferred files others wrote into it)."""
        d = self.block_dir(level, index)
        for name in ("done", "lut.npz", "graph.npz", "boundary.npz", "stats.json"):
            p = d / name
            if p.exists():
                p.unlink()

    # -- per-block artifacts -------------------------------------------------

    def write_lut(self, level: int, index: Index, in_ids, out_ids) -> None:
        _atomic_savez(
            self.block_dir(level, index) / "lut.npz",
            in_ids=np.asarray(in_ids, dtype=np.int64),
            out_ids=np.asarray(out_ids, dtype=np.int64),
        )

    def read_lut(self, level: int, index: Index) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.block_dir(level, index) / "lut.npz") as d:
            return d["in_ids"], d["out_ids"]

    def write_graph(self, level: int, index: Index, nodes, edges: SignedEdges) -> None:
        _atomic_savez(
            self.block_dir(level, index) / "graph.npz",
            nodes=np.asarray(nodes, dtype=np.int64),
            **edges.arrays(),
        )

    def read_graph(self, level: int, index: Index) -> tuple[np.ndarray, SignedEdges]:
        with np.load(self.block_dir(level, index) / "graph.npz") as d:
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
        with np.load(self.block_dir(level, index) / "boundary.npz") as d:
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
        if not d.exists():
            return SignedEdges.empty()
        parts = []
        for p in sorted(d.glob("from_*.npz")):
            with np.load(p) as arrays:
                parts.append(SignedEdges.from_arrays(arrays))
        return SignedEdges.concat(parts)

    def write_stats(self, level: int, index: Index, stats: dict) -> None:
        _atomic_write_text(
            self.block_dir(level, index) / "stats.json",
            json.dumps(stats, sort_keys=True),
        )

    def read_stats(self, level: int, index: Index) -> dict:
        return json.loads((self.block_dir(level, index) / "stats.json").read_text())

    def level_stats(self, level: int) -> list[dict]:
        out = []
        for d in self.block_dirs(level):
            p = d / "stats.json"
            if p.exists():
                out.append(json.loads(p.read_text()))
        return out

    # -- composition --------------------------------------------------------

    def level_lut(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """Concatenation of every block LUT at ``level`` (disjoint inputs)."""
        ins, outs = [], []
        for d in self.block_dirs(level):
            p = d / "lut.npz"
            if p.exists():
                with np.load(p) as arrays:
                    ins.append(arrays["in_ids"])
                    outs.append(arrays["out_ids"])
        if not ins:
            z = np.zeros(0, dtype=np.int64)
            return z, z
        return np.concatenate(ins), np.concatenate(outs)

    # -- cleanup ------------------------------------------------------------

    def drop_level(self, level: int) -> None:
        d = self.level_dir(level)
        if d.exists():
            shutil.rmtree(d)

    def drop(self) -> None:
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
