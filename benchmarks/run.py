#!/usr/bin/env python3
"""Emit benchmarks/output.json in github-action-benchmark's customSmallerIsBetter format.

Metrics:
  import … (cold)               wall time of `python -I -c "import <IMPORTS>"` minus
                                `python -I -c "pass"`, median of N runs
  pytest wall time              the duration of the test suite, passed in with
                                --pytest-seconds by the tests job that just ran it

Run from the repo root: uv run python benchmarks/run.py --pytest-seconds 12.3
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

# The modules a user imports first; the top-level package where its __init__
# carries the public surface, the headline submodules where it does not.
IMPORTS = ["volara.blockwise", "volara.datasets"]
ROOT = Path(__file__).resolve().parent.parent


def wall(cmd: list[str], runs: int) -> list[float]:
    out = []
    for _ in range(runs):
        t = time.perf_counter()
        subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
        out.append(time.perf_counter() - t)
    return out


def import_time(runs: int) -> dict:
    base = wall([sys.executable, "-I", "-c", "pass"], runs)
    full = wall([sys.executable, "-I", "-c", f"import {', '.join(IMPORTS)}"], runs)
    delta = [max(0.0, f - statistics.median(base)) for f in full]
    return entry(
        f"import {', '.join(IMPORTS)} (cold)",
        statistics.median(delta) * 1e3,
        "ms",
        pstdev(delta) * 1e3,
        f"interpreter startup {statistics.median(base) * 1e3:.1f} ms subtracted; n={runs}",
    )


def pytest_time(seconds: float, command: str) -> dict:
    return entry(
        "pytest wall time", seconds, "s", None, f"`{command}` in the tests job"
    )


def pstdev(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def entry(name, value, unit, rng, extra):
    e = {"name": name, "unit": unit, "value": round(value, 3)}
    if rng is not None:
        e["range"] = f"± {rng:.3f}"
    if extra:
        e["extra"] = extra
    return e


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="benchmarks/output.json")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--pytest-seconds", type=float, default=None)
    ap.add_argument("--pytest-command", default="uv run pytest")
    args = ap.parse_args()
    results = [import_time(args.runs)]
    if args.pytest_seconds is not None:
        results.append(pytest_time(args.pytest_seconds, args.pytest_command))
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
