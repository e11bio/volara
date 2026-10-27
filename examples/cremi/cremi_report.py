"""Resource report for a finished volara run, from daisy's run records.

With ``resource_tracking=True`` (volara's default) daisy appends one JSON
line per run to ``runs.jsonl`` in every task's tracking store
(``BlockwiseTask.tracking_path``, under volara's log basedir). This reads
the last line of each, rebuilds daisy's report from it, and prints:

1. a Markdown table, one row per task, in pipeline order;
2. daisy's own Resource Utilization panel for the same run.

usage: python cremi_report.py [log basedir, default ./volara_logs]
"""

import json
import sys
from pathlib import Path

import daisy

GB, MB = 1e9, 1e6

# The CREMI example's pipeline, upstream first. All four tasks are scheduled
# in one daisy pass and start together, so the records cannot order
# themselves; tasks not listed here follow alphabetically.
PIPELINE = [
    "fragments-extract-frags",
    "fragments-aff-agglom",
    "lut-graph-mws",
    "segments-relabel",
]


def last_run(path: Path) -> daisy.TaskReport | None:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return daisy.TaskReport.from_dict(json.loads(lines[-1])) if lines else None


def main(log_dir: Path):
    reports = [r for p in sorted(log_dir.glob("**/runs.jsonl")) if (r := last_run(p))]
    if not reports:
        sys.exit(f"no runs.jsonl under {log_dir}: has the example run yet?")
    reports.sort(
        key=lambda r: (
            (PIPELINE.index(r.task_id), "")
            if r.task_id in PIPELINE
            else (len(PIPELINE), r.task_id)
        )
    )
    start = min(r.started_at for r in reports)
    end = max(r.finished_at for r in reports)

    print(f"Pipeline wall time: {end - start:.1f} s\n")
    print(
        "| task | blocks | workers | block time s | CPU s | CPU / block time"
        " | mean block | peak RSS (max block) | IO read | IO write |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    totals = {"wall": 0.0, "cpu": 0.0, "read": 0.0, "write": 0.0}
    for r in reports:
        b = r.blocks
        if not b.has_stats:
            print(
                f"| {r.task_id} | {b.count} | {r.workers.count} | not measured |||||||"
            )
            continue
        totals["wall"] += b.wall.total
        totals["cpu"] += b.cpu.total
        totals["read"] += b.io_read.total
        totals["write"] += b.io_write.total
        print(
            f"| {r.task_id} | {b.count} | {r.workers.count} | {b.wall.total:.1f}"
            f" | {b.cpu.total:.1f} | {b.cpu.total / b.wall.total:.2f}"
            f" | {b.wall.mean:.3f} s | {b.peak_rss.max / GB:.2f} GB"
            f" | {b.io_read.total / MB:.0f} MB | {b.io_write.total / MB:.0f} MB |"
        )
    print(
        f"| **total** | | | {totals['wall']:.1f} | {totals['cpu']:.1f}"
        f" | {totals['cpu'] / totals['wall']:.2f} | | |"
        f" {totals['read'] / MB:.0f} MB | {totals['write'] / MB:.0f} MB |"
    )
    print()
    run = daisy.RunReport({r.task_id: r for r in reports})
    print(run.render(task_order=[r.task_id for r in reports]))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "volara_logs"))
