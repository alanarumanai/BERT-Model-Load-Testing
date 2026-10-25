"""Show how evenly the load was spread across Gunicorn workers in each run.

Usage:
  python profiling/worker_balance.py SESSION [SESSION ...] [--output FILE]

For every run in each session, reads resources.csv and prints each worker's average
CPU (as % of one core, highest first), how many workers were busy (>= 50% of a core)
and how many cores the whole service kept busy on average. Samples from the first
5 seconds of each run are skipped, as in loadtests/summarize.py.
"""

import argparse
import collections
import csv
import re
import statistics
from pathlib import Path

WARMUP_SECONDS = 5
BUSY_THRESHOLD = 50  # % of one core


def run_balance(resources: Path) -> tuple[list[int], float]:
    per_worker = collections.defaultdict(list)
    server_by_time = collections.defaultdict(float)
    with open(resources, newline="") as file:
        for row in csv.DictReader(file):
            if float(row["elapsed_s"]) < WARMUP_SECONDS:
                continue
            if row["role"] in ("master", "worker"):
                server_by_time[row["timestamp"]] += float(row["cpu_percent"])
            if row["role"] == "worker":
                per_worker[row["pid"]].append(float(row["cpu_percent"]))
    workers = sorted((round(statistics.mean(v)) for v in per_worker.values()), reverse=True)
    busy_cores = statistics.mean(server_by_time.values()) / 100 if server_by_time else 0.0
    return workers, busy_cores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="also write the report to this file")
    args = parser.parse_args()

    lines = [
        "# Worker load balance",
        "",
        "Per-worker CPU is the run average as % of one core, highest first. "
        f"A worker counts as busy at >= {BUSY_THRESHOLD}%.",
    ]
    for session in args.sessions:
        label = re.sub(r"^\d{8}-\d{6}_", "", session.name)
        lines += [
            "",
            f"## {label}",
            "",
            "| Run | Per-worker CPU % | Busy workers | Busy cores |",
            "|---|---|---|---|",
        ]
        for run_dir in sorted((session / "runs").iterdir()):
            resources = run_dir / "resources.csv"
            if not resources.exists():
                continue
            workers, busy_cores = run_balance(resources)
            busy = sum(w >= BUSY_THRESHOLD for w in workers)
            name = re.sub(r"^\d+_", "", run_dir.name)
            lines.append(
                f"| {name} | {', '.join(map(str, workers))} | {busy} of {len(workers)} "
                f"| {busy_cores:.1f} |"
            )
    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        args.output.write_text(report)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
