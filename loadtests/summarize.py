"""Summarize a load-test session into one table.

Usage:  python loadtests/summarize.py loadtests/results/<session-folder>

Reads each run's Locust statistics and resource samples and writes summary.csv and
summary.md into the session folder (and prints the Markdown table).
"""

import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean

WARMUP_SECONDS = 5  # resource samples taken while users are still starting are skipped

COLUMNS = [
    ("run", "Run"),
    ("users", "Users"),
    ("text_mix", "Texts"),
    ("requests", "Requests"),
    ("failures", "Failures"),
    ("failure_pct", "Fail %"),
    ("rps", "Req/s"),
    ("p50_ms", "p50 ms"),
    ("p95_ms", "p95 ms"),
    ("p99_ms", "p99 ms"),
    ("max_ms", "Max ms"),
    ("server_cpu_pct", "Server CPU %"),
    ("worker_rss_max_mb", "Worker RSS max MB"),
    ("server_rss_max_mb", "Server RSS max MB"),
    ("steal_pct", "Steal %"),
    ("locust_cpu_max_pct", "Locust CPU max %"),
]


def server_cpu_count(session: Path) -> int | None:
    match = re.search(r"on (\d+) CPU\(s\)", (session / "server.log").read_text(errors="replace"))
    return int(match.group(1)) if match else None


def locust_totals(run_dir: Path) -> dict:
    with open(run_dir / "locust_stats.csv", newline="") as file:
        rows = {row["Name"]: row for row in csv.DictReader(file)}
    total = rows["Aggregated"]
    requests = int(total["Request Count"])
    failures = int(total["Failure Count"])
    return {
        "requests": requests,
        "failures": failures,
        "failure_pct": round(100 * failures / requests, 2) if requests else 0.0,
        "rps": round(float(total["Requests/s"]), 1),
        "p50_ms": total["50%"],
        "p95_ms": total["95%"],
        "p99_ms": total["99%"],
        "max_ms": round(float(total["Max Response Time"])),
    }


def resource_totals(run_dir: Path, cpus: int | None) -> dict:
    path = run_dir / "resources.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as file:
        rows = [r for r in csv.DictReader(file) if float(r["elapsed_s"]) >= WARMUP_SECONDS]
    if not rows:
        return {}

    # Add up the master and all workers at each sampling moment.
    server_cpu_by_time: dict[str, float] = {}
    server_rss_by_time: dict[str, float] = {}
    for r in rows:
        if r["role"] in ("master", "worker"):
            moment = r["timestamp"]
            server_cpu_by_time[moment] = server_cpu_by_time.get(moment, 0.0) + float(
                r["cpu_percent"]
            )
            server_rss_by_time[moment] = server_rss_by_time.get(moment, 0.0) + float(r["rss_mb"])
    worker_rss = [float(r["rss_mb"]) for r in rows if r["role"] == "worker"]
    steal = [float(r["steal_percent"]) for r in rows if r["role"] == "system"]
    locust_cpu = [float(r["cpu_percent"]) for r in rows if r["role"] == "locust"]

    result = {}
    if server_cpu_by_time and cpus:
        # Process CPU is relative to one core; express it as a share of the server's CPUs.
        result["server_cpu_pct"] = round(mean(server_cpu_by_time.values()) / cpus, 1)
    if worker_rss:
        result["worker_rss_max_mb"] = round(max(worker_rss))
    if server_rss_by_time:
        # The whole service's memory at its highest moment (shared libraries are
        # counted once per process, so this slightly overstates the true total).
        result["server_rss_max_mb"] = round(max(server_rss_by_time.values()))
    if steal:
        result["steal_pct"] = round(mean(steal), 1)
    if locust_cpu:
        result["locust_cpu_max_pct"] = round(max(locust_cpu))
    return result


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    session = Path(sys.argv[1])
    cpus = server_cpu_count(session)

    rows = []
    for run_dir in sorted((session / "runs").iterdir()):
        if not (run_dir / "locust_stats.csv").exists():
            continue
        info = json.loads((run_dir / "run.json").read_text())
        rows.append(
            {
                "run": run_dir.name,
                "users": info["users"],
                "text_mix": info["text_mix"],
                **locust_totals(run_dir),
                **resource_totals(run_dir, cpus),
            }
        )

    with open(session / "summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[key for key, _ in COLUMNS])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| " + " | ".join(title for _, title in COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in COLUMNS) + " |")
    note = f"Server CPU % is the share of the server's {cpus} CPU(s); Locust CPU is % of one core."
    table = "\n".join(lines) + "\n\n" + note + "\n"
    (session / "summary.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
