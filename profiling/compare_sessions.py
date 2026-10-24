"""Compare load-test sessions against a baseline session.

Usage:
  python profiling/compare_sessions.py BASELINE_SESSION EXPERIMENT_SESSION [...] [--output FILE]

Each session is a folder made by loadtests/run_load_tests.sh (it must contain summary.csv).
Runs are matched by name without their number, e.g. "03_step_8u" matches "05_step_8u".
"""

import argparse
import csv
import re
from pathlib import Path

# (column, title, True if higher is better)
METRICS = [
    ("rps", "Req/s", True),
    ("p50_ms", "p50 ms", False),
    ("p95_ms", "p95 ms", False),
    ("p99_ms", "p99 ms", False),
    ("failure_pct", "Fail %", False),
    ("server_cpu_pct", "Server CPU %", None),
    ("worker_rss_max_mb", "Worker RSS MB", False),
]


def load(session: Path) -> dict[str, dict]:
    with open(session / "summary.csv", newline="") as file:
        return {re.sub(r"^\d+_", "", row["run"]): row for row in csv.DictReader(file)}


def label(session: Path) -> str:
    return re.sub(r"^\d{8}-\d{6}_", "", session.name)


def cell(before: str, after: str) -> str:
    if not before or not after:
        return f"{before or '-'} → {after or '-'}"
    b, a = float(before), float(after)
    if b:
        change = f"{100 * (a - b) / b:+.0f}%"
    else:
        change = "no change" if a == b else "from zero"
    return f"{before} → {after} ({change})"


def peak_rps(rows: dict[str, dict]) -> float:
    steps = [float(r["rps"]) for name, r in rows.items() if name.startswith("step")]
    return max(steps) if steps else 0.0


def compare(baseline: Path, experiment: Path) -> str:
    base, exp = load(baseline), load(experiment)
    lines = [f"## {label(experiment)} vs {label(baseline)}", ""]

    b_peak, e_peak = peak_rps(base), peak_rps(exp)
    if b_peak and e_peak:
        lines += [
            f"**Peak throughput (step runs): {b_peak:.1f} → {e_peak:.1f} req/s "
            f"({100 * (e_peak - b_peak) / b_peak:+.0f}%)**",
            "",
        ]

    lines.append("| Run | " + " | ".join(title for _, title, _ in METRICS) + " |")
    lines.append("|---|" + "|".join("---" for _ in METRICS) + "|")
    for name in base:
        if name in exp:
            cells = [cell(base[name][key], exp[name][key]) for key, _, _ in METRICS]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    missing = sorted(set(base) ^ set(exp))
    if missing:
        lines += ["", f"Runs in only one session (not compared): {', '.join(missing)}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="also write the comparison to this file")
    args = parser.parse_args()

    report = "# Load-test comparison\n\nValues are baseline → experiment (change).\n\n"
    report += "\n".join(compare(args.baseline, e) for e in args.experiments)
    print(report)
    if args.output:
        args.output.write_text(report)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
