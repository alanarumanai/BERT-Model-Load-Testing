"""Record CPU and memory once per interval while a load test runs.

Writes one row per process per sample (the Gunicorn master, each worker, and
optionally Locust) plus one "system" row with machine-wide figures, including
CPU steal time, which shows how much CPU a VM's host took away.

Usage:
  python loadtests/monitor.py --server-pid PID [--locust-pid PID] --output resources.csv

Runs until it receives SIGTERM or SIGINT (Ctrl+C), or the server process exits.
Note: a process's cpu_percent is relative to ONE core, so 250 means 2.5 cores busy.
"""

import argparse
import csv
import signal
import time

import psutil

FIELDS = [
    "timestamp",
    "elapsed_s",
    "role",
    "pid",
    "cpu_percent",
    "rss_mb",
    "num_threads",
    "system_cpu_percent",
    "steal_percent",
    "mem_used_mb",
    "mem_available_mb",
    "load_1m",
]

running = True


def stop(signum, frame):
    global running
    running = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server-pid", type=int, required=True, help="Gunicorn master PID")
    parser.add_argument("--locust-pid", type=int, help="Locust PID, to check it isn't saturated")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    parser.add_argument("--output", required=True, help="CSV file to write")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    master = psutil.Process(args.server_pid)
    tracked: dict[int, tuple[str, psutil.Process]] = {master.pid: ("master", master)}
    if args.locust_pid:
        tracked[args.locust_pid] = ("locust", psutil.Process(args.locust_pid))

    # psutil's CPU percentages measure the time since the previous call, so the first
    # call only sets a starting point. Prime everything before the first sample.
    psutil.cpu_percent(interval=None)
    psutil.cpu_times_percent(interval=None)
    for _, process in tracked.values():
        process.cpu_percent(interval=None)

    started = time.time()
    next_sample = started + args.interval

    with open(args.output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        while running and master.is_running():
            time.sleep(max(0.0, next_sample - time.time()))
            next_sample += args.interval
            now = time.time()
            base = {"timestamp": f"{now:.3f}", "elapsed_s": f"{now - started:.1f}"}

            # Pick up workers started since the last sample (e.g. after a restart).
            try:
                for child in master.children():
                    if child.pid not in tracked:
                        child.cpu_percent(interval=None)
                        tracked[child.pid] = ("worker", child)
            except psutil.NoSuchProcess:
                break

            for pid, (role, process) in list(tracked.items()):
                try:
                    with process.oneshot():
                        writer.writerow(
                            {
                                **base,
                                "role": role,
                                "pid": pid,
                                "cpu_percent": f"{process.cpu_percent(interval=None):.1f}",
                                "rss_mb": f"{process.memory_info().rss / 2**20:.1f}",
                                "num_threads": process.num_threads(),
                            }
                        )
                except psutil.NoSuchProcess:
                    del tracked[pid]  # a worker exited or Locust finished

            memory = psutil.virtual_memory()
            writer.writerow(
                {
                    **base,
                    "role": "system",
                    "system_cpu_percent": f"{psutil.cpu_percent(interval=None):.1f}",
                    "steal_percent": f"{getattr(psutil.cpu_times_percent(None), 'steal', 0):.1f}",
                    "mem_used_mb": f"{memory.used / 2**20:.0f}",
                    "mem_available_mb": f"{memory.available / 2**20:.0f}",
                    "load_1m": f"{psutil.getloadavg()[0]:.2f}",
                }
            )
            file.flush()  # keep the file complete even if the run is interrupted


if __name__ == "__main__":
    main()
