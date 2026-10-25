"""Summarize a py-spy flame graph (SVG) as numbers you can put in a report.

Usage:  python profiling/analyze_flamegraph.py path/to/flamegraph.svg [--top 15]

Prints:
  1. a breakdown of all samples into categories (GIL waiting, matrix maths, ...)
  2. the functions with the most *self* time (where the samples actually were)
  3. how the time inside the /predict endpoint divides between stages

Note: py-spy samples every thread, including threads that are waiting. Waiting
samples are time a request spent stuck, not CPU used. Shares are of thread-time.
"""

import argparse
import collections
import html
import re
from pathlib import Path

FRAME = re.compile(r'<g><title>([^<]*)</title><rect x="([\d.]+)%" y="([\d.]+)" width="([\d.]+)%"')
TITLE = re.compile(r"(.*) \(([\d,]+) samples?, ([\d.]+)%\)$")
WAIT_FUNCS = (
    "pthread_cond_timedwait",
    "__lll_lock_wait",
    "pthread_cond_signal",
    "__condvar_cancel_waiting",
    "__pthread_mutex_cond_lock",
    "pthread_mutex_lock",
)


def base(name: str) -> str:
    return re.sub(r" \(.*\)$", "", name)


def load(path: Path) -> list[dict]:
    frames = FRAME.findall(path.read_text())
    depths = sorted({float(y) for _, _, y, _ in frames})
    nodes = []
    for title, x, y, w in frames:
        match = TITLE.match(html.unescape(title))
        nodes.append(
            {
                "name": match.group(1),
                "samples": int(match.group(2).replace(",", "")),
                "depth": depths.index(float(y)),
                "x": float(x),
                "w": float(w),
                "kids": [],
                "parent": None,
            }
        )
    by_depth = collections.defaultdict(list)
    for node in nodes:
        by_depth[node["depth"]].append(node)
    for depth in sorted(by_depth)[1:]:
        for node in by_depth[depth]:
            for parent in by_depth[depth - 1]:
                if parent["x"] - 1e-6 <= node["x"] and node["x"] + node["w"] <= (
                    parent["x"] + parent["w"] + 0.02
                ):
                    parent["kids"].append(node)
                    node["parent"] = parent
                    break
    return nodes


def stack(node: dict) -> list[str]:
    names = []
    while node:
        names.append(base(node["name"]))
        node = node["parent"]
    return names[::-1]


def self_samples(node: dict) -> int:
    return node["samples"] - sum(kid["samples"] for kid in node["kids"])


OPENMP_MARKERS = ("gomp_", "GOMP_", "__kmp", "kmp_", "omp_", "do_spin", "sched_yield")


def category(node: dict) -> str:
    name, frames = node["name"], stack(node)
    # PyTorch's thread pool (OpenMP) coordinating its threads: spinning, barriers,
    # yielding. Large shares mean threads are waiting on each other, not computing.
    # libgomp (GNU OpenMP, used by PyTorch's Linux wheels) often has no symbol names,
    # so py-spy shows its frames as bare addresses "0x... (libgomp.so.1)".
    in_openmp_library = "libgomp" in name or "libiomp" in name
    if in_openmp_library or name.startswith(OPENMP_MARKERS):
        return "OpenMP thread synchronisation (spinning, barriers)"
    if name.startswith(WAIT_FUNCS):
        if any("gil_scoped_release" in f or "THPVariable" in f for f in frames):
            return "waiting for the GIL (inside PyTorch calls)"
        return "other waiting (locks, queues)"
    if "gemm" in name or name.startswith("mkl_"):
        return "matrix multiplication (BLAS)"
    if "epoll" in name:
        return "event loop idle (epoll)"
    if "tokenize" in frames:
        return "tokenization"
    if "postprocess" in frames:
        return "post-processing"
    if "forward" in frames:
        return "other model work (PyTorch ops, Python glue)"
    if any(f in ("makeRecord", "emit", "handle", "info") for f in frames):
        return "logging"
    return "web framework and event loop"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("svg", type=Path)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    nodes = load(args.svg)
    total = max(node["samples"] for node in nodes)

    def pct(value: float) -> str:
        return f"{100 * value / total:6.2f}%"

    print(f"{args.svg}: {total} samples\n\n== Where all samples were ==")
    categories = collections.Counter()
    for node in nodes:
        if self_samples(node) > 0:
            categories[category(node)] += self_samples(node)
    for name, count in categories.most_common():
        print(f"{pct(count)}  {name}")

    print(f"\n== Top {args.top} functions by self time ==")
    selfs = collections.Counter()
    for node in nodes:
        selfs[node["name"]] += self_samples(node)
    for name, count in selfs.most_common(args.top):
        print(f"{pct(count)}  {name[:110]}")

    # The /predict endpoint as run in the thread pool: predict (app/main.py)
    # directly under FastAPI's _run_sync_endpoint.
    roots = [
        n
        for n in nodes
        if base(n["name"]) == "predict"
        and "app/main.py" in n["name"]
        and n["parent"]
        and base(n["parent"]["name"]) == "_run_sync_endpoint"
    ]
    endpoint = sum(n["samples"] for n in roots)
    if endpoint:
        print(f"\n== Inside the /predict endpoint ({pct(endpoint).strip()} of all samples) ==")
        stages = collections.Counter()
        inside = collections.Counter()

        def walk(node: dict, seen: frozenset) -> None:
            name = base(node["name"])
            if "app/model.py" in node["name"] and name not in seen:
                stages[name] += node["samples"]
            if self_samples(node) > 0:
                inside[category(node)] += self_samples(node)
            for kid in node["kids"]:
                walk(kid, seen | {name})

        for root in roots:
            walk(root, frozenset())
        for name in ("tokenize", "forward", "postprocess"):
            print(f"{100 * stages[name] / endpoint:6.1f}%  {name} (stage)")
        print("  of which:")
        for name, count in inside.most_common():
            print(f"{100 * count / endpoint:6.1f}%    {name}")


if __name__ == "__main__":
    main()