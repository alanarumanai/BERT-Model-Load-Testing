"""Time each stage of a prediction: tokenization, forward pass and post-processing.

Also measures the web layer's overhead (FastAPI, validation, middleware, logging and
JSON) by comparing an in-process HTTP request with a direct model call.

Usage (match ONE Gunicorn worker of the baseline: 4 cores, 4 torch threads):
  APP_TORCH_NUM_THREADS=4 taskset -c 0-3 python profiling/profile_stages.py --label baseline

Any APP_* setting applies, e.g. APP_QUANTIZE=true or APP_MAX_LENGTH=64.
Results are printed and saved to profiling/results/stages_<label>_<time>.{csv,md}.
"""

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # for `import app`
sys.path.insert(0, str(ROOT / "loadtests"))  # for the shared texts
os.chdir(ROOT)  # so the settings find .env

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from texts import TEXTS  # noqa: E402

from app.main import app  # noqa: E402


def ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=20)[-1] if len(values) >= 20 else max(values)


def time_stages(classifier, texts: list[str], repeats: int, warmup: int) -> dict:
    """Run the three stages `repeats` times and return median timings."""
    tokenize, forward, post, total = [], [], [], []
    tokens = 0
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        inputs = classifier.tokenize(texts)
        t1 = time.perf_counter()
        logits = classifier.forward(inputs)
        t2 = time.perf_counter()
        classifier.postprocess(logits)
        t3 = time.perf_counter()
        if i >= warmup:  # discard the warm-up iterations
            tokenize.append(t1 - t0)
            forward.append(t2 - t1)
            post.append(t3 - t2)
            total.append(t3 - t0)
            tokens = inputs["input_ids"].shape[1]  # padded length of this batch
    median_total = statistics.median(total)
    return {
        "tokens": tokens,
        "tokenize_ms": ms(statistics.median(tokenize)),
        "forward_ms": ms(statistics.median(forward)),
        "postprocess_ms": ms(statistics.median(post)),
        "total_ms": ms(median_total),
        "total_p95_ms": ms(p95(total)),
        "per_text_ms": ms(median_total / len(texts)),
        "forward_share_pct": round(100 * statistics.median(forward) / median_total, 1),
    }


def time_http(client, classifier, text: str, repeats: int, warmup: int) -> dict:
    """Compare a direct predict() call with a full in-process HTTP request."""
    direct, http = [], []
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        classifier.predict([text])
        t1 = time.perf_counter()
        response = client.post("/predict", json={"text": text})
        t2 = time.perf_counter()
        response.raise_for_status()
        if i >= warmup:
            direct.append(t1 - t0)
            http.append(t2 - t1)
    d, h = statistics.median(direct), statistics.median(http)
    return {
        "predict_ms": ms(d),
        "http_ms": ms(h),
        "web_overhead_ms": ms(h - d),
        "web_overhead_pct": round(100 * (h - d) / h, 1),
    }


def to_markdown(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    lines += ["| " + " | ".join(str(row[k]) for k in keys) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="baseline", help="name used in the output files")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--repeats", type=int, default=50, help="timed iterations per case")
    parser.add_argument("--warmup", type=int, default=5, help="untimed iterations per case")
    args = parser.parse_args()

    # Entering the client runs the app's lifespan, which loads the model with the
    # current settings, exactly as a Gunicorn worker would.
    with TestClient(app) as client:
        classifier = client.app.state.classifier
        print(f"Model: {classifier.model_id}, torch threads: {torch.get_num_threads()}")

        stage_rows = []
        for group, texts in TEXTS.items():
            for batch_size in args.batch_sizes:
                batch = (texts * batch_size)[:batch_size]  # repeat the group's texts
                print(f"  timing {group} texts, batch size {batch_size} ...", flush=True)
                stage_rows.append(
                    {
                        "texts": group,
                        "batch": batch_size,
                        **time_stages(classifier, batch, args.repeats, args.warmup),
                    }
                )

        http_rows = []
        for group, texts in TEXTS.items():
            print(f"  timing HTTP vs direct for {group} texts ...", flush=True)
            http_rows.append(
                {"texts": group, **time_http(client, classifier, texts[0], args.repeats, 5)}
            )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "profiling" / "results" / f"stages_{args.label}_{stamp}"
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (("", stage_rows), ("_http", http_rows)):
        with open(f"{out}{suffix}.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report = (
        f"# Stage timings: {args.label}\n\n"
        f"Model `{classifier.model_id}`, {torch.get_num_threads()} torch threads, "
        f"median of {args.repeats} runs (times in ms).\n\n"
        f"## Stages\n\n{to_markdown(stage_rows)}\n\n"
        f"## Web layer overhead (batch of 1)\n\n{to_markdown(http_rows)}\n"
    )
    Path(f"{out}.md").write_text(report)
    print("\n" + report)
    print(f"Saved {out}.md and .csv files")


if __name__ == "__main__":
    main()