"""Compare accuracy, agreement, speed and size of model variants on SST-2.

Uses the SST-2 validation set (872 labelled movie-review sentences), the same
benchmark the model was evaluated on. The first variant is the reference that the
others are compared against.

Usage:
  APP_TORCH_NUM_THREADS=4 taskset -c 0-3 python profiling/check_accuracy.py
  python profiling/check_accuracy.py --variants baseline quantized max_length=64 \\
      quantized+max_length=64

Variant names: "baseline" (current settings), "quantized", "max_length=N", or several
joined with "+". The dataset is downloaded once from the Hugging Face Hub, so run this
without HF_HUB_OFFLINE the first time.
"""

import argparse
import csv
import gc
import io
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.model import SentimentClassifier  # noqa: E402

SST2_LABELS = {0: "NEGATIVE", 1: "POSITIVE"}  # the dataset's label numbers


def load_sst2_validation(limit: int | None) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    data = load_dataset("stanfordnlp/sst2", split="validation")
    if limit:
        data = data.select(range(min(limit, len(data))))
    return list(data["sentence"]), [SST2_LABELS[label] for label in data["label"]]


def parse_variant(name: str) -> dict:
    options: dict = {}
    for part in name.split("+"):
        if part == "baseline":
            continue
        if part == "quantized":
            options["quantize"] = True
        elif part.startswith("max_length="):
            options["max_length"] = int(part.split("=", 1)[1])
        else:
            raise SystemExit(f"Unknown variant part: {part!r}")
    return options


def model_size_mb(model: torch.nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return round(buffer.tell() / 2**20, 1)


def evaluate(name: str, sentences: list[str], latency_samples: int) -> dict:
    settings = get_settings()
    options = {"max_length": settings.max_length, "quantize": settings.quantize}
    options.update(parse_variant(name))
    classifier = SentimentClassifier(
        model_name=settings.model_name,
        revision=settings.model_revision,
        num_threads=settings.torch_num_threads,
        **options,
    )

    # Predictions for the whole set, in batches for speed.
    predictions = []
    for start in range(0, len(sentences), 32):
        predictions += classifier.predict(sentences[start : start + 32])

    # Serving-style latency: one sentence per call, as the API receives them.
    latencies = []
    for sentence in sentences[:latency_samples]:
        t0 = time.perf_counter()
        classifier.predict([sentence])
        latencies.append(time.perf_counter() - t0)

    result = {
        "labels": [p.label for p in predictions],
        "positive": [p.scores["POSITIVE"] for p in predictions],
        "latency_ms": round(1000 * statistics.median(latencies), 2),
        "size_mb": model_size_mb(classifier.model),
        "model_id": classifier.model_id,
    }
    del classifier
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=["baseline", "quantized", "max_length=64"])
    parser.add_argument("--limit", type=int, help="use only the first N sentences")
    parser.add_argument("--latency-samples", type=int, default=200)
    args = parser.parse_args()

    sentences, truth = load_sst2_validation(args.limit)
    print(f"SST-2 validation: {len(sentences)} sentences, torch threads: {torch.get_num_threads()}")

    results = {}
    for name in args.variants:
        print(f"  evaluating {name} ...", flush=True)
        results[name] = evaluate(name, sentences, args.latency_samples)

    reference = results[args.variants[0]]
    rows = []
    for name, r in results.items():
        correct = sum(p == t for p, t in zip(r["labels"], truth, strict=True))
        agree = sum(a == b for a, b in zip(r["labels"], reference["labels"], strict=True))
        diffs = [abs(a - b) for a, b in zip(r["positive"], reference["positive"], strict=True)]
        rows.append(
            {
                "variant": name,
                "accuracy_pct": round(100 * correct / len(truth), 2),
                "agreement_pct": round(100 * agree / len(truth), 2),
                "mean_prob_diff": round(statistics.mean(diffs), 4),
                "latency_ms": r["latency_ms"],
                "speedup": round(reference["latency_ms"] / r["latency_ms"], 2),
                "size_mb": r["size_mb"],
            }
        )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "profiling" / "results" / f"accuracy_{stamp}"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = list(rows[0])
    table = "\n".join(
        ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
        + ["| " + " | ".join(str(row[k]) for k in keys) + " |" for row in rows]
    )
    report = (
        f"# Accuracy check on SST-2 validation ({len(truth)} sentences)\n\n"
        f"Reference variant: `{args.variants[0]}`. Agreement and probability difference "
        f"are measured against it. Latency is the median of {args.latency_samples} "
        f"single-sentence calls with {torch.get_num_threads()} torch threads.\n\n{table}\n"
    )
    Path(f"{out}.md").write_text(report)
    print("\n" + report)
    print(f"Saved {out}.md and .csv")


if __name__ == "__main__":
    main()