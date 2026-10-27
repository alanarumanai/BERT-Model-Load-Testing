# DistilBERT Sentiment Service

![CI](https://github.com/alanarumanai/BERT-Model-Load-Testing/actions/workflows/ci.yml/badge.svg)

A FastAPI service that classifies English text as **POSITIVE** or **NEGATIVE** with a pretrained
DistilBERT model, served by Gunicorn with Uvicorn workers. The repository also contains the tests,
the CI pipeline, the Locust load tests, the profiling tools and the results behind the report.

## Results in brief

Measured on 8 pinned cores of an AMD EPYC 9334, against a baseline run in the same sitting:

| Configuration | Peak throughput | p95 at 64 users | SST-2 accuracy | Peak memory |
|---|---|---|---|---|
| Default: FP32, 2 workers × 4 threads | 129.6 req/s | 830 ms | 91.06% | 2.2 GB |
| `PROFILE=throughput`: int8, 8 × 1, limit 2 | 491.6 req/s (3.8×) | 260 ms | 89.91% | 5.1 GB |
| `PROFILE=accuracy`: BF16, 8 × 1, limit 2 | 282.9 req/s (2.2×) | 420 ms | 91.06% | 4.7 GB |

No request failed in any test. The full analysis, stage by stage, is in
[`report/REPORT.md`](report/REPORT.md) (also as [PDF](report/REPORT.pdf)).

## Requirements

- Linux (Gunicorn does not run on Windows; use WSL2 there)
- Python 3.11
- About 3 GB of disk for the virtual environment and the model
- RAM: about 1.5–2.5 GB for the default configuration, about 5–6 GB for the 8-worker profiles
- For `PROFILE=accuracy`: a CPU with BF16 support (`grep avx512_bf16 /proc/cpuinfo`).
  Without it, BF16 can be slower than FP32.

## Setup

```bash
git clone https://github.com/alanarumanai/BERT-Model-Load-Testing.git
cd BERT-Model-Load-Testing
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/install_deps.sh --dev     # CPU-only torch, then everything else
cp .env.example .env                   # settings, including the pinned model revision
python scripts/download_model.py       # downloads the model once (~260 MB)
```

`install_deps.sh` installs the CPU-only build of PyTorch, which is much smaller than the default
build. Use `bash scripts/install_deps.sh` without `--dev` for runtime dependencies only.

## Running the service

**Development** (one process, reloads on code changes):

```bash
uvicorn app.main:app --reload
```

**Production-style** under Gunicorn:

```bash
bash scripts/run_server.sh                         # all CPUs, default configuration
bash scripts/run_server.sh 0-7                     # pinned to CPUs 0-7
PROFILE=throughput bash scripts/run_server.sh 0-7  # int8, 8 workers x 1 thread, limit 2
PROFILE=accuracy bash scripts/run_server.sh 0-7    # BF16, 8 workers x 1 thread, limit 2
```

The service listens on port 8000. Interactive API docs are at <http://127.0.0.1:8000/docs>.

**Example requests:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was wonderful"}'
```

Response (scores vary slightly):

```json
{"label": "POSITIVE", "score": 0.9998, "scores": {"NEGATIVE": 0.0002, "POSITIVE": 0.9998},
 "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english@714eb0fa89d2f80546fda750413ed43d93601a13"}
```

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great acting.", "The plot made no sense."]}'

curl http://127.0.0.1:8000/health
```

Empty or whitespace-only text, text over the character limit, or a batch over the size limit
returns HTTP 422 with an explanation. `/health` returns 503 until the model has loaded.

## Configuration

Every setting comes from an environment variable or `.env`. Variables set in the environment
take precedence over `.env` and over a profile.

| Variable | Default | Meaning |
|---|---|---|
| `APP_MODEL_NAME` | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | Hugging Face model |
| `APP_MODEL_REVISION` | `main` (pinned in `.env.example`) | model commit hash |
| `APP_MAX_LENGTH` | `128` | tokens read per text; longer texts are truncated (8–512) |
| `APP_MAX_TEXT_CHARS` | `5000` | longer requests are rejected with 422 |
| `APP_MAX_BATCH_SIZE` | `32` | maximum texts per batch request |
| `APP_TORCH_NUM_THREADS` | CPUs ÷ workers | PyTorch threads per worker |
| `APP_MAX_CONCURRENT_PREDICTIONS` | no limit | predictions allowed at once per worker |
| `APP_QUANTIZE` | `false` | int8 dynamic quantization of the Linear layers |
| `APP_PRECISION` | `fp32` | `fp32`, `bf16` or `fp16` (cannot be combined with `APP_QUANTIZE`) |
| `APP_LOG_LEVEL` | `INFO` | `WARNING` turns off the per-request log line |
| `GUNICORN_WORKERS` | `min(2, CPUs)` | worker processes |
| `GUNICORN_BIND` | `0.0.0.0:8000` | listen address |
| `GUNICORN_TIMEOUT` | `120` | seconds before an unresponsive worker is restarted |
| `PROFILE` | none | `throughput` or `accuracy`: loads `profiles/<name>.env` |

## Tests and CI

```bash
HF_HUB_OFFLINE=1 pytest                 # 70 tests, using the real model
HF_HUB_OFFLINE=1 pytest --cov=app       # with coverage
ruff check . && ruff format --check .   # lint and formatting
bash scripts/smoke_test.sh              # starts Gunicorn, checks /health and one prediction
```

GitHub Actions (`.github/workflows/ci.yml`) runs on every push to `main` and every pull request:
lint, the full test suite with coverage, and the server smoke test. Packages and the model are
cached between runs.

## Load testing

```bash
export SERVER_CPUS=0-7 LOCUST_CPUS=8-9   # service and Locust on separate cores
bash loadtests/run_load_tests.sh step length spike soak
```

Each session starts a fresh server, runs the chosen scenarios and saves everything in
`loadtests/results/<date-time>_<label>/`: Locust CSVs and HTML reports per run, CPU and memory
per process, startup time, the server log, the environment and a `summary.md`. Settings such as
`PROFILE=throughput`, `GUNICORN_WORKERS=8` or `LABEL=my-test` go in front of the command.

| Scenario | What it does |
|---|---|
| `step` | 1, 2, 4, 8, 16, 32, 64 users (`USER_LEVELS`), `STEP_DURATION` each (default 2 min) |
| `length` | 8 users, short texts only and then long texts only |
| `spike` | 4 users, then 64, then 4; 60 s each |
| `soak` | 8 users for 15 minutes |

Compare two sessions with:

```bash
python profiling/compare_sessions.py <baseline-session> <other-session> --output comparison.md
```

## Profiling

```bash
bash profiling/run_experiments.sh "<name>|<VAR=value ...>" ...   # several configurations vs a fresh baseline
python profiling/profile_stages.py --label <label>               # time tokenize / forward / postprocess
bash profiling/profile_server.sh <label>                          # py-spy flame graph under load
python profiling/analyze_flamegraph.py <flamegraph.svg>           # flame graph as numbers
python profiling/check_accuracy.py --variants baseline quantized bf16 fp16
python profiling/worker_balance.py <session>                      # load spread across workers
```

Each script's header explains its options. The exact commands used for the report are in each
stage of [`report/REPORT.md`](report/REPORT.md).

## Project structure

```
app/                  FastAPI service: config, model wrapper, schemas, endpoints
tests/                pytest suite (real model)
scripts/              install, model download, server start, smoke test, startup measurement
gunicorn.conf.py      Gunicorn settings
profiles/             recommended settings: throughput.env, accuracy.env
loadtests/            Locust file, texts, resource monitor, session runner, summaries, results
profiling/            experiment runner, stage timings, flame graphs, accuracy, comparisons, results
report/               the report (REPORT.md, REPORT.pdf) and its figures
docs/decisions.md     design decisions, with reasons and evidence
REPRODUCE.md          every step needed to reproduce the report
.github/workflows/    CI pipeline
```

## Reproducing the report

[`REPRODUCE.md`](REPRODUCE.md) lists every command, stage by stage, in the order the results were
produced, with where to compare each output with the report and how to troubleshoot.