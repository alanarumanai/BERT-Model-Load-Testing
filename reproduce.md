# How to reproduce the results

This file lists every step needed to reproduce the results in [`report/REPORT.md`](report/REPORT.md),
in the order they were run. The full plan takes about 5 to 6 hours, mostly unattended.

Absolute numbers depend on the CPU. On different hardware, compare configurations with each other
(for example, int8 against FP32) rather than with the exact figures in the report.

## 1. What you need

- Linux with at least 10 CPUs: 8 for the service and 2 for Locust (the original runs used an
  AMD EPYC 9334). With fewer, change `SERVER_CPUS` and `LOCUST_CPUS` below and expect different numbers.
- Python 3.11, about 3 GB of disk, and about 6 GB of free RAM for the 8-worker configurations.
- Network access to PyPI and the Hugging Face Hub for the first setup.
- For the BF16 results to match: a CPU with `avx512_bf16`. For the flame graphs: permission to
  attach py-spy to your own processes (see Troubleshooting).

## 2. Set up

```bash
git clone https://github.com/alanarumanai/BERT-Model-Load-Testing.git
cd BERT-Model-Load-Testing
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/install_deps.sh --dev
cp .env.example .env              # pins APP_MODEL_REVISION=714eb0fa89d2f80546fda750413ed43d93601a13
python scripts/download_model.py
```

Check the setup before any load test:

```bash
HF_HUB_OFFLINE=1 pytest           # 70 tests pass
bash scripts/smoke_test.sh        # ends with "Smoke test passed"
uptime                            # load average near 0: nothing else heavy running
```

Set these once per terminal. Every command below assumes them:

```bash
export SERVER_CPUS=0-7 LOCUST_CPUS=8-9 STEP_DURATION=1m LENGTH_DURATION=1m
export USER_LEVELS="1 2 4 8 16 32 64"
```

Do not export any `GUNICORN_*`, `APP_*` or `PROFILE` variable; put them in front of individual
commands only, as shown.

Each command creates a new folder named `<date-time>_<label>`, so your folder names will differ
from the ones in the report. Where a later command needs an earlier result, it uses a pattern
such as `loadtests/results/*_baseline`.

## 3. Stage 1: baseline (about 40 minutes)

```bash
LABEL=baseline bash loadtests/run_load_tests.sh step length spike soak
LABEL=baseline-repeat bash loadtests/run_load_tests.sh step
python profiling/compare_sessions.py loadtests/results/*_baseline loadtests/results/*_baseline-repeat \
    --output profiling/results/noise_level.md
```

**Compare with the report:** Stage 1 results table: the baseline session's `summary.md`.
Noise table: `profiling/results/noise_level.md`.

## 4. Stage 2: workers, threads and concurrency (about 2 hours 15 minutes)

```bash
SCENARIOS="step length spike" bash profiling/run_experiments.sh \
  "oversubscribed-2x8|GUNICORN_WORKERS=2 APP_TORCH_NUM_THREADS=8" \
  "workers-1x8|GUNICORN_WORKERS=1 APP_TORCH_NUM_THREADS=8" \
  "workers-4x2|GUNICORN_WORKERS=4 APP_TORCH_NUM_THREADS=2" \
  "workers-8x1|GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1" \
  "workers-2x4-limit1|APP_MAX_CONCURRENT_PREDICTIONS=1" \
  "workers-8x1-limit1|GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=1" \
  "workers-8x1-limit2|GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=2" \
  "quiet-logs|APP_LOG_LEVEL=WARNING"
```

The script runs its own baseline first and writes `comparison.md` at the end. Note the batch
folder it prints (`profiling/results/experiments_<date-time>`) and set it for later steps:

```bash
S2=profiling/results/experiments_<date-time>
python profiling/worker_balance.py $S2/*_workers-4x2 --output profiling/results/worker_balance_4x2.md
python profiling/worker_balance.py $S2/*_workers-8x1 $S2/*_workers-8x1-limit1 $S2/*_workers-8x1-limit2 \
    --output profiling/results/worker_balance_8x1.md
```

**Compare with the report:** Stage 2 table: `$S2/comparison.md`. Per-worker CPU tables (Stage 2
and Section 4.5): `worker_balance_8x1.md` and `worker_balance_4x2.md`.

## 5. Stage 3: number formats (about 1 hour 30 minutes)

```bash
APP_TORCH_NUM_THREADS=1 taskset -c 0 python profiling/check_accuracy.py \
    --variants baseline quantized bf16 fp16

SCENARIOS="step length spike" bash profiling/run_experiments.sh \
  "fp32-8x1-limit2|GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=2" \
  "int8-8x1-limit2|APP_QUANTIZE=true GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=2" \
  "bf16-8x1-limit2|APP_PRECISION=bf16 GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=2" \
  "fp16-8x1-limit2|APP_PRECISION=fp16 GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=2"

S3=profiling/results/experiments_<date-time>
python profiling/compare_sessions.py $S3/*_fp32-8x1-limit2 $S3/*_int8-8x1-limit2 \
    $S3/*_bf16-8x1-limit2 $S3/*_fp16-8x1-limit2 --output $S3/format_comparison.md
```

The accuracy check downloads the SST-2 validation set the first time, so run it without
`HF_HUB_OFFLINE`.

**Compare with the report:** accuracy table: `profiling/results/accuracy_<date-time>.md`.
Format tables: `$S3/format_comparison.md` and each session's `summary.md`. "Peak" in the report
is the highest step value, wherever it occurs.

## 6. Stage 4: profiling (about 50 minutes)

CPU features:

```bash
grep -o -E 'avx512_bf16|avx512_fp16|avx512_vnni|avx512f' /proc/cpuinfo | sort -u \
    | tee profiling/results/cpu_features.txt
```

Stage timings (one prediction at a time, no server):

```bash
APP_TORCH_NUM_THREADS=4 taskset -c 0-3 python profiling/profile_stages.py --label fp32-4threads
APP_TORCH_NUM_THREADS=1 taskset -c 0 python profiling/profile_stages.py --label fp32-1thread
APP_TORCH_NUM_THREADS=1 APP_QUANTIZE=true taskset -c 0 python profiling/profile_stages.py --label int8-1thread
APP_TORCH_NUM_THREADS=1 APP_PRECISION=bf16 taskset -c 0 python profiling/profile_stages.py --label bf16-1thread
APP_TORCH_NUM_THREADS=1 APP_PRECISION=fp16 taskset -c 0 python profiling/profile_stages.py --label fp16-1thread
```

Flame graphs at 32 users (about 2 minutes each):

```bash
RATE=20 USERS=32 bash profiling/profile_server.sh baseline-32u
RATE=25 USERS=32 GUNICORN_WORKERS=2 APP_TORCH_NUM_THREADS=8 bash profiling/profile_server.sh oversubscribed-32u
RATE=20 USERS=32 GUNICORN_WORKERS=1 APP_TORCH_NUM_THREADS=8 bash profiling/profile_server.sh workers-1x8-32u
RATE=25 USERS=32 GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 bash profiling/profile_server.sh workers-8x1-32u
RATE=25 USERS=32 GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=1 \
    bash profiling/profile_server.sh workers-8x1-limit1-32u
RATE=25 USERS=32 GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=2 \
    bash profiling/profile_server.sh workers-8x1-limit2-32u
RATE=25 USERS=32 APP_QUANTIZE=true GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=2 \
    bash profiling/profile_server.sh int8-8x1-limit2-32u
RATE=25 USERS=32 APP_PRECISION=bf16 GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=2 \
    bash profiling/profile_server.sh bf16-8x1-limit2-32u
RATE=25 USERS=32 APP_PRECISION=fp16 GUNICORN_WORKERS=8 APP_TORCH_NUM_THREADS=1 APP_MAX_CONCURRENT_PREDICTIONS=2 \
    bash profiling/profile_server.sh fp16-8x1-limit2-32u
```

4×2, every worker separately, at a normal and the worst load:

```bash
TARGET=each RATE=25 USERS=4 GUNICORN_WORKERS=4 APP_TORCH_NUM_THREADS=2 bash profiling/profile_server.sh workers-4x2-4u
TARGET=each RATE=25 USERS=16 GUNICORN_WORKERS=4 APP_TORCH_NUM_THREADS=2 bash profiling/profile_server.sh workers-4x2-16u
```

After each recording, check for "behind in sampling" in the output; if it appears, rerun that
line with a lower `RATE` (for example 20 or 10). Then turn the graphs into numbers:

```bash
for f in profiling/results/*_flame_*-32u/flamegraph.svg; do
  echo "=================== $f"; python profiling/analyze_flamegraph.py "$f" --top 10
done > profiling/results/flamegraph_analysis.txt

for f in profiling/results/*_flame_workers-4x2-*/flamegraph*.svg; do
  echo "=================== $f"; python profiling/analyze_flamegraph.py "$f" --top 12
done > profiling/results/flamegraph_4x2.txt
```

**Compare with the report:** stage-timing tables: `profiling/results/stages_*.md`. Flame-graph
shares: open each `flamegraph.svg` in a browser, type the search term from the figure's caption in
the Search box (top right), and read the "Matched" percentage at the bottom right. Category shares:
`flamegraph_analysis.txt` and `flamegraph_4x2.txt`.

## 7. Stage 5: final configuration

The final comparison uses the `int8-8x1-limit2`, `bf16-8x1-limit2` and `baseline` sessions of the
Stage 3 batch, so no further run is needed. Optionally, to check the recommended profiles yourself
(these runs are not part of the report):

```bash
PROFILE=throughput bash scripts/smoke_test.sh
PROFILE=throughput LABEL=throughput-profile bash loadtests/run_load_tests.sh step length spike
PROFILE=accuracy LABEL=accuracy-profile bash loadtests/run_load_tests.sh step length spike
```

**Compare with the report:** Stage 5 table: the Stage 3 sessions' `summary.md` files, or the
profile sessions above.

## 8. Where each figure comes from

| Figures in the report | Source |
|---|---|
| Locust charts | the `report.html` of the run named in the caption; each chart's download icon (top right) saves it as an image |
| Flame graphs | the `flamegraph.svg` in the folder named in the caption, with the caption's search term |
| Stage 2 and Stage 3 summary charts | the `summary.csv` of each session in the batch |
| Stage 4 timing chart | the `stages_*.csv` files, batch size 1 |

Every session folder also records its hardware, packages, settings and code version in
`environment/`, and its startup time and idle memory in `startup.txt`.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: uvicorn_worker` or `transformers` | the project's virtual environment isn't active: `source .venv/bin/activate`, then `which python` |
| A script fails with "already answering at http://127.0.0.1:8000" | stop the server that's still running (Ctrl+C in its terminal) |
| `taskset: failed to set pid's affinity` | the machine has fewer CPUs than `SERVER_CPUS`/`LOCUST_CPUS` ask for; change them |
| py-spy: "No python processes found" or permission errors | the script handles `ptrace_scope=1`; for other settings run `sudo sysctl kernel.yama.ptrace_scope=0` (until reboot) |
| py-spy: "behind in sampling" | rerun with a lower `RATE`, or `NATIVE=0` (Python frames only) |
| py-spy crashes with a segmentation fault | use the default target (busiest worker) or `TARGET=each`, not `TARGET=all` |
| Model download fails during a test | run `python scripts/download_model.py` once with network access; tests and servers run offline |