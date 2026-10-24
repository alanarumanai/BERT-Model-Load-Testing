# Profiling findings

Fill this in as you go. Every number should point to a file in `profiling/results/`
or `loadtests/results/`, so a reader can check it.

**Setup:** service pinned to CPUs `<0-7>`, Locust on `<8-9>`, baseline 2 workers x
`<4>` torch threads, max_length 128. Experiments: step load at `<1, 8, 32>` users,
`<90s>` per level. Results folder: `profiling/results/experiments_<...>/`.

## Summary

| # | Investigation | Evidence | Change | Before | After | Decision |
|---|---|---|---|---|---|---|
| 1 | Where request time goes | stage timings, flame graph | none (diagnosis) | | | |
| 2 | Worker / thread split | step load | | | | |
| 3 | Thread oversubscription | step load, flame graph | | | | |
| 4 | Competing inferences in a worker | step load | | | | |
| 5 | Input length / max_length | length runs, accuracy check | | | | |
| 6 | Dynamic quantization | step load, accuracy check | | | | |
| 7 | Per-request logging | step load | | | | |
| 8 | Combined best configuration | full step load vs Phase 6 baseline | | | | |

## 1. Where request time goes

- **Question:**
- **Evidence:** `stages_baseline_<...>.md`, `<...>_flame_baseline/flamegraph.svg`
- **What it showed:**
- **Conclusion:**

## 2. Worker / thread split

- **Hypothesis:**
- **Results:** (peak req/s and p95 at 1, 8 and 32 users for 1xN, 2xN/2, 4xN/4, Nx1)
- **Run-to-run variation:** (difference between repeated baseline runs)
- **Conclusion and decision:**

## 3. Thread oversubscription

## 4. Competing inferences in a worker

## 5. Input length and max_length

## 6. Dynamic quantization

- **Speed:** (from comparison.md)
- **Accuracy:** (from accuracy_<...>.md: accuracy, agreement with baseline)
- **Decision:**

## 7. Per-request logging

## 8. Combined best configuration

## Ideas not implemented, and why

- Dynamic batching:
- ONNX Runtime:
- GPU:
