# Design decisions

Each decision, the reason for it, and where the evidence is. Section numbers refer
to report/REPORT.md.

## Service

| Decision | Reason | Evidence |
|---|---|---|
| DistilBERT fine-tuned on SST-2, revision pinned | Already fine-tuned (no training), ~40% smaller than BERT-base, pinned so the weights never change | model card; `.env` |
| Plain `def` endpoints, not `async def` | Inference is CPU-bound; `def` runs in the thread pool and keeps the event loop free | Stage 4, reading guide |
| Model loaded once per worker in the lifespan, with a warm-up call | Loading per request would cost seconds; the warm-up moves the slow first call to startup | `startup.txt` in every session |
| Gunicorn with `uvicorn_worker.UvicornWorker` | Gunicorn manages processes; Uvicorn runs the ASGI app; the worker class moved out of Uvicorn into its own package | `gunicorn.conf.py` |
| Torch threads = CPUs ÷ workers by default | Avoids thread oversubscription | Stage 2 (2×8 −20%), 4.3 |
| 120 s worker timeout | Workers report in only after model loading; the 30 s default risked a restart loop | `gunicorn.conf.py` |
| Concurrency limit as an option (`APP_MAX_CONCURRENT_PREDICTIONS`) | Removes GIL contention between requests in one worker; useful with single-threaded workers | 4.4 |
| Precision and quantization as options, off by default | Lets every format be measured on the same code; defaults make no assumption about CPU features | Stage 3 |

## Configuration

| Decision | Reason | Evidence |
|---|---|---|
| Code defaults stay FP32, 2 workers × 4 threads | Lowest memory (2.2 GB peak), no dependency on BF16/VNNI support, accurate | Stage 1 |
| `profiles/throughput.env`: int8, 8 × 1, limit 2 | 3.8× peak, p95 −69% at 64 users, −1.15 accuracy points | Stage 5 |
| `profiles/accuracy.env`: BF16, 8 × 1, limit 2 | 2.2× peak, accuracy unchanged; needs BF16 hardware support | Stage 5, 4.6 |
| FP16 not offered as a profile | 2.6× slower than FP32 on CPUs without AVX-512 FP16 | 3.3, 4.6 |

## Testing and measurement

| Decision | Reason | Evidence |
|---|---|---|
| Real model in the test suite, no fake | Small enough to load once per session; tests the real behaviour | `tests/` |
| Service and Locust pinned to separate CPUs | The load generator must not take CPU from the service | Stage 0 |
| Locust users with no wait time | Users then equal requests in flight, so results are easy to read | Stage 0 |
| Fresh server and baseline in every batch | Comparisons made under the same machine conditions | Stages 2–3 |
| One change at a time | A change in results can be attributed to one cause | Stage 2 |
| Thresholds: 5% throughput, 10–15% p50, 25–30% p95/p99 | Measured run-to-run variation of identical runs | 1.6 |

## Known limitations

- PyTorch dynamic quantization relies on quantized tensor types that PyTorch has deprecated.
- Uneven connection-to-worker distribution at low concurrency is partly a Locust artefact.
- Accuracy measured on SST-2 only (short sentences).