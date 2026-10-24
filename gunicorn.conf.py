"""Gunicorn configuration. Every value can be overridden with an environment variable.

Run with:  gunicorn -c gunicorn.conf.py app.main:app
"""

import os


def _cpu_count() -> int:
    # sched_getaffinity counts only the CPUs this process may use, so it respects
    # pinning with `taskset -c 0-3 gunicorn ...`. It does not see CPU quotas
    # (e.g. systemd CPUQuota); set GUNICORN_WORKERS and APP_TORCH_NUM_THREADS
    # explicitly in that case.
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS
        return os.cpu_count() or 1


CPUS = _cpu_count()

# --- Server socket ------------------------------------------------------------
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# --- Worker processes -----------------------------------------------------------
# Each worker is a separate process holding its own copy of the model (~250-400 MB).
# The usual web formula (2 x cores + 1) suits I/O-bound apps, not CPU-bound
# inference, so start small and tune with load tests in Phases 6 and 7.
workers = int(os.getenv("GUNICORN_WORKERS", str(min(2, CPUS))))

# Gunicorn itself speaks WSGI; this worker class runs the ASGI app (FastAPI) under Uvicorn.
worker_class = "uvicorn_worker.UvicornWorker"

# --- Avoid CPU oversubscription -------------------------------------------------
# By default PyTorch uses one thread per core in EVERY worker, so N workers fight
# over the same cores. Give each worker an equal share unless set explicitly.
# This runs in the master before workers are forked, and workers inherit it.
os.environ.setdefault("APP_TORCH_NUM_THREADS", str(max(1, CPUS // workers)))
# The Rust tokenizer has its own thread pool; keep it off so it doesn't compete too.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- Timeouts --------------------------------------------------------------------
# A worker that stops reporting in for this many seconds is killed and restarted.
# Model loading happens before the worker starts reporting, so this must be longer
# than the load time, or workers get killed during startup in a loop.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
# On shutdown or restart, give in-flight requests this long to finish.
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
# How long to keep idle client connections open for reuse.
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# --- Worker recycling --------------------------------------------------------------
# Restarting workers after N requests guards against memory leaks, but every restart
# reloads the model and shows up as a latency spike, so it's off by default (0).
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "0"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "0"))

# Load the app separately in each worker after forking. Preloading wouldn't share the
# model anyway (FastAPI's lifespan runs per worker), and starting PyTorch's thread
# pools before forking can deadlock the child processes.
preload_app = False

# --- Logging ---------------------------------------------------------------------
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
errorlog = "-"  # stderr
# The app's middleware already logs every request with its duration, so Gunicorn's
# access log is off by default. Set GUNICORN_ACCESS_LOG=- to turn it on.
accesslog = os.getenv("GUNICORN_ACCESS_LOG") or None

# The heartbeat file workers touch to show they're alive. Keeping it in memory
# (/dev/shm) avoids stalls when the disk is slow or busy.
worker_tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None


def on_starting(server):
    server.log.info(
        "Starting %d worker(s), %s torch thread(s) each, on %d CPU(s)",
        workers,
        os.environ["APP_TORCH_NUM_THREADS"],
        CPUS,
    )
