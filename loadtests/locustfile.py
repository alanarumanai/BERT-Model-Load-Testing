"""Locust load test for the sentiment service.

Each simulated user sends POST /predict requests back to back with no pause, so the
number of users equals the number of requests in flight at any moment.

Settings (environment variables, all optional):
  LOAD_TEXT_MIX   mixed (default), short, medium or long
  LOAD_TIMEOUT    seconds before a request is counted as failed (default 30)
  LOAD_SEED       seed for choosing texts, so runs send the same sequence (default 42)
  LOAD_SHAPE      "spike" to replace -u/-r/-t with the spike pattern defined below
  SPIKE_BASE_USERS, SPIKE_PEAK_USERS, SPIKE_STAGE_SECONDS   spike pattern settings

Run directly, for example:
  locust -f loadtests/locustfile.py --headless --host http://127.0.0.1:8000 -u 8 -r 8 -t 2m
"""

import itertools
import os
import random

from locust import FastHttpUser, LoadTestShape, constant, task
from texts import TEXTS  # Locust adds this file's folder to the import path

TEXT_MIX = os.getenv("LOAD_TEXT_MIX", "mixed")
TIMEOUT = float(os.getenv("LOAD_TIMEOUT", "30"))
SEED = int(os.getenv("LOAD_SEED", "42"))

# How often each length is chosen. "mixed" leans towards short texts, as real
# review-style traffic usually does, while still including long, expensive ones.
MIXES = {
    "mixed": {"short": 5, "medium": 3, "long": 2},
    "short": {"short": 1},
    "medium": {"medium": 1},
    "long": {"long": 1},
}
if TEXT_MIX not in MIXES:
    raise ValueError(f"LOAD_TEXT_MIX must be one of {sorted(MIXES)}, not {TEXT_MIX!r}")

_user_numbers = itertools.count()


class PredictUser(FastHttpUser):
    # No pause between requests: users = concurrency.
    wait_time = constant(0)
    # A request with no answer within TIMEOUT seconds is recorded as a failure.
    connection_timeout = TIMEOUT
    network_timeout = TIMEOUT

    def on_start(self):
        # Each user gets its own seeded random generator, so a given run always sends
        # the same sequence of texts and results can be compared fairly.
        self.rng = random.Random(SEED + next(_user_numbers))
        self.lengths, self.weights = zip(*MIXES[TEXT_MIX].items(), strict=True)

    @task
    def predict(self):
        length = self.rng.choices(self.lengths, self.weights)[0]
        text = self.rng.choice(TEXTS[length])

        # name= groups results by text length in Locust's statistics.
        # catch_response=True lets us decide what counts as success.
        with self.client.post(
            "/predict", json={"text": text}, name=f"/predict [{length}]", catch_response=True
        ) as response:
            if response.status_code == 0:
                # No HTTP response at all: a timeout or a refused/broken connection.
                response.failure(f"no response: {response.error!r}")
                return
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                label = response.json().get("label")
            except ValueError:
                response.failure("response is not valid JSON")
                return
            if label not in ("POSITIVE", "NEGATIVE"):
                response.failure(f"unexpected label: {label!r}")


# The spike pattern is only defined when requested, because Locust uses any load
# shape class it finds in the locustfile and would otherwise ignore -u, -r and -t.
if os.getenv("LOAD_SHAPE") == "spike":
    BASE = int(os.getenv("SPIKE_BASE_USERS", "4"))
    PEAK = int(os.getenv("SPIKE_PEAK_USERS", "64"))
    STAGE = int(os.getenv("SPIKE_STAGE_SECONDS", "60"))

    class SpikeShape(LoadTestShape):
        """BASE users, then a sudden jump to PEAK, then back to BASE; STAGE seconds each."""

        def tick(self):
            elapsed = self.get_run_time()
            if elapsed < STAGE:
                return BASE, BASE
            if elapsed < 2 * STAGE:
                return PEAK, PEAK  # spawn every extra user within one second
            if elapsed < 3 * STAGE:
                return BASE, PEAK  # and remove them just as fast
            return None  # end of test
