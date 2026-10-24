"""Shared test fixtures. Pytest loads this file automatically."""

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture(scope="session")
def client():
    """A test client for the whole session.

    Entering the `with` block runs the app's lifespan, which loads the real model.
    scope="session" means this happens once for the entire test run, not per test.
    raise_server_exceptions=False lets tests see the 500 response the app sends,
    instead of the exception being re-raised inside the test.
    """
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def classifier(client):
    """The real SentimentClassifier the app loaded, reused by the model tests."""
    return client.app.state.classifier


@pytest.fixture
def override_settings(monkeypatch):
    """Change settings for one test, e.g. override_settings(max_batch_size=2).

    Sets the matching APP_* environment variable and clears the settings cache so
    the next get_settings() call reads it. After the test, monkeypatch restores the
    environment and the cache is cleared again, so other tests see normal settings.
    """

    def _override(**values):
        for name, value in values.items():
            monkeypatch.setenv(f"APP_{name.upper()}", str(value))
        get_settings.cache_clear()

    yield _override
    get_settings.cache_clear()
