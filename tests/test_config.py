"""Settings: defaults, environment variables, .env files and validation.

These tests construct Settings directly and never load the model.
"""

import os

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove any APP_* variables from the shell so each test starts from defaults."""
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name)


def make_settings(**kwargs) -> Settings:
    # _env_file=None ignores the project's .env, so local settings can't affect results.
    return Settings(_env_file=None, **kwargs)


def test_defaults():
    settings = make_settings()
    assert settings.model_name == "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    assert settings.model_revision == "main"
    assert settings.max_length == 128
    assert settings.max_text_chars == 5000
    assert settings.max_batch_size == 32
    assert settings.torch_num_threads is None
    assert settings.log_level == "INFO"
    assert settings.quantize is False
    assert settings.max_concurrent_predictions is None
    assert settings.precision == "fp32"


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("APP_MAX_LENGTH", "256")
    monkeypatch.setenv("APP_TORCH_NUM_THREADS", "2")
    monkeypatch.setenv("APP_MODEL_REVISION", "abc123")

    settings = make_settings()

    assert settings.max_length == 256  # converted from the string "256"
    assert settings.torch_num_threads == 2
    assert settings.model_revision == "abc123"


def test_experiment_settings_from_environment(monkeypatch):
    monkeypatch.setenv("APP_QUANTIZE", "true")
    monkeypatch.setenv("APP_MAX_CONCURRENT_PREDICTIONS", "1")
    monkeypatch.setenv("APP_PRECISION", "bf16")

    settings = make_settings()

    assert settings.quantize is True  # the text "true" becomes a real boolean
    assert settings.max_concurrent_predictions == 1
    assert settings.precision == "bf16"


def test_reads_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("APP_MAX_BATCH_SIZE=8\nUNRELATED_VARIABLE=ignored\n")

    settings = Settings(_env_file=env_file)

    assert settings.max_batch_size == 8  # and the unrelated variable caused no error


def test_environment_variable_beats_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("APP_MAX_BATCH_SIZE=8\n")
    monkeypatch.setenv("APP_MAX_BATCH_SIZE", "16")

    assert Settings(_env_file=env_file).max_batch_size == 16


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("APP_MAX_LENGTH", "9999"),  # above BERT's 512-token limit
        ("APP_MAX_LENGTH", "4"),  # below the minimum of 8
        ("APP_MAX_BATCH_SIZE", "0"),
        ("APP_MAX_TEXT_CHARS", "0"),
        ("APP_TORCH_NUM_THREADS", "0"),
        ("APP_MAX_LENGTH", "not-a-number"),
        ("APP_MAX_CONCURRENT_PREDICTIONS", "0"),
        ("APP_QUANTIZE", "maybe"),
        ("APP_PRECISION", "int4"),
    ],
)
def test_invalid_values_are_rejected(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError):
        make_settings()


def test_get_settings_is_cached():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()