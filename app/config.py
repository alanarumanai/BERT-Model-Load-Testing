"""Service settings, read from environment variables (prefix APP_) or a .env file."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        extra="ignore",
        # Allow field names starting with "model_" without Pydantic warnings.
        protected_namespaces=("settings_",),
    )

    # Which model to load. Set APP_MODEL_REVISION to the commit hash you pinned,
    # so the exact same weights are used on every machine.
    model_name: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    model_revision: str = "main"

    # Inputs longer than this many tokens are truncated. SST-2 sentences are short,
    # so 128 covers almost all real inputs while capping the cost of long ones.
    max_length: int = Field(default=128, ge=8, le=512)

    # Requests are rejected (422) before tokenization if they exceed these limits.
    max_text_chars: int = Field(default=5000, ge=1)
    max_batch_size: int = Field(default=32, ge=1)

    # PyTorch intra-op threads per worker process. None keeps PyTorch's default
    # (all cores), which oversubscribes the CPU when several workers run.
    torch_num_threads: int | None = Field(default=None, ge=1)

    # Phase 7 experiments, both off by default.
    # Convert the model's Linear layers to 8-bit integers (dynamic quantization).
    quantize: bool = False
    # Number format for the whole model: fp32 (default), bf16 or fp16. The 16-bit
    # formats halve the memory of every weight. Cannot be combined with quantize.
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    # At most this many predictions run at once in each worker; extra requests wait
    # their turn. None means no limit (FastAPI's thread pool runs them in parallel).
    max_concurrent_predictions: int | None = Field(default=None, ge=1)

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, created once per process."""
    return Settings()
