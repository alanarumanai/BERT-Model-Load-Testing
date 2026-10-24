"""Request and response shapes. FastAPI returns 422 for anything that breaks these rules."""

from pydantic import BaseModel, Field, field_validator

from app.config import get_settings


def _check_text(text: str) -> str:
    if not text.strip():
        raise ValueError("text must not be empty or only whitespace")
    limit = get_settings().max_text_chars
    if len(text) > limit:
        raise ValueError(f"text must be at most {limit} characters")
    return text


class PredictRequest(BaseModel):
    text: str = Field(..., examples=["This movie was wonderful."])

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _check_text(value)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, examples=[["Great acting.", "The plot made no sense."]]
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        limit = get_settings().max_batch_size
        if len(values) > limit:
            raise ValueError(f"at most {limit} texts per request")
        return [_check_text(value) for value in values]


class PredictionResult(BaseModel):
    label: str = Field(..., examples=["POSITIVE"])
    score: float = Field(..., ge=0.0, le=1.0, examples=[0.9998])
    scores: dict[str, float] = Field(..., examples=[{"NEGATIVE": 0.0002, "POSITIVE": 0.9998}])


class PredictResponse(PredictionResult):
    model: str


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResult]
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str | None = None
