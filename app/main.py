"""FastAPI application: endpoints, startup, logging and error handling."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.model import Prediction, SentimentClassifier
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictionResult,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s",
    )
    # HTTP client libraries log every request at INFO: the Hugging Face downloader
    # uses httpx, and FastAPI's TestClient uses httpx2. Keep them quiet.
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    app.state.classifier = None
    # Runs once per worker process, before it accepts any request.
    app.state.classifier = SentimentClassifier(
        model_name=settings.model_name,
        revision=settings.model_revision,
        max_length=settings.max_length,
        num_threads=settings.torch_num_threads,
        quantize=settings.quantize,
        max_concurrent=settings.max_concurrent_predictions,
    )
    yield
    app.state.classifier = None


app = FastAPI(
    title="Sentiment Classification Service",
    description=(
        "Classifies English text as POSITIVE or NEGATIVE using DistilBERT fine-tuned on SST-2."
    ),
    version=__version__,
    lifespan=lifespan,
)


def get_classifier(request: Request) -> SentimentClassifier:
    """Return the loaded model, or answer 503 if it isn't ready yet."""
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return classifier


def _to_result(prediction: Prediction) -> PredictionResult:
    return PredictionResult(
        label=prediction.label, score=prediction.score, scores=prediction.scores
    )


@app.middleware("http")
async def time_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "%s %s %d %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    # Log the full traceback, but never send it to the client.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Plain `def`, not `async def`: inference is CPU-bound, so FastAPI runs these
# in its thread pool and the event loop stays free for other requests.
@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    classifier = get_classifier(request)
    prediction = classifier.predict([body.text])[0]
    return PredictResponse(**_to_result(prediction).model_dump(), model=classifier.model_id)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    classifier = get_classifier(request)
    predictions = classifier.predict(body.texts)
    return BatchPredictResponse(
        predictions=[_to_result(p) for p in predictions], model=classifier.model_id
    )


# Cheap, so async is fine here. Returns 503 until the model is ready.
@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(status="loading", model_loaded=False).model_dump(),
        )
    return HealthResponse(status="ok", model_loaded=True, model=classifier.model_id)