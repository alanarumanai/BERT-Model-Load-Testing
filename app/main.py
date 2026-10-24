"""FastAPI application: endpoints, startup, logging and error handling."""

import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.config import Settings, get_settings
from app.interfaces import Classifier, Prediction
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictionResult,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger("app")

ClassifierFactory = Callable[[Settings], Classifier]


def load_sentiment_classifier(settings: Settings) -> Classifier:
    """Default factory. Imported lazily so tests with a fake never import PyTorch."""
    from app.model import SentimentClassifier

    return SentimentClassifier(
        model_name=settings.model_name,
        revision=settings.model_revision,
        max_length=settings.max_length,
        num_threads=settings.torch_num_threads,
    )


def get_classifier(request: Request) -> Classifier:
    """Endpoint dependency. Tests can replace it via app.dependency_overrides."""
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return classifier


ClassifierDep = Annotated[Classifier, Depends(get_classifier)]


def _to_result(prediction: Prediction) -> PredictionResult:
    return PredictionResult(
        label=prediction.label, score=prediction.score, scores=prediction.scores
    )


def create_app(classifier_factory: ClassifierFactory = load_sentiment_classifier) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        logging.basicConfig(
            level=settings.log_level.upper(),
            format="%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s",
        )
        # The Hugging Face downloader logs every HTTP call at INFO; keep it quiet.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        app.state.classifier = None
        # Runs once per worker process, before it accepts any request.
        app.state.classifier = classifier_factory(settings)
        yield
        app.state.classifier = None

    app = FastAPI(
        title="Sentiment Classification Service",
        description="Classifies English text as POSITIVE or NEGATIVE using DistilBERT fine-tuned on SST-2.",
        version=__version__,
        lifespan=lifespan,
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
    def predict(body: PredictRequest, classifier: ClassifierDep) -> PredictResponse:
        prediction = classifier.predict([body.text])[0]
        return PredictResponse(**_to_result(prediction).model_dump(), model=classifier.model_id)

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(body: BatchPredictRequest, classifier: ClassifierDep) -> BatchPredictResponse:
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

    return app


# Entry point for uvicorn and Gunicorn: "app.main:app"
app = create_app()
