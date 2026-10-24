"""The HTTP API: successful requests, validation, health, errors and headers."""

import pytest

from app.config import get_settings

POSITIVE = "This movie was absolutely wonderful, I loved every minute of it."
NEGATIVE = "This was a terrible waste of time."


# --- /predict ---------------------------------------------------------------------


def test_predict_positive(client, classifier):
    response = client.post("/predict", json={"text": POSITIVE})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"label", "score", "scores", "model"}
    assert body["label"] == "POSITIVE"
    assert 0.5 < body["score"] <= 1.0
    assert set(body["scores"]) == {"NEGATIVE", "POSITIVE"}
    assert body["model"] == classifier.model_id


def test_predict_negative(client):
    response = client.post("/predict", json={"text": NEGATIVE})
    assert response.status_code == 200
    assert response.json()["label"] == "NEGATIVE"


@pytest.mark.parametrize(
    "body",
    [
        {},  # missing field
        {"text": ""},  # empty
        {"text": "   \n\t "},  # only whitespace
        {"text": 123},  # wrong type
        {"text": None},
        {"text": ["a list", "not a string"]},
        {"wrong_field": "hello"},
    ],
)
def test_predict_rejects_invalid_body(client, body):
    response = client.post("/predict", json=body)
    assert response.status_code == 422


def test_predict_rejects_non_json(client):
    response = client.post(
        "/predict", content="not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_whitespace_error_message_is_clear(client):
    response = client.post("/predict", json={"text": "   "})
    assert "whitespace" in str(response.json()["detail"])


def test_text_length_limit_boundary(client):
    limit = get_settings().max_text_chars
    at_limit = ("great movie " * limit)[:limit]

    assert client.post("/predict", json={"text": at_limit}).status_code == 200
    assert client.post("/predict", json={"text": at_limit + "x"}).status_code == 422


def test_text_length_limit_follows_settings(client, override_settings):
    override_settings(max_text_chars=10)
    assert client.post("/predict", json={"text": "short"}).status_code == 200
    assert client.post("/predict", json={"text": "much too long"}).status_code == 422


# --- /predict/batch ---------------------------------------------------------------


def test_batch_returns_results_in_order(client, classifier):
    texts = [POSITIVE, NEGATIVE, POSITIVE]
    response = client.post("/predict/batch", json={"texts": texts})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == classifier.model_id
    assert [p["label"] for p in body["predictions"]] == ["POSITIVE", "NEGATIVE", "POSITIVE"]


def test_batch_matches_single_predictions(client):
    single = client.post("/predict", json={"text": NEGATIVE}).json()
    batch = client.post("/predict/batch", json={"texts": [NEGATIVE]}).json()
    assert batch["predictions"][0]["label"] == single["label"]
    assert batch["predictions"][0]["score"] == pytest.approx(single["score"], abs=1e-4)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"texts": []},  # empty list
        {"texts": "not a list"},
        {"texts": ["fine", ""]},  # one bad item rejects the whole batch
        {"texts": ["fine", "   "]},
        {"texts": ["fine", 5]},
    ],
)
def test_batch_rejects_invalid_body(client, body):
    assert client.post("/predict/batch", json=body).status_code == 422


def test_batch_size_limit(client, override_settings):
    override_settings(max_batch_size=2)
    assert client.post("/predict/batch", json={"texts": ["a", "b"]}).status_code == 200
    assert client.post("/predict/batch", json={"texts": ["a", "b", "c"]}).status_code == 422


# --- /health and the "model not loaded" state ---------------------------------------


def test_health_when_model_loaded(client, classifier):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model": classifier.model_id}


def test_not_ready_without_model(client, monkeypatch):
    # Simulate the moment before the lifespan has finished loading the model.
    monkeypatch.setattr(client.app.state, "classifier", None)

    health = client.get("/health")
    assert health.status_code == 503
    assert health.json() == {"status": "loading", "model_loaded": False, "model": None}

    assert client.post("/predict", json={"text": POSITIVE}).status_code == 503
    assert client.post("/predict/batch", json={"texts": [POSITIVE]}).status_code == 503


# --- Errors, headers and docs -----------------------------------------------------


def test_unexpected_error_returns_generic_500(client, classifier, monkeypatch):
    def broken_predict(texts):
        raise RuntimeError("secret internal detail")

    # Replace predict on the loaded model for this test only; monkeypatch restores it.
    monkeypatch.setattr(classifier, "predict", broken_predict)

    response = client.post("/predict", json={"text": POSITIVE})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret" not in response.text  # internal details never reach the client


def test_process_time_header(client):
    response = client.post("/predict", json={"text": POSITIVE})
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_unknown_path_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_wrong_method_returns_405(client):
    assert client.get("/predict").status_code == 405


def test_openapi_docs_are_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/predict", "/predict/batch", "/health"} <= set(paths)
