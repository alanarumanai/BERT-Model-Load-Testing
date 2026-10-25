"""The real model wrapper: labels, predictions, probabilities, batching, truncation,
and the Phase 7 options (stages, concurrency limit, quantization)."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.config import get_settings
from app.model import SentimentClassifier

POSITIVE_TEXTS = [
    "This movie was absolutely wonderful, I loved every minute of it.",
    "What a fantastic, heartwarming film with brilliant acting.",
]
NEGATIVE_TEXTS = [
    "This was a terrible waste of time.",
    "The plot was boring and the acting was awful.",
]


def test_labels_come_from_model_config(classifier):
    config = classifier.model.config
    assert classifier.labels == [config.id2label[i] for i in range(config.num_labels)]
    assert set(classifier.labels) == {"NEGATIVE", "POSITIVE"}


def test_model_id_includes_name_and_revision(classifier):
    name, revision = classifier.model_id.rsplit("@", 1)
    assert name.endswith("distilbert-base-uncased-finetuned-sst-2-english")
    assert revision


@pytest.mark.parametrize("text", POSITIVE_TEXTS)
def test_clearly_positive_text(classifier, text):
    prediction = classifier.predict([text])[0]
    assert prediction.label == "POSITIVE"
    assert prediction.score > 0.9


@pytest.mark.parametrize("text", NEGATIVE_TEXTS)
def test_clearly_negative_text(classifier, text):
    prediction = classifier.predict([text])[0]
    assert prediction.label == "NEGATIVE"
    assert prediction.score > 0.9


def test_scores_are_valid_probabilities(classifier):
    prediction = classifier.predict(["An average film."])[0]
    assert set(prediction.scores) == set(classifier.labels)
    assert all(0.0 <= p <= 1.0 for p in prediction.scores.values())
    assert sum(prediction.scores.values()) == pytest.approx(1.0, abs=1e-6)
    # The reported label and score are the highest-scoring entry.
    assert prediction.score == max(prediction.scores.values())
    assert prediction.scores[prediction.label] == prediction.score


def test_empty_input_returns_empty_list(classifier):
    assert classifier.predict([]) == []


def test_batch_returns_one_result_per_text_in_order(classifier):
    texts = [POSITIVE_TEXTS[0], NEGATIVE_TEXTS[0], POSITIVE_TEXTS[1]]
    labels = [p.label for p in classifier.predict(texts)]
    assert labels == ["POSITIVE", "NEGATIVE", "POSITIVE"]


def test_padding_does_not_change_results(classifier):
    # In a batch, the short text is padded to the long one's length. The attention
    # mask must make the model ignore that padding, so its scores stay the same.
    short = "Great film."
    long = "The cinematography, the score and the performances were all superb. " * 5
    alone = classifier.predict([short])[0]
    in_batch = classifier.predict([short, long])[0]
    for label in classifier.labels:
        assert in_batch.scores[label] == pytest.approx(alone.scores[label], abs=1e-4)


def test_long_text_is_truncated_not_rejected(classifier):
    very_long = "This film was great. " * 1000  # thousands of tokens, far over the limit
    prediction = classifier.predict([very_long])[0]
    assert prediction.label in classifier.labels


def test_predictions_are_deterministic(classifier):
    # Evaluation mode disables dropout, so repeated calls must agree.
    first = classifier.predict([NEGATIVE_TEXTS[1]])[0]
    second = classifier.predict([NEGATIVE_TEXTS[1]])[0]
    for label in classifier.labels:
        assert second.scores[label] == pytest.approx(first.scores[label], abs=1e-6)


def test_stages_together_match_predict(classifier):
    # Profiling times tokenize, forward and postprocess separately; together they
    # must give exactly what predict() returns.
    text = POSITIVE_TEXTS[0]
    staged = classifier.postprocess(classifier.forward(classifier.tokenize([text])))[0]
    direct = classifier.predict([text])[0]
    assert staged.label == direct.label
    assert staged.score == pytest.approx(direct.score, abs=1e-6)


def _peak_concurrency(classifier, monkeypatch, slots):
    """Run 4 predictions from 4 threads and return how many ran at the same time."""
    active = peak = 0
    lock = threading.Lock()
    real_predict = classifier._predict

    def tracked(texts):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)  # make overlapping calls likely
        try:
            return real_predict(texts)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(classifier, "_predict", tracked)
    monkeypatch.setattr(classifier, "_slots", slots)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda text: classifier.predict([text]), POSITIVE_TEXTS * 2))
    assert len(results) == 4
    return peak


def test_without_limit_predictions_overlap(classifier, monkeypatch):
    # Shows the next test is meaningful: with no limit, calls do run concurrently.
    assert _peak_concurrency(classifier, monkeypatch, slots=None) > 1


def test_concurrency_limit_runs_one_prediction_at_a_time(classifier, monkeypatch):
    assert _peak_concurrency(classifier, monkeypatch, threading.BoundedSemaphore(1)) == 1


@pytest.fixture(scope="module")
def quantized_classifier():
    settings = get_settings()
    return SentimentClassifier(
        model_name=settings.model_name,
        revision=settings.model_revision,
        max_length=settings.max_length,
        quantize=True,
    )


def test_quantized_model_uses_int8_layers(quantized_classifier):
    assert quantized_classifier.model_id.endswith("+int8")
    quantized_layers = [
        m for m in quantized_classifier.model.modules() if "quantized" in type(m).__module__
    ]
    assert quantized_layers, "no quantized layers found"


@pytest.mark.parametrize(
    ("text", "label"),
    [(t, "POSITIVE") for t in POSITIVE_TEXTS] + [(t, "NEGATIVE") for t in NEGATIVE_TEXTS],
)
def test_quantized_model_still_classifies_correctly(quantized_classifier, text, label):
    assert quantized_classifier.predict([text])[0].label == label


def test_quantize_and_16_bit_precision_cannot_be_combined():
    # Checked before anything is loaded, so this is instant.
    with pytest.raises(ValueError, match="cannot be combined"):
        SentimentClassifier(model_name="unused", quantize=True, precision="bf16")


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="precision must be one of"):
        SentimentClassifier(model_name="unused", precision="int4")


@pytest.fixture(scope="module", params=["bf16", "fp16"])
def half_precision_classifier(request):
    settings = get_settings()
    return SentimentClassifier(
        model_name=settings.model_name,
        revision=settings.model_revision,
        max_length=settings.max_length,
        precision=request.param,
    )


def test_half_precision_model_uses_16_bit_weights(half_precision_classifier):
    import torch

    precision = half_precision_classifier.precision
    expected = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    assert half_precision_classifier.model_id.endswith(f"+{precision}")
    assert {p.dtype for p in half_precision_classifier.model.parameters()} == {expected}


def test_half_precision_model_classifies_correctly(half_precision_classifier):
    texts = POSITIVE_TEXTS + NEGATIVE_TEXTS
    expected = ["POSITIVE"] * len(POSITIVE_TEXTS) + ["NEGATIVE"] * len(NEGATIVE_TEXTS)
    predictions = half_precision_classifier.predict(texts)
    assert [p.label for p in predictions] == expected
    # Scores are computed in FP32, so they are still valid probabilities.
    for p in predictions:
        assert sum(p.scores.values()) == pytest.approx(1.0, abs=1e-6)
