"""The real model wrapper: labels, predictions, probabilities, batching, truncation."""

import pytest

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
