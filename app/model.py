"""Loads the pretrained model once and runs predictions. No web code here."""

import logging
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    scores: dict[str, float]


class SentimentClassifier:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        max_length: int = 128,
        num_threads: int | None = None,
    ) -> None:
        if num_threads is not None:
            torch.set_num_threads(num_threads)

        started = time.perf_counter()
        self.max_length = max_length
        self.model_id = f"{model_name}@{revision}"

        # use_fast selects the Rust tokenizer, much faster than the Python one.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision
        )
        self.model.eval()  # inference behaviour: disables dropout

        # Label names come from the model's own config, not hardcoded strings,
        # so any sequence-classification checkpoint works unchanged.
        config = self.model.config
        self.labels = [config.id2label[i] for i in range(config.num_labels)]

        # The first forward pass is slow (memory allocation, kernel setup).
        # Pay that cost at startup instead of on the first real request.
        self.predict(["Warm-up sentence."])

        logger.info(
            "Loaded %s in %.1fs (labels=%s, torch threads=%d, max_length=%d)",
            self.model_id,
            time.perf_counter() - started,
            self.labels,
            torch.get_num_threads(),
            self.max_length,
        )

    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []

        # padding=True pads only to the longest text in this batch,
        # not to max_length, so short inputs stay cheap.
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # No gradients are needed for inference; this saves time and memory.
        with torch.inference_mode():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1).tolist()

        predictions = []
        for row in probabilities:
            best = max(range(len(row)), key=row.__getitem__)
            predictions.append(
                Prediction(
                    label=self.labels[best],
                    score=row[best],
                    scores=dict(zip(self.labels, row, strict=True)),
                )
            )
        return predictions
