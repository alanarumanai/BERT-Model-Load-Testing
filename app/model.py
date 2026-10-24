"""Loads the pretrained model once and runs predictions. No web code here."""

import logging
import threading
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


def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    """Store the Linear layers' weights as 8-bit integers (dynamic quantization).

    Activations are quantized on the fly during inference. On CPUs this usually makes
    inference faster and the model smaller, at a small cost in accuracy.
    """
    from torch.ao.quantization import quantize_dynamic as _quantize

    return _quantize(model, {torch.nn.Linear}, dtype=torch.qint8)


class SentimentClassifier:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        max_length: int = 128,
        num_threads: int | None = None,
        quantize: bool = False,
        max_concurrent: int | None = None,
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

        if quantize:
            self.model = quantize_dynamic(self.model)
            self.model_id += "+int8"  # responses show which variant answered

        # Label names come from the model's own config, not hardcoded strings,
        # so any sequence-classification checkpoint works unchanged.
        config = self.model.config
        self.labels = [config.id2label[i] for i in range(config.num_labels)]

        # Optional limit on predictions running at once in this process.
        self._slots = threading.BoundedSemaphore(max_concurrent) if max_concurrent else None

        # The first forward pass is slow (memory allocation, kernel setup).
        # Pay that cost at startup instead of on the first real request.
        self.predict(["Warm-up sentence."])

        logger.info(
            "Loaded %s in %.1fs (labels=%s, torch threads=%d, max_length=%d, "
            "quantized=%s, max_concurrent=%s)",
            self.model_id,
            time.perf_counter() - started,
            self.labels,
            torch.get_num_threads(),
            self.max_length,
            quantize,
            max_concurrent,
        )

    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []
        if self._slots is None:
            return self._predict(texts)
        with self._slots:  # wait here if the limit is reached
            return self._predict(texts)

    def _predict(self, texts: list[str]) -> list[Prediction]:
        return self.postprocess(self.forward(self.tokenize(texts)))

    # The three stages are separate methods so profiling can time each one
    # using exactly the code the service runs.

    def tokenize(self, texts: list[str]):
        # padding=True pads only to the longest text in this batch,
        # not to max_length, so short inputs stay cheap.
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, inputs) -> torch.Tensor:
        # No gradients are needed for inference; this saves time and memory.
        with torch.inference_mode():
            return self.model(**inputs).logits

    def postprocess(self, logits: torch.Tensor) -> list[Prediction]:
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
