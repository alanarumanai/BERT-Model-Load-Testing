"""What the API expects from a classifier.

Kept separate from model.py so the API and its tests never need to import
PyTorch: any object with these attributes works, including a test fake.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    scores: dict[str, float]


class Classifier(Protocol):
    model_id: str
    labels: list[str]

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Classify each text, returning one Prediction per input, in order."""
        ...
