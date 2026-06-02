"""Abstract base class for Layer-1 image classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from censorship.core.verdict import ClassifierResult


class ImageClassifier(ABC):
    """
    Layer 1 — fast classifier.
    Returns confidence scores per category without extended rationale.
    """

    model_name: str
    supported_categories: list[str]

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def classify(self, image_path: Union[str, Path]) -> ClassifierResult:
        """
        Classify a single image.

        Args:
            image_path: path to image file

        Returns:
            ClassifierResult with scores per supported category
        """
        ...

    def classify_batch(self, image_paths: list[Union[str, Path]]) -> list[ClassifierResult]:
        return [self.classify(p) for p in image_paths]

    def is_loaded(self) -> bool:
        return False
