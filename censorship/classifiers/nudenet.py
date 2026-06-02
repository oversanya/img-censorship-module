"""NudeNet detector wrapper (CPU-fast NSFW classifier)."""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Union

from censorship.core.verdict import ClassifierResult
from .base import ImageClassifier

logger = logging.getLogger(__name__)

# NudeNet classes that map to sexual_explicit
_EXPLICIT_CLASSES = {
    "EXPOSED_ANUS",
    "EXPOSED_BUTTOCKS",
    "EXPOSED_BREAST_F",
    "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M",
    "EXPOSED_BELLY",       # partial — lower weight
}

_HIGH_WEIGHT_CLASSES = {
    "EXPOSED_ANUS",
    "EXPOSED_BREAST_F",
    "EXPOSED_GENITALIA_F",
    "EXPOSED_GENITALIA_M",
}


class NudeNetClassifier(ImageClassifier):
    model_name = "nudenet"
    supported_categories = ["sexual_explicit"]

    def __init__(self):
        self._detector = None

    def load(self) -> None:
        from nudenet import NudeDetector
        logger.info("Loading NudeNet detector ...")
        self._detector = NudeDetector()
        logger.info("NudeNet loaded.")

    def is_loaded(self) -> bool:
        return self._detector is not None

    def classify(self, image_path: Union[str, Path]) -> ClassifierResult:
        if not self.is_loaded():
            self.load()

        t0 = time.perf_counter()
        detections = self._detector.detect(str(image_path))

        # Aggregate detections into a single sexual_explicit score
        # Weight high-severity classes more heavily
        max_score = 0.0
        for det in detections:
            class_name = det.get("class", "")
            score = det.get("score", 0.0)
            if class_name in _HIGH_WEIGHT_CLASSES:
                max_score = max(max_score, score)
            elif class_name in _EXPLICIT_CLASSES:
                # Lower-weight classes contribute at 70%
                max_score = max(max_score, score * 0.7)

        latency = (time.perf_counter() - t0) * 1000
        scores = {"sexual_explicit": max_score}
        triggered = ["sexual_explicit"] if max_score >= 0.50 else []

        return ClassifierResult(
            model=self.model_name,
            scores=scores,
            is_unsafe=max_score >= 0.50,
            triggered_categories=triggered,
            latency_ms=latency,
            metadata={"raw_detections": detections},
        )
