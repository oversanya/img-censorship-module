"""Q16 CLIP-based NSFW classifier wrapper (LAION-AI)."""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Union

from censorship.core.verdict import ClassifierResult
from .base import ImageClassifier

logger = logging.getLogger(__name__)

# Two known HF IDs for Q16-style models — try in order
_HF_IDS = [
    "LAION-AI/CLIP-based-NSFW-Detector",
    "Falconsai/nsfw_image_detection",   # fallback: well-known NSFW ViT
]


class Q16Classifier(ImageClassifier):
    """
    CLIP/ViT-based NSFW classifier.

    Tries LAION-AI Q16 first; falls back to Falconsai NSFW detector.
    Maps the single NSFW probability to multiple taxonomy categories
    with different weights (explicit > violence > hate).
    """

    model_name = "q16"
    supported_categories = ["sexual_explicit", "violence_gore", "hate_speech"]

    def __init__(self, hf_id: str | None = None, hf_token: str | None = None):
        self.hf_id = hf_id
        self.hf_token = hf_token
        self._pipeline = None
        self._resolved_id: str | None = None

    def load(self) -> None:
        from transformers import pipeline as hf_pipeline

        ids_to_try = [self.hf_id] if self.hf_id else _HF_IDS
        for hf_id in ids_to_try:
            try:
                logger.info(f"Trying to load Q16-style model: {hf_id}")
                self._pipeline = hf_pipeline(
                    "image-classification",
                    model=hf_id,
                    token=self.hf_token,
                )
                self._resolved_id = hf_id
                logger.info(f"Q16 classifier loaded: {hf_id}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {hf_id}: {e}")

        raise RuntimeError(
            "Could not load any Q16-style NSFW classifier. "
            f"Tried: {ids_to_try}"
        )

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def classify(self, image_path: Union[str, Path]) -> ClassifierResult:
        if not self.is_loaded():
            self.load()

        t0 = time.perf_counter()
        results = self._pipeline(str(image_path))

        # Extract NSFW probability
        nsfw_score = 0.0
        for r in results:
            label = r["label"].lower()
            if "nsfw" in label or "unsafe" in label or "explicit" in label:
                nsfw_score = max(nsfw_score, r["score"])

        # Map single NSFW score to categories with differentiated weights.
        # NSFW classifiers are strongest for sexual content.
        scores = {
            "sexual_explicit": nsfw_score,
            "violence_gore": nsfw_score * 0.5,
            "hate_speech": nsfw_score * 0.3,
        }

        latency = (time.perf_counter() - t0) * 1000
        triggered = [c for c, s in scores.items() if s >= 0.50]

        return ClassifierResult(
            model=f"q16/{self._resolved_id}",
            scores=scores,
            is_unsafe=nsfw_score >= 0.50,
            triggered_categories=triggered,
            latency_ms=latency,
            metadata={"raw_results": results},
        )
