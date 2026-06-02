from __future__ import annotations

import os
from pathlib import Path

from censor_guard.schemas import SignalResult


UNSAFE_LABEL_MARKERS = ("nsfw", "porn", "hentai", "sexy", "explicit")


class ExplicitContentAdapter:
    name = "explicit_content_detector"

    def __init__(self, enabled: bool, model_id: str, cache_dir: str) -> None:
        self.enabled = enabled
        self.model_id = model_id
        self.cache_dir = cache_dir
        self._pipeline = None
        self._load_error: str | None = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._load_error is not None:
            return None
        try:
            from transformers import pipeline
        except ImportError:
            return None
        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", self.cache_dir)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", self.cache_dir)
            self._pipeline = pipeline(task="image-classification", model=self.model_id)
        except Exception as exc:  # pragma: no cover - backend-specific failures
            self._load_error = str(exc)
            return None
        return self._pipeline

    def moderate(self, image) -> SignalResult:
        if not self.enabled:
            return SignalResult(name=self.name, status="skipped", reason="Explicit-content detector disabled by configuration.")
        detector = self._load()
        if detector is None:
            if self._load_error:
                return SignalResult(
                    name=self.name,
                    status="error",
                    reason=f"Explicit-content detector load failed: {self._load_error}",
                )
            return SignalResult(
                name=self.name,
                status="skipped",
                reason="transformers is not installed.",
            )

        try:
            results = detector(image)
        except Exception as exc:  # pragma: no cover - backend-specific failures
            return SignalResult(
                name=self.name,
                status="error",
                reason=f"Explicit-content detector failed: {exc}",
            )

        max_score = 0.0
        raw = {}
        for item in results:
            label = str(item["label"]).lower()
            score = float(item["score"])
            raw[label] = score
            if any(marker in label for marker in UNSAFE_LABEL_MARKERS):
                max_score = max(max_score, score)

        categories = {"sexual": max_score} if max_score > 0 else {}
        return SignalResult(
            name=self.name,
            status="ok",
            categories=categories,
            reason="Computed specialized explicit-content score.",
            raw=raw,
        )
