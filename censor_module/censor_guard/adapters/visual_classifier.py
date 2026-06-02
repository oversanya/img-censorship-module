from __future__ import annotations

import os
from pathlib import Path

from censor_guard.schemas import SignalResult
from censor_guard.taxonomy import VISUAL_LABELS, VISUAL_LABEL_TO_CODE


class VisualClassifierAdapter:
    name = "visual_classifier"

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
            self._pipeline = pipeline(
                task="zero-shot-image-classification",
                model=self.model_id,
            )
        except Exception as exc:  # pragma: no cover - backend-specific failures
            self._load_error = str(exc)
            return None
        return self._pipeline

    def moderate(self, image) -> SignalResult:
        if not self.enabled:
            return SignalResult(name=self.name, status="skipped", reason="Visual classifier disabled by configuration.")
        classifier = self._load()
        if classifier is None:
            if self._load_error:
                return SignalResult(
                    name=self.name,
                    status="error",
                    reason=f"Visual classifier load failed: {self._load_error}",
                )
            return SignalResult(
                name=self.name,
                status="skipped",
                reason="transformers is not installed.",
            )
        try:
            results = classifier(image, candidate_labels=VISUAL_LABELS)
        except Exception as exc:  # pragma: no cover - backend-specific failures
            return SignalResult(
                name=self.name,
                status="error",
                reason=f"Visual classifier failed: {exc}",
            )

        scores: dict[str, float] = {}
        raw = {}
        for item in results:
            label = item["label"]
            score = float(item["score"])
            code = VISUAL_LABEL_TO_CODE.get(label)
            if code is None:
                continue
            scores[code] = score
            raw[label] = score

        return SignalResult(
            name=self.name,
            status="ok",
            categories=scores,
            reason="Computed multi-label visual safety scores.",
            raw=raw,
        )
