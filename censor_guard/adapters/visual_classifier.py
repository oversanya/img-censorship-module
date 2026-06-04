from __future__ import annotations

import os
from pathlib import Path

from censor_guard.calibration import calibrate_against_safe
from censor_guard.schemas import SignalResult
from censor_guard.taxonomy import (
    SAFE_VISUAL_LABEL_SET,
    VISUAL_LABELS,
    VISUAL_LABEL_TO_CODE,
)


class VisualClassifierAdapter:
    """Zero-shot мульти-классификатор по нашей таксономии (по умолчанию CLIP).

    Модель не обучалась на наших категориях — мы передаём ей текстовые описания
    категорий (candidate_labels из taxonomy.VISUAL_LABELS) плюс несколько
    нейтральных safe-якорей, и она оценивает сходство картинки с каждым описанием.

    Сырой zero-shot softmax-ит оценки по всем меткам (сумма ≈ 1.0), поэтому у любой
    картинки всегда есть «самая вероятная» категория нарушения — сравнивать такие
    числа с порогом нельзя. Поэтому адаптер НЕ отдаёт сырой softmax в categories:
    он калибрует оценки относительно safe-якоря (calibration.calibrate_against_safe),
    превращая «относительное сходство» в честную оценку опасности 0..1. Сырой softmax
    сохраняется в raw["softmax"] для отладки.
    """

    name = "visual_classifier"

    def __init__(self, enabled: bool, model_id: str, cache_dir: str, calibration_floor: float = 0.5) -> None:
        self.enabled = enabled
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.calibration_floor = calibration_floor
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

        softmax = {item["label"]: float(item["score"]) for item in results}

        # Калибруем относительно safe-якоря: сырой softmax → честная оценка опасности.
        calibrated = calibrate_against_safe(
            raw_scores=softmax,
            label_to_code=VISUAL_LABEL_TO_CODE,
            safe_labels=SAFE_VISUAL_LABEL_SET,
            floor=self.calibration_floor,
        )

        return SignalResult(
            name=self.name,
            status="ok",
            categories=calibrated.scores,
            reason="Computed calibrated visual safety scores (zero-shot vs safe anchor).",
            raw={
                "softmax": softmax,
                "safe_score": calibrated.safe_score,
                "calibration_floor": calibrated.floor,
            },
        )
