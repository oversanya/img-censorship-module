from __future__ import annotations

import os
from pathlib import Path

from censor_guard.schemas import SignalResult


# Метки, которые у разных NSFW-моделей означают «небезопасно». Сверяем
# подстрокой (label.lower()), поэтому подходят и "nsfw", и "porn", и "hentai".
UNSAFE_LABEL_MARKERS = ("nsfw", "porn", "hentai", "sexy", "explicit")


class ExplicitContentAdapter:
    """Узкоспециализированный детектор откровенного контента (по умолчанию
    Falconsai/nsfw_image_detection).

    В отличие от zero-shot классификатора, это обычная обученная модель
    image-classification: она выдаёт нормальные откалиброванные вероятности
    по своим родным меткам (обычно "normal" / "nsfw"). Мы берём максимальную
    оценку среди «небезопасных» меток и кладём её в категорию "sexual".
    """

    name = "explicit_content_detector"

    def __init__(self, enabled: bool, model_id: str, cache_dir: str) -> None:
        self.enabled = enabled
        self.model_id = model_id
        self.cache_dir = cache_dir
        # Кэш загруженного pipeline и текст ошибки загрузки — чтобы не пытаться
        # грузить тяжёлую модель повторно на каждый запрос.
        self._pipeline = None
        self._load_error: str | None = None

    def _load(self):
        # Ленивая загрузка: модель скачивается/инициализируется при первом вызове
        # и далее переиспользуется. Любой сбой превращается в «мягкую» деградацию
        # (skipped/error), а не в падение всего сервиса.
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

        # Из всех меток модели берём максимальную оценку среди тех, что
        # распознаны как «небезопасные». raw сохраняет полный расклад для отладки.
        max_score = 0.0
        raw = {}
        for item in results:
            label = str(item["label"]).lower()
            score = float(item["score"])
            raw[label] = score
            if any(marker in label for marker in UNSAFE_LABEL_MARKERS):
                max_score = max(max_score, score)

        # Категорию добавляем, только если действительно что-то нашли (>0),
        # иначе сенсор «молчит» и не влияет на решение.
        categories = {"sexual": max_score} if max_score > 0 else {}
        return SignalResult(
            name=self.name,
            status="ok",
            categories=categories,
            reason="Computed specialized explicit-content score.",
            raw=raw,
        )
