from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from censor_guard.adapters.explicit_detector import ExplicitContentAdapter
from censor_guard.adapters.robust_guard import RobustGuardAdapter
from censor_guard.adapters.visual_classifier import VisualClassifierAdapter
from censor_guard.config import Settings
from censor_guard.decision import DecisionEngine
from censor_guard.fusion import FUSION_SIGNAL_NAME, fuse
from censor_guard.schemas import ModerationRequest, SignalResult


# Маппинг 11 топиковых категорий UnsafeBench → коды нашей таксономии.
# Совпадение почти 1:1, что делает датасет удобным для бенчмарка.
UNSAFEBENCH_TO_TAXONOMY: dict[str, str] = {
    "Hate": "hate_extremism",
    "Violence": "violence_gore",
    "Self-harm": "self_harm",
    "Sexual": "sexual",
    "Shocking": "shocking",
    "Illegal activity": "illegal_activity",
    "Deception": "deception_fraud",
    "Political": "political_persuasion",
    "Public and personal health": "health_misinformation",
    "Spam": "spam_scams",
    "Harassment": "harassment",
}


@dataclass
class ClassifierResult:
    """Результат прогона одной картинки через image-классификатор.

    Хранит и финальный вердикт движка, и сырые оценки по каждому сенсору —
    чтобы в ноутбуке можно было анализировать вклад CLIP vs NSFW-детектора
    по отдельности, а не только итог.
    """

    verdict: str  # allow / review / block
    categories: list[str]
    confidence: float
    # Слитая (max по сенсорам) оценка для каждой категории — то, на что смотрит движок.
    fused_scores: dict[str, float]
    # Сырые оценки по каждому ok-сенсору: {имя_сенсора: {категория: оценка}}.
    signal_scores: dict[str, dict[str, float]]
    notes: list[str] = field(default_factory=list)
    # Робастный индикатор adversarial (None, если гард выключен/недоступен):
    # P(unsafe) робастной модели, его расхождение с unsafe_score основного раннера
    # и флаг «расхождение ≥ порога» (подозрение на adversarial-пример).
    robust_unsafe: float | None = None
    robust_divergence: float | None = None
    robust_adversarial: bool = False

    @property
    def flagged(self) -> bool:
        """Картинка помечена как небезопасная (review или block)."""
        return self.verdict in {"review", "block"}

    @property
    def blocked(self) -> bool:
        return self.verdict == "block"

    @property
    def unsafe_score(self) -> float:
        """Единая непрерывная «оценка небезопасности» = максимум по всем категориям.

        Удобна для ROC / подбора порогов: бинаризуем по любому порогу и сравниваем
        с истинной меткой Safe/Unsafe.
        """
        return max(self.fused_scores.values(), default=0.0)

    def score_for(self, code: str) -> float:
        return self.fused_scores.get(code, 0.0)


class ImageClassifierRunner:
    """Тонкая обёртка над реальными сенсорами картинки + DecisionEngine.

    Намеренно НЕ запускает OCR и текстовый гард: первый требует Tesseract,
    второй — заглушка. Так раннер изолирует именно image-классификатор
    (zero-shot CLIP + NSFW-детектор + эвристический судья), который мы и
    хотим измерить. Модели грузятся лениво и переиспользуются между вызовами.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        use_policy_judge: bool = True,
        use_robust_guard: bool = False,
    ) -> None:
        self.settings = settings or Settings()
        self.visual = VisualClassifierAdapter(
            enabled=True,
            model_id=self.settings.visual_model_id,
            cache_dir=self.settings.hf_cache_dir,
            calibration_floor=self.settings.calibration_floor,
        )
        self.explicit = ExplicitContentAdapter(
            enabled=True,
            model_id=self.settings.explicit_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        # use_policy_judge=True → сводим сенсоры взвешенным noisy-OR (fusion.py);
        # False → движок берёт максимум по сенсорам (легаси-режим для сравнения).
        self.use_policy_judge = use_policy_judge
        # Робастный детектор adversarial (опционален в бенчмарке). Веса/модель грузятся
        # лениво при первом вызове; если недоступны — гард молча молчит (score=None).
        self.use_robust_guard = use_robust_guard
        self.robust_guard = RobustGuardAdapter(
            enabled=use_robust_guard,
            probe_path=self.settings.robust_probe_path,
            model_dir=self.settings.robust_model_dir,
        )
        self.robust_unsafe_min = self.settings.robust_unsafe_min
        self.review_threshold = self.settings.review_threshold
        self.engine = DecisionEngine(
            block_threshold=self.settings.block_threshold,
            review_threshold=self.settings.review_threshold,
        )
        # Сценарий/стадия на решение пока не влияют — фиксируем один запрос.
        self._request = ModerationRequest(scenario="output", stage="output")

    def classify(self, image: Image.Image) -> ClassifierResult:
        if image.mode != "RGB":
            image = image.convert("RGB")

        signals = [self.visual.moderate(image), self.explicit.moderate(image)]

        # Сырые оценки по каждому сенсору считаем до fusion (для разбора в ноутбуке).
        signal_scores: dict[str, dict[str, float]] = {
            signal.name: dict(signal.categories)
            for signal in signals
            if signal.status == "ok"
        }

        if self.use_policy_judge:
            fusion = fuse(signals)
            fused: dict[str, float] = fusion.scores()
            signals.append(
                SignalResult(
                    name=FUSION_SIGNAL_NAME,
                    status="ok",
                    categories=fused,
                    reason="Fused calibrated sensor evidence via weighted noisy-OR.",
                    raw={
                        "contributions": {c: cat.contributions for c, cat in fusion.categories.items()},
                        "agreement": {c: cat.agreement for c, cat in fusion.categories.items()},
                    },
                )
            )
        else:
            fused = {}
            for scores in signal_scores.values():
                for code, score in scores.items():
                    fused[code] = max(fused.get(code, 0.0), score)

        response = self.engine.decide(self._request, signals)

        result = ClassifierResult(
            verdict=response.verdict,
            categories=list(response.categories),
            confidence=response.confidence,
            fused_scores=fused,
            signal_scores=signal_scores,
            notes=list(response.notes),
        )

        # Робастный детектор adversarial: сверяем P(unsafe) робастной модели с
        # unsafe_score основного раннера. Вердикт НЕ трогаем — пишем сырые поля,
        # чтобы бенчмарк честно сравнил baseline и baseline+robust.
        if self.use_robust_guard:
            robust_unsafe = self.robust_guard.unsafe_score(image)
            if robust_unsafe is not None:
                result.robust_unsafe = robust_unsafe
                result.robust_divergence = robust_unsafe - result.unsafe_score  # знаковое
                # Направленное правило: робастная уверенно unsafe, основная пропускает.
                result.robust_adversarial = (
                    robust_unsafe >= self.robust_unsafe_min
                    and result.unsafe_score < self.review_threshold
                )

        return result

    def raw_visual_scores(self, image: Image.Image, candidate_labels: list[str]) -> dict[str, float]:
        """Прямой вызов zero-shot CLIP с произвольными метками.

        Нужен для эксперимента с калибровкой (нейтральные якоря): передаём
        список меток сами и получаем сырые softmax-оценки без маппинга в таксономию.
        Возвращает {} если бэкенд недоступен.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        pipe = self.visual._load()
        if pipe is None:
            return {}
        results = pipe(image, candidate_labels=candidate_labels)
        return {item["label"]: float(item["score"]) for item in results}
