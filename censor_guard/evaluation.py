from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from censor_guard.adapters.explicit_detector import ExplicitContentAdapter
from censor_guard.adapters.policy_judge import HeuristicPolicyJudge
from censor_guard.adapters.visual_classifier import VisualClassifierAdapter
from censor_guard.config import Settings
from censor_guard.decision import DecisionEngine
from censor_guard.schemas import ModerationRequest


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

    def __init__(self, settings: Settings | None = None, use_policy_judge: bool = True) -> None:
        self.settings = settings or Settings()
        self.visual = VisualClassifierAdapter(
            enabled=True,
            model_id=self.settings.visual_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.explicit = ExplicitContentAdapter(
            enabled=True,
            model_id=self.settings.explicit_model_id,
            cache_dir=self.settings.hf_cache_dir,
        )
        self.judge = HeuristicPolicyJudge() if use_policy_judge else None
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
        if self.judge is not None:
            # Эвристический судья усиливает консенсус сенсоров (см. policy_judge.py).
            signals.append(self.judge.moderate(signals))

        response = self.engine.decide(self._request, signals)

        signal_scores: dict[str, dict[str, float]] = {}
        fused: dict[str, float] = {}
        for signal in signals:
            if signal.status != "ok":
                continue
            signal_scores[signal.name] = dict(signal.categories)
            for code, score in signal.categories.items():
                fused[code] = max(fused.get(code, 0.0), score)

        return ClassifierResult(
            verdict=response.verdict,
            categories=list(response.categories),
            confidence=response.confidence,
            fused_scores=fused,
            signal_scores=signal_scores,
            notes=list(response.notes),
        )

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
