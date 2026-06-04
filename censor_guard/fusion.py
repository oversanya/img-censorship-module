from __future__ import annotations

from dataclasses import dataclass, field

from censor_guard.schemas import SignalResult


# Вес = надёжность сенсора как источника решения. Откалиброванные специалисты и
# судья получают высокий вес; слабый zero-shot CLIP — пониже (но достаточно, чтобы
# при сильной калиброванной оценке он мог дотянуть до блокировки — таково решение
# по продукту). Неизвестные сенсоры получают DEFAULT_WEIGHT.
DEFAULT_SENSOR_WEIGHTS: dict[str, float] = {
    "explicit_content_detector": 1.0,
    "policy_judge_shieldgemma": 1.0,
    "text_guard_heuristic": 0.85,
    "ocr_text_guard_heuristic": 0.85,
    "visual_classifier": 0.9,
}
DEFAULT_WEIGHT = 0.5

# Калиброванная оценка, ниже которой сигнал считается шумом и НЕ учитывается как
# самостоятельное «свидетельство» (в т.ч. при подсчёте согласия сенсоров). Это
# заменяет прежний бессмысленный evidence_count, где даже softmax-шум CLIP по
# каждой категории засчитывался за улику.
EVIDENCE_FLOOR = 0.3

FUSION_SIGNAL_NAME = "policy_fusion"


@dataclass
class CategoryFusion:
    """Сведённая оценка по одной категории + откуда она взялась."""

    score: float
    # [{"sensor": str, "score": float, "weight": float}] — реальный вклад каждого
    # сенсора, увидевшего категорию (для прозрачности вместо магических чисел).
    contributions: list[dict] = field(default_factory=list)
    # Сколько независимых сенсоров уверены выше EVIDENCE_FLOOR (осмысленное согласие).
    agreement: int = 0


@dataclass
class FusionResult:
    categories: dict[str, CategoryFusion]

    def scores(self) -> dict[str, float]:
        return {code: cat.score for code, cat in self.categories.items()}


def fuse(
    signals: list[SignalResult],
    weights: dict[str, float] | None = None,
    evidence_floor: float = EVIDENCE_FLOOR,
) -> FusionResult:
    """Сводит сигналы сенсоров в одну оценку на категорию через взвешенный noisy-OR.

    Для категории c с оценками p_i от сенсоров с весами w_i::

        score(c) = 1 - Π_i (1 - w_i * p_i)

    Свойства, которых не было у старой эвристики:
      - согласие усиливает уверенность ЕСТЕСТВЕННО (два сенсора по 0.6 дают больше,
        чем один), без магических «+0.10»;
      - сенсор, не увидевший категорию, не влияет (множитель = 1);
      - вклад каждого сенсора виден в contributions — никаких загадочных счётчиков;
      - «согласие» считается только по сигналам выше evidence_floor, поэтому
        softmax-шум CLIP больше не выдаёт себя за улику.
    """

    weights = weights or DEFAULT_SENSOR_WEIGHTS
    per_category: dict[str, CategoryFusion] = {}

    for signal in signals:
        if signal.status != "ok":
            continue
        # Сам сигнал fusion не сливаем сам в себя (на случай повторного прогона).
        if signal.name == FUSION_SIGNAL_NAME:
            continue
        weight = weights.get(signal.name, DEFAULT_WEIGHT)
        for code, raw_score in signal.categories.items():
            if raw_score <= 0.0:
                continue
            cat = per_category.setdefault(code, CategoryFusion(score=0.0))
            cat.contributions.append(
                {"sensor": signal.name, "score": round(float(raw_score), 4), "weight": weight}
            )
            if raw_score >= evidence_floor:
                cat.agreement += 1

    # Считаем noisy-OR по собранным вкладам.
    for cat in per_category.values():
        complement = 1.0
        for contrib in cat.contributions:
            complement *= 1.0 - min(1.0, contrib["weight"] * contrib["score"])
        cat.score = round(1.0 - complement, 4)

    return FusionResult(categories=per_category)
