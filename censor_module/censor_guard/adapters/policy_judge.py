from __future__ import annotations

from collections import defaultdict

from censor_guard.schemas import SignalResult
from censor_guard.taxonomy import HARD_BLOCK_CATEGORIES


class HeuristicPolicyJudge:
    """Эвристический «арбитр» на случай, когда настоящей модели-судьи нет.

    Логика простая: чем больше независимых сенсоров увидели одну и ту же
    категорию, тем выше доверие. Поэтому:
      - 2+ подтверждения  → поднимаем оценку на +0.10 (консенсус сенсоров);
      - 1 подтверждение в hard-категории → оставляем как есть (не глушим опасное);
      - 1 подтверждение в soft-категории → снижаем на -0.05 (гасим одиночный шум).
    """

    name = "policy_judge_heuristic"

    def moderate(self, signals: list[SignalResult]) -> SignalResult:
        scores: dict[str, float] = defaultdict(float)
        evidence_count: dict[str, int] = defaultdict(int)

        for signal in signals:
            if signal.status != "ok":
                continue
            for category, score in signal.categories.items():
                scores[category] = max(scores[category], score)
                if score > 0:
                    evidence_count[category] += 1

        adjusted: dict[str, float] = {}
        for category, score in scores.items():
            if evidence_count[category] >= 2:
                adjusted[category] = min(1.0, score + 0.10)
            elif category in HARD_BLOCK_CATEGORIES:
                adjusted[category] = score
            else:
                adjusted[category] = max(0.0, score - 0.05)

        return SignalResult(
            name=self.name,
            status="ok",
            categories=adjusted,
            reason="Heuristic policy judge fused available sensor evidence.",
            raw={"mode": "heuristic", "evidence_count": dict(evidence_count)},
        )


class ShieldGemmaJudge:
    name = "policy_judge_shieldgemma"

    def __init__(self, enabled: bool, model_id: str) -> None:
        self.enabled = enabled
        self.model_id = model_id

    def moderate(self, image, prompt: str | None) -> SignalResult:
        if not self.enabled:
            return SignalResult(
                name=self.name,
                status="skipped",
                reason="ShieldGemma judge disabled by configuration.",
            )
        return SignalResult(
            name=self.name,
            status="skipped",
            reason=(
                "ShieldGemma integration point is enabled in architecture but not wired in this MVP runtime. "
                "Use the heuristic judge until model loading/inference is attached."
            ),
            raw={"model_id": self.model_id, "prompt_present": bool(prompt)},
        )


class PolicyJudgeFacade:
    """Фасад над судьёй: пытается использовать ShieldGemma, при недоступности —
    откатывается на эвристику. Пайплайн вызывает только этот фасад и не знает,
    какой именно судья отработал."""

    def __init__(self, enabled: bool, model_id: str) -> None:
        self.shieldgemma = ShieldGemmaJudge(enabled=enabled, model_id=model_id)
        self.heuristic = HeuristicPolicyJudge()

    def moderate(self, image, prompt: str | None, signals: list[SignalResult]) -> SignalResult:
        # Сначала пробуем «настоящую» модель. Сейчас она всегда возвращает
        # skipped (не подключена), поэтому де-факто всегда работает эвристика.
        shieldgemma_result = self.shieldgemma.moderate(image=image, prompt=prompt)
        if shieldgemma_result.status == "ok":
            return shieldgemma_result
        # Фолбэк на эвристику; в raw кладём причину отката для прозрачности.
        heuristic_result = self.heuristic.moderate(signals)
        heuristic_result.raw["fallback_from"] = shieldgemma_result.model_dump()
        return heuristic_result

