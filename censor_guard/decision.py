from __future__ import annotations

from collections import defaultdict

from censor_guard.fusion import EVIDENCE_FLOOR, FUSION_SIGNAL_NAME
from censor_guard.schemas import ModerationResponse, ModerationRequest, SignalResult
from censor_guard.taxonomy import HARD_BLOCK_CATEGORIES, SOFT_REVIEW_CATEGORIES

SHIELDGEMMA_SENSOR = "policy_judge_shieldgemma"


class DecisionEngine:
    """Финальный вердикт на ОТКАЛИБРОВАННЫХ сведённых оценках.

    Источник оценок — сигнал `policy_fusion` (взвешенный noisy-OR из fusion.py):
    каждая категория уже имеет честную оценку 0..1 на единой шкале и список
    сенсоров, которые её подняли. Если сигнал fusion отсутствует (например, движок
    вызывают изолированно в тестах), откатываемся на максимум по сенсорам.

    Правила:
      - block_threshold  (по умолч. 0.85) — выше = блок;
      - review_threshold (по умолч. 0.55) — выше = ручная проверка.
    Hard-категории блокируются от сведённой оценки напрямую. Soft-категории
    блокируются только при подтверждении: согласие ≥2 независимых сенсоров ИЛИ
    решение эскалационного судьи ShieldGemma. Это гасит ложные soft-блокировки от
    одиночного шумного сенсора.
    """

    def __init__(self, block_threshold: float, review_threshold: float) -> None:
        self.block_threshold = block_threshold
        self.review_threshold = review_threshold

    def _collect(
        self, signals: list[SignalResult]
    ) -> tuple[dict[str, float], dict[str, list[str]], dict[str, int], list[str]]:
        """Возвращает (оценки, источники, согласие, notes).

        Приоритет — сигналу fusion. Без него считаем максимум по ok-сенсорам
        (легаси-путь для изолированных тестов движка)."""
        notes: list[str] = []
        fusion_signal = next(
            (s for s in signals if s.name == FUSION_SIGNAL_NAME and s.status == "ok"), None
        )
        for signal in signals:
            if signal.status != "ok" and signal.reason:
                notes.append(f"{signal.name}: {signal.reason}")

        if fusion_signal is not None:
            scores = dict(fusion_signal.categories)
            contributions = fusion_signal.raw.get("contributions", {})
            sources = {
                code: sorted({c["sensor"] for c in contribs})
                for code, contribs in contributions.items()
            }
            agreement = dict(fusion_signal.raw.get("agreement", {}))
            return scores, sources, agreement, notes

        # Легаси-фолбэк: ни одного fusion-сигнала.
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, list[str]] = defaultdict(list)
        agreement: dict[str, int] = defaultdict(int)
        for signal in signals:
            if signal.status != "ok":
                continue
            for category, score in signal.categories.items():
                scores[category] = max(scores[category], score)
                if score > 0:
                    sources[category].append(signal.name)
                if score >= EVIDENCE_FLOOR:
                    agreement[category] += 1
        return dict(scores), {k: sorted(set(v)) for k, v in sources.items()}, dict(agreement), notes

    def decide(self, request: ModerationRequest, signals: list[SignalResult]) -> ModerationResponse:
        scores, sources, agreement, notes = self._collect(signals)

        hard_block_matches = [
            category
            for category in HARD_BLOCK_CATEGORIES
            if scores.get(category, 0.0) >= self.block_threshold
        ]
        # Soft-блок требует подтверждения: согласие ≥2 сенсоров или вердикт ShieldGemma.
        soft_block_matches = [
            category
            for category in SOFT_REVIEW_CATEGORIES
            if scores.get(category, 0.0) >= self.block_threshold
            and (
                agreement.get(category, 0) >= 2
                or SHIELDGEMMA_SENSOR in sources.get(category, [])
            )
        ]
        blocked = set(hard_block_matches) | set(soft_block_matches)
        review_matches = [
            category
            for category, score in scores.items()
            if score >= self.review_threshold and category not in blocked
        ]

        if blocked:
            verdict = "block"
            categories = sorted(blocked)
            confidence = max(scores[category] for category in categories)
            reason = "Blocked by image guardrail due to unsafe content."
        elif review_matches:
            verdict = "review"
            categories = sorted(review_matches)
            confidence = max(scores[category] for category in categories)
            reason = "Request requires secondary review due to medium-confidence policy signals."
        else:
            verdict = "allow"
            categories = []
            confidence = max(scores.values(), default=0.0)
            reason = "No blocking policy signals exceeded review thresholds."

        evidence = {
            category: sorted(set(sources.get(category, [])))
            for category in categories
        }
        return ModerationResponse(
            request_id=request.request_id,
            scenario=request.scenario,
            stage=request.stage,
            verdict=verdict,
            categories=categories,
            confidence=round(confidence, 4),
            reason=reason,
            evidence=evidence,
            signals=signals,
            notes=notes,
        )
