from __future__ import annotations

from collections import defaultdict
from typing import Any

from censor_guard.fusion import EVIDENCE_FLOOR, FUSION_SIGNAL_NAME
from censor_guard.observability import DecisionExplanation
from censor_guard.schemas import ModerationResponse, ModerationRequest, SignalResult
from censor_guard.taxonomy import HARD_BLOCK_CATEGORIES, SOFT_REVIEW_CATEGORIES

SHIELDGEMMA_SENSOR = "policy_judge_shieldgemma"
# Обученные policy-aware судьи: их одиночного вердикта достаточно, чтобы
# подтвердить soft-блок (наравне с согласием ≥2 дешёвых сенсоров).
TRUSTED_JUDGES = (SHIELDGEMMA_SENSOR, "llava_guard")


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

    def __init__(
        self,
        block_threshold: float,
        review_threshold: float,
        policy_version: str = "v1",
    ) -> None:
        self.block_threshold = block_threshold
        self.review_threshold = review_threshold
        self.policy_version = policy_version

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
                or any(judge in sources.get(category, []) for judge in TRUSTED_JUDGES)
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
            reason_code = self._block_reason_code(categories, hard_block_matches, sources)
            reason = self._human_reason(reason_code)
        elif review_matches:
            verdict = "review"
            categories = sorted(review_matches)
            confidence = max(scores[category] for category in categories)
            reason_code = "category_above_review_threshold"
            reason = self._human_reason(reason_code)
        else:
            verdict = "allow"
            categories = []
            confidence = max(scores.values(), default=0.0)
            reason_code = "no_policy_signal_above_review_threshold"
            reason = self._human_reason(reason_code)

        evidence = {
            category: sorted(set(sources.get(category, [])))
            for category in categories
        }
        audit = DecisionExplanation(
            reason_code=reason_code,
            human_reason=reason,
            policy_version=self.policy_version,
            thresholds={
                "block": self.block_threshold,
                "review": self.review_threshold,
            },
            matched_categories=self._matched_categories(categories, scores, sources, agreement),
            decision_path=self._decision_path(
                scores=scores,
                sources=sources,
                agreement=agreement,
                hard_block_matches=hard_block_matches,
                soft_block_matches=soft_block_matches,
                review_matches=review_matches,
                verdict=verdict,
                reason_code=reason_code,
            ),
        )
        return ModerationResponse(
            request_id=request.request_id,
            scenario=request.scenario,
            stage=request.stage,
            verdict=verdict,
            categories=categories,
            confidence=round(confidence, 4),
            reason=reason,
            evidence=evidence,
            audit=audit,
            signals=signals,
            notes=notes,
        )

    def _block_reason_code(
        self,
        categories: list[str],
        hard_block_matches: list[str],
        sources: dict[str, list[str]],
    ) -> str:
        if any(category in hard_block_matches for category in categories):
            return "hard_category_above_block_threshold"
        if any(SHIELDGEMMA_SENSOR in sources.get(category, []) for category in categories):
            return "soft_category_confirmed_by_policy_judge"
        return "soft_category_confirmed_by_multiple_sensors"

    def _human_reason(self, reason_code: str) -> str:
        reasons = {
            "hard_category_above_block_threshold": (
                "Blocked because a hard-block policy category reached the block threshold."
            ),
            "soft_category_confirmed_by_multiple_sensors": (
                "Blocked because a soft policy category reached the block threshold and was confirmed by multiple sensors."
            ),
            "soft_category_confirmed_by_policy_judge": (
                "Blocked because a soft policy category reached the block threshold and was confirmed by the policy judge."
            ),
            "category_above_review_threshold": (
                "Request requires secondary review because a policy category reached the review threshold."
            ),
            "no_policy_signal_above_review_threshold": (
                "Allowed because no policy signal reached the review threshold."
            ),
        }
        return reasons[reason_code]

    def _matched_categories(
        self,
        categories: list[str],
        scores: dict[str, float],
        sources: dict[str, list[str]],
        agreement: dict[str, int],
    ) -> list[dict[str, Any]]:
        return [
            {
                "category": category,
                "score": round(scores.get(category, 0.0), 4),
                "sources": sorted(set(sources.get(category, []))),
                "agreement": agreement.get(category, 0),
            }
            for category in categories
        ]

    def _decision_path(
        self,
        scores: dict[str, float],
        sources: dict[str, list[str]],
        agreement: dict[str, int],
        hard_block_matches: list[str],
        soft_block_matches: list[str],
        review_matches: list[str],
        verdict: str,
        reason_code: str,
    ) -> list[dict[str, Any]]:
        path: list[dict[str, Any]] = [
            {
                "step": "collect_scores",
                "scores": {category: round(score, 4) for category, score in scores.items()},
                "sources": sources,
                "agreement": agreement,
            },
            {
                "step": "apply_block_rules",
                "threshold": self.block_threshold,
                "hard_block_matches": hard_block_matches,
                "soft_block_matches": soft_block_matches,
            },
            {
                "step": "apply_review_rules",
                "threshold": self.review_threshold,
                "review_matches": review_matches,
            },
            {
                "step": "final_verdict",
                "verdict": verdict,
                "reason_code": reason_code,
            },
        ]
        return path
