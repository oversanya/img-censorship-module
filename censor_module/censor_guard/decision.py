from __future__ import annotations

from collections import defaultdict

from censor_guard.schemas import ModerationResponse, ModerationRequest, SignalResult
from censor_guard.taxonomy import HARD_BLOCK_CATEGORIES, SOFT_REVIEW_CATEGORIES


class DecisionEngine:
    def __init__(self, block_threshold: float, review_threshold: float) -> None:
        self.block_threshold = block_threshold
        self.review_threshold = review_threshold

    def decide(self, request: ModerationRequest, signals: list[SignalResult]) -> ModerationResponse:
        category_scores: dict[str, float] = defaultdict(float)
        category_sources: dict[str, list[str]] = defaultdict(list)
        notes: list[str] = []

        for signal in signals:
            if signal.status != "ok":
                if signal.reason:
                    notes.append(f"{signal.name}: {signal.reason}")
                continue
            for category, score in signal.categories.items():
                if score >= category_scores[category]:
                    category_scores[category] = score
                if score > 0:
                    category_sources[category].append(signal.name)

        hard_block_matches = [
            category
            for category in HARD_BLOCK_CATEGORIES
            if category_scores.get(category, 0.0) >= self.block_threshold
        ]
        soft_block_matches = [
            category
            for category in SOFT_REVIEW_CATEGORIES
            if category_scores.get(category, 0.0) >= self.block_threshold
            and "policy_judge_heuristic" in category_sources.get(category, [])
        ]
        review_matches = [
            category
            for category, score in category_scores.items()
            if score >= self.review_threshold and category not in hard_block_matches and category not in soft_block_matches
        ]

        if hard_block_matches or soft_block_matches:
            verdict = "block"
            categories = sorted(set(hard_block_matches + soft_block_matches))
            confidence = max(category_scores[category] for category in categories)
            reason = "Blocked by image guardrail due to unsafe content."
        elif review_matches:
            verdict = "review"
            categories = sorted(review_matches)
            confidence = max(category_scores[category] for category in categories)
            reason = "Request requires secondary review due to medium-confidence policy signals."
        else:
            verdict = "allow"
            categories = []
            confidence = max(category_scores.values(), default=0.0)
            reason = "No blocking policy signals exceeded review thresholds."

        evidence = {category: sorted(set(sources)) for category, sources in category_sources.items() if category in categories}
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

