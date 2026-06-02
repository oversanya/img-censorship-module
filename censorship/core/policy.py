"""Policy engine — maps (category, confidence) → ALLOW/BLOCK/REVIEW."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import NamedTuple

import yaml

from .taxonomy import TaxonomyLoader


class Decision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


class PolicyDecision(NamedTuple):
    decision: Decision
    category: str | None
    confidence: float
    reason: str


class PolicyEngine:
    """
    Applies bank policy thresholds to classifier/reasoner scores.

    Pipeline-level thresholds (fast_threshold_block / fast_threshold_allow)
    are used by ImagePipeline to decide whether to call the reasoner.
    Per-category thresholds (from taxonomy.yaml) are used here to map
    individual category scores to a final decision.
    """

    def __init__(
        self,
        taxonomy: TaxonomyLoader,
        policy_config_path: str | Path | None = None,
    ):
        self.taxonomy = taxonomy
        self._fast_block = 0.90
        self._fast_allow = 0.50
        if policy_config_path:
            self._load_policy(policy_config_path)

    def _load_policy(self, path: str | Path) -> None:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        pipeline = cfg.get("pipeline", {})
        self._fast_block = pipeline.get("fast_threshold_block", 0.90)
        self._fast_allow = pipeline.get("fast_threshold_allow", 0.50)

    @property
    def fast_threshold_block(self) -> float:
        return self._fast_block

    @property
    def fast_threshold_allow(self) -> float:
        return self._fast_allow

    def evaluate(self, scores: dict[str, float]) -> PolicyDecision:
        """Evaluate a dict of {category_id: confidence} → PolicyDecision."""
        # Critical categories always block regardless of score
        for cat_id, score in scores.items():
            cat = self.taxonomy.get(cat_id)
            if cat and cat.priority == "critical" and score > 0:
                return PolicyDecision(
                    decision=Decision.BLOCK,
                    category=cat_id,
                    confidence=score,
                    reason=f"Critical category '{cat_id}' detected (zero-tolerance policy).",
                )

        # Find highest-score category that exceeds block threshold
        block_candidates: list[tuple[float, str]] = []
        review_candidates: list[tuple[float, str]] = []

        for cat_id, score in scores.items():
            cat = self.taxonomy.get(cat_id)
            if cat is None:
                continue
            if score >= cat.threshold_block:
                block_candidates.append((score, cat_id))
            elif score >= cat.threshold_review:
                review_candidates.append((score, cat_id))

        if block_candidates:
            score, cat_id = max(block_candidates)
            return PolicyDecision(
                decision=Decision.BLOCK,
                category=cat_id,
                confidence=score,
                reason=f"Score {score:.3f} ≥ block threshold for '{cat_id}'.",
            )

        if review_candidates:
            score, cat_id = max(review_candidates)
            return PolicyDecision(
                decision=Decision.REVIEW,
                category=cat_id,
                confidence=score,
                reason=f"Score {score:.3f} in review zone for '{cat_id}'.",
            )

        max_score = max(scores.values()) if scores else 0.0
        return PolicyDecision(
            decision=Decision.ALLOW,
            category=None,
            confidence=max_score,
            reason="All scores below review thresholds.",
        )

    def needs_reasoner(self, max_score: float) -> bool:
        """True if max classifier score falls in the gray zone."""
        return self._fast_allow <= max_score < self._fast_block
