"""Core data structures for pipeline verdicts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Union


def _image_sha256(image_source: Union[str, Path, bytes]) -> str:
    """Compute SHA-256 of an image file or bytes."""
    if isinstance(image_source, bytes):
        return hashlib.sha256(image_source).hexdigest()
    with open(image_source, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ClassifierResult:
    """Output from a Layer-1 fast classifier."""
    model: str
    scores: dict[str, float]          # {category_id: confidence 0..1}
    is_unsafe: bool
    triggered_categories: list[str]   # categories that exceeded threshold
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def max_score(self) -> float:
        if not self.scores:
            return 0.0
        val = max(self.scores.values())
        return 0.0 if val != val else val  # guard: NaN != NaN

    def top_category(self) -> str | None:
        if not self.scores:
            return None
        return max(self.scores, key=self.scores.__getitem__)


@dataclass
class ReasonerResult:
    """Output from a Layer-2 VLM reasoner."""
    model: str
    verdict: str                # "ALLOW" | "BLOCK" | "REVIEW"
    category: str | None        # primary unsafe category
    confidence: float           # 0..1
    rationale: str              # human-readable explanation
    raw_response: str = ""
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PromptGuardResult:
    """Output from the prompt guard module."""
    model: str
    verdict: str                # "ALLOW" | "BLOCK"
    category: str | None        # unsafe category if blocked
    confidence: float
    rationale: str = ""
    latency_ms: float = 0.0


@dataclass
class Verdict:
    """Final pipeline verdict for one image (and optional prompt)."""

    image_id: str                     # sha256 of image bytes
    timestamp: str                    # ISO-8601 UTC
    decision: str                     # "ALLOW" | "BLOCK" | "REVIEW"

    # Layer 1
    classifier_model: str
    classifier_scores: dict           # {category_id: confidence}
    classifier_triggered: list[str]

    # Layer 2 (populated if reasoner was called)
    reasoner_model: str | None = None
    reasoner_rationale: str | None = None
    reasoner_confidence: float | None = None

    # Interpretability
    primary_category: str | None = None
    explanation_for_user: str | None = None
    explanation_for_regulator: str | None = None

    # Prompt guard (if prompt was checked)
    prompt_verdict: str | None = None
    prompt_category: str | None = None

    # Technical
    latency_ms: float = 0.0
    pipeline_version: str = "1.0.0"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_classifier_allow(
        cls,
        image_id: str,
        classifier_result: ClassifierResult,
        latency_ms: float = 0.0,
    ) -> "Verdict":
        return cls(
            image_id=image_id,
            timestamp=_now_iso(),
            decision="ALLOW",
            classifier_model=classifier_result.model,
            classifier_scores=classifier_result.scores,
            classifier_triggered=[],
            explanation_for_user="No prohibited content detected.",
            explanation_for_regulator=(
                f"Image classified as safe by {classifier_result.model}. "
                f"Max confidence score: {classifier_result.max_score():.3f}."
            ),
            latency_ms=latency_ms,
        )

    @classmethod
    def from_classifier_block(
        cls,
        image_id: str,
        classifier_result: ClassifierResult,
        policy_thresholds: dict[str, float],
        latency_ms: float = 0.0,
    ) -> "Verdict":
        top_cat = classifier_result.top_category()
        top_score = classifier_result.max_score()
        return cls(
            image_id=image_id,
            timestamp=_now_iso(),
            decision="BLOCK",
            classifier_model=classifier_result.model,
            classifier_scores=classifier_result.scores,
            classifier_triggered=classifier_result.triggered_categories,
            primary_category=top_cat,
            explanation_for_user=(
                f"Image blocked: prohibited content detected ({top_cat})."
            ),
            explanation_for_regulator=(
                f"Image blocked by {classifier_result.model}. "
                f"Category: {top_cat}. Confidence: {top_score:.3f}. "
                f"Threshold: {policy_thresholds.get(top_cat, 'N/A')}."
            ),
            latency_ms=latency_ms,
        )
