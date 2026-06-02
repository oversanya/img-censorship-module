"""Two-layer image classification pipeline.

Layer 1: Fast ImageClassifier  (ShieldGemma-2 / NudeNet / Q16)
  - score ≥ 0.90  → immediate BLOCK
  - score < 0.50  → immediate ALLOW
  - 0.50 – 0.90   → escalate to Layer 2

Layer 2: VLM ImageReasoner  (LlavaGuard / SG2-reason)
  - Returns structured verdict + rationale
"""

from __future__ import annotations

import time
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import yaml

from censorship.core.verdict import Verdict, ClassifierResult, _image_sha256, _now_iso
from censorship.core.taxonomy import TaxonomyLoader
from censorship.core.policy import PolicyEngine, Decision
from censorship.classifiers.base import ImageClassifier
from censorship.classifiers.registry import get_classifier
from censorship.reasoners.base import ImageReasoner
from censorship.reasoners.registry import get_reasoner
from censorship.audit.logger import AuditLogger

logger = logging.getLogger(__name__)


class ImagePipeline:
    """
    Two-layer image safety pipeline.

    Usage:
        pipeline = ImagePipeline.from_config("config/models.yaml")
        verdict = pipeline.run("image.jpg")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        classifier: ImageClassifier,
        reasoner: ImageReasoner | None,
        taxonomy: TaxonomyLoader,
        policy_engine: PolicyEngine,
        audit_logger: AuditLogger | None = None,
        policy_descriptions: dict[str, str] | None = None,
    ):
        self.classifier = classifier
        self.reasoner = reasoner
        self.taxonomy = taxonomy
        self.policy = policy_engine
        self.audit_logger = audit_logger
        self._policy_descriptions = policy_descriptions or {}
        # Public alias for notebook / external access
        self.policy_descriptions = self._policy_descriptions

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config/models.yaml",
        taxonomy_path: str | Path = "config/taxonomy.yaml",
        policy_path: str | Path = "config/policy_bank.yaml",
        classifier: str = "shieldgemma2",
        reasoner: str | None = "shieldgemma2_reason",
        audit_log: str | None = "audit.jsonl",
        hf_token: str | None = None,
    ) -> "ImagePipeline":
        config_path = Path(config_path)
        taxonomy_path = Path(taxonomy_path)

        with open(config_path) as f:
            models_cfg = yaml.safe_load(f)

        taxonomy = TaxonomyLoader(taxonomy_path)
        policy_engine = PolicyEngine(taxonomy, policy_path if Path(policy_path).exists() else None)

        # Load policy descriptions
        policy_descriptions: dict[str, str] = {}
        if Path(policy_path).exists():
            with open(policy_path) as f:
                policy_cfg = yaml.safe_load(f)
            policy_descriptions = policy_cfg.get("policy_descriptions", {})

        hf_token = hf_token or os.environ.get("HF_TOKEN")

        clf_cfg = models_cfg.get("image_classifiers", {}).get(classifier, {})
        clf = get_classifier(classifier, clf_cfg, hf_token=hf_token)

        rsn = None
        if reasoner:
            rsn_cfg = models_cfg.get("image_reasoners", {}).get(reasoner, {})
            rsn = get_reasoner(reasoner, rsn_cfg, hf_token=hf_token)

        audit = AuditLogger(audit_log) if audit_log else None

        return cls(
            classifier=clf,
            reasoner=rsn,
            taxonomy=taxonomy,
            policy_engine=policy_engine,
            audit_logger=audit,
            policy_descriptions=policy_descriptions,
        )

    def _build_policy_text(self, triggered_categories: list[str]) -> str:
        """Build combined policy text for reasoner from triggered categories."""
        parts = []
        for cat in triggered_categories:
            desc = self._policy_descriptions.get(cat)
            if desc:
                parts.append(desc.strip())
        if not parts:
            # Fall back to generic bank policy
            parts = [
                desc.strip()
                for desc in self._policy_descriptions.values()
            ]
        return "\n\n".join(parts) if parts else "Do not generate harmful, explicit, or prohibited content."

    def run(
        self,
        image_path: Union[str, Path],
        user_id: str | None = None,
    ) -> Verdict:
        t0 = time.perf_counter()
        image_path = Path(image_path)
        image_id = _image_sha256(image_path)

        # --- Layer 1: Fast classifier ---
        clf_result: ClassifierResult = self.classifier.classify(image_path)
        max_conf = clf_result.max_score()
        top_cat = clf_result.top_category()

        # Guard: NaN scores arise when float16 overflows in PaliGemma-2 attention.
        # max_score() already coerces NaN→0.0 for control-flow purposes, but we must
        # also sanitize clf_result.scores so the Verdict never contains NaN.
        # Only the NaN entries are zeroed; valid scores are preserved.
        import math
        nan_keys = [k for k, v in clf_result.scores.items() if math.isnan(v)]
        if nan_keys:
            logger.warning(
                "Classifier returned NaN scores for %s (categories: %s) — "
                "check torch_dtype (must be bfloat16, not float16) and "
                "ShieldGemma2Processor API (use policies= kwarg). Coercing NaN→0.0.",
                image_path,
                nan_keys,
            )
            clf_result.scores = {
                k: (0.0 if math.isnan(v) else v)
                for k, v in clf_result.scores.items()
            }
            max_conf = clf_result.max_score()

        if max_conf >= self.policy.fast_threshold_block:
            # High-confidence detection → immediate BLOCK
            decision = Decision.BLOCK
            rationale = (
                f"High-confidence detection by {self.classifier.model_name} "
                f"(score={max_conf:.3f} ≥ {self.policy.fast_threshold_block})."
            )
            reasoner_model = None
            reasoner_conf = max_conf
            primary_category = top_cat

        elif max_conf < self.policy.fast_threshold_allow:
            # Low confidence → ALLOW
            decision = Decision.ALLOW
            rationale = "No unsafe content detected by classifier."
            reasoner_model = None
            reasoner_conf = max_conf
            primary_category = None

        else:
            # Gray zone → escalate to Layer 2
            if self.reasoner:
                policy_text = self._build_policy_text(clf_result.triggered_categories or [top_cat] if top_cat else [])
                reason_result = self.reasoner.reason(image_path, policy_text)
                decision = Decision(reason_result.verdict)
                rationale = reason_result.rationale
                reasoner_model = reason_result.model
                reasoner_conf = reason_result.confidence
                primary_category = reason_result.category or top_cat
            else:
                decision = Decision.REVIEW
                rationale = "Uncertain — manual review required (no reasoner configured)."
                reasoner_model = None
                reasoner_conf = max_conf
                primary_category = top_cat

        latency = (time.perf_counter() - t0) * 1000

        # Apply per-category policy thresholds (overrides fast thresholds for fine-grained logic)
        if decision == Decision.BLOCK and top_cat:
            policy_dec = self.policy.evaluate(clf_result.scores)
            if policy_dec.decision == Decision.ALLOW:
                decision = Decision.ALLOW
                rationale = policy_dec.reason

        verdict = Verdict(
            image_id=image_id,
            timestamp=_now_iso(),
            decision=decision.value,
            classifier_model=self.classifier.model_name,
            classifier_scores=clf_result.scores,
            classifier_triggered=clf_result.triggered_categories,
            reasoner_model=reasoner_model if self.reasoner and max_conf >= self.policy.fast_threshold_allow else None,
            reasoner_rationale=rationale,
            reasoner_confidence=reasoner_conf,
            primary_category=primary_category,
            explanation_for_user=self._user_explanation(decision, primary_category, max_conf),
            explanation_for_regulator=self._regulator_explanation(
                decision, primary_category, max_conf, self.classifier.model_name
            ),
            latency_ms=latency,
            pipeline_version=self.VERSION,
        )

        if self.audit_logger:
            self.audit_logger.log(verdict, user_id=user_id)

        return verdict

    def run_batch(
        self,
        image_paths: list[Union[str, Path]],
        user_id: str | None = None,
    ) -> list[Verdict]:
        return [self.run(p, user_id=user_id) for p in image_paths]

    @staticmethod
    def _user_explanation(decision: Decision, category: str | None, confidence: float) -> str:
        if decision == Decision.ALLOW:
            return "Image approved. No prohibited content detected."
        if decision == Decision.BLOCK:
            cat_str = f" ({category})" if category else ""
            return f"Image blocked: prohibited content detected{cat_str}."
        return "Image requires manual review."

    @staticmethod
    def _regulator_explanation(
        decision: Decision, category: str | None, confidence: float, model: str
    ) -> str:
        if decision == Decision.ALLOW:
            return f"Image classified as safe by {model}. Max confidence: {confidence:.3f}."
        if decision == Decision.BLOCK:
            return (
                f"Image blocked by {model}. "
                f"Primary category: {category or 'unknown'}. "
                f"Confidence: {confidence:.3f}."
            )
        return f"Image requires human review. Confidence {confidence:.3f} in uncertain range."
