"""Combined pipeline: prompt guard + image pipeline.

Prompt is checked first. If blocked, image scan is skipped.
"""

from __future__ import annotations

import time
import logging
import os
from pathlib import Path
from typing import Union

import yaml

from censorship.core.verdict import Verdict, _image_sha256, _now_iso
from censorship.pipeline.image_pipeline import ImagePipeline
from censorship.pipeline.prompt_pipeline import PromptPipeline
from censorship.prompt_guard.registry import get_prompt_guard
from censorship.audit.logger import AuditLogger

logger = logging.getLogger(__name__)


class CombinedPipeline:
    """
    Full pipeline: text prompt → image.

    If the prompt is blocked, returns a Verdict immediately without running
    the image classifier (saves latency, enforces defense-in-depth).

    Usage:
        pipeline = CombinedPipeline.from_config("config/models.yaml")
        verdict = pipeline.run(image_path="image.jpg", prompt="beach scene")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        image_pipeline: ImagePipeline,
        prompt_pipeline: PromptPipeline | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.image_pipeline = image_pipeline
        self.prompt_pipeline = prompt_pipeline
        self.audit_logger = audit_logger

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config/models.yaml",
        taxonomy_path: str | Path = "config/taxonomy.yaml",
        policy_path: str | Path = "config/policy_bank.yaml",
        classifier: str = "shieldgemma2",
        reasoner: str | None = "shieldgemma2_reason",
        prompt_guard: str | None = "llamaguard4",
        audit_log: str | None = "audit.jsonl",
        hf_token: str | None = None,
    ) -> "CombinedPipeline":
        hf_token = hf_token or os.environ.get("HF_TOKEN")

        img_pipeline = ImagePipeline.from_config(
            config_path=config_path,
            taxonomy_path=taxonomy_path,
            policy_path=policy_path,
            classifier=classifier,
            reasoner=reasoner,
            audit_log=None,  # audit at combined level
            hf_token=hf_token,
        )

        pmt_pipeline = None
        if prompt_guard:
            pmt_pipeline = PromptPipeline.from_config(
                config_path=config_path,
                guard=prompt_guard,
                hf_token=hf_token,
            )

        audit = AuditLogger(audit_log) if audit_log else None
        return cls(image_pipeline=img_pipeline, prompt_pipeline=pmt_pipeline, audit_logger=audit)

    def run(
        self,
        image_path: Union[str, Path] | None = None,
        prompt: str | None = None,
        user_id: str | None = None,
    ) -> Verdict:
        t0 = time.perf_counter()

        # Step 1: Check prompt (if provided)
        prompt_verdict = None
        prompt_category = None

        if prompt and self.prompt_pipeline:
            pg_result = self.prompt_pipeline.check(prompt)
            prompt_verdict = pg_result.verdict
            prompt_category = pg_result.category

            if pg_result.verdict == "BLOCK":
                # Short-circuit: blocked at prompt level
                latency = (time.perf_counter() - t0) * 1000
                verdict = Verdict(
                    image_id="prompt_only",
                    timestamp=_now_iso(),
                    decision="BLOCK",
                    classifier_model="prompt_guard",
                    classifier_scores={},
                    classifier_triggered=[pg_result.category] if pg_result.category else [],
                    primary_category=pg_result.category,
                    explanation_for_user=f"Request blocked at prompt level: prohibited content ({pg_result.category}).",
                    explanation_for_regulator=(
                        f"Text prompt blocked by {pg_result.model}. "
                        f"Category: {pg_result.category}. "
                        f"Rationale: {pg_result.rationale}"
                    ),
                    prompt_verdict=pg_result.verdict,
                    prompt_category=pg_result.category,
                    latency_ms=latency,
                    pipeline_version=self.VERSION,
                )
                if self.audit_logger:
                    self.audit_logger.log(verdict, user_id=user_id)
                return verdict

        # Step 2: Image scan (if image provided)
        if image_path:
            verdict = self.image_pipeline.run(image_path, user_id=user_id)
            # Attach prompt verdict to the image verdict
            verdict.prompt_verdict = prompt_verdict
            verdict.prompt_category = prompt_category
            if self.audit_logger:
                self.audit_logger.log(verdict, user_id=user_id)
            return verdict

        # Prompt only, no image, not blocked
        if prompt and not image_path:
            latency = (time.perf_counter() - t0) * 1000
            return Verdict(
                image_id="prompt_only",
                timestamp=_now_iso(),
                decision="ALLOW",
                classifier_model="prompt_guard",
                classifier_scores={},
                classifier_triggered=[],
                explanation_for_user="Prompt approved. No prohibited content detected.",
                explanation_for_regulator="Prompt classified as safe.",
                prompt_verdict=prompt_verdict or "ALLOW",
                prompt_category=None,
                latency_ms=latency,
                pipeline_version=self.VERSION,
            )

        raise ValueError("At least one of image_path or prompt must be provided.")
