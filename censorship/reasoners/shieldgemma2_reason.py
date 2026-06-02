"""ShieldGemma-2 classifier-based reasoner.

google/shieldgemma-2-4b-it is a classification model (ShieldGemma2ForImageClassification),
NOT a generative model. Loading it as AutoModelForImageTextToText initialises most
language-model weights from scratch (random), making generated text meaningless.

This reasoner uses the classification head correctly via ShieldGemma2Processor with the
policies= API: runs all 3 built-in policies, picks the highest-scoring one, and
constructs a structured rationale from the per-policy violation probabilities.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Union

from censorship.core.verdict import ReasonerResult
from censorship.core.device_utils import auto_device_dtype, load_to_device
from .base import ImageReasoner

logger = logging.getLogger(__name__)

# Built-in ShieldGemma-2 policy keys → our taxonomy category names
_POLICY_TO_CATEGORY: dict[str, str] = {
    "dangerous": "extremism",
    "sexual":    "sexual_explicit",
    "violence":  "violence_gore",
}


class ShieldGemma2Reasoner(ImageReasoner):
    """ShieldGemma-2 reasoner — uses classification scores as structured reasoning."""

    model_name = "shieldgemma2-reason"

    def __init__(
        self,
        hf_id: str = "google/shieldgemma-2-4b-it",
        torch_dtype: str = "bfloat16",
        hf_token: str | None = None,
    ):
        self.hf_id = hf_id
        self.torch_dtype = torch_dtype
        self.hf_token = hf_token
        self._model = None
        self._processor = None

    def load(self) -> None:
        from transformers import ShieldGemma2Processor, ShieldGemma2ForImageClassification

        device, dtype = auto_device_dtype(self.torch_dtype)
        logger.info(f"Loading {self.hf_id} as reasoner on {device} ({dtype}) ...")
        self._processor = ShieldGemma2Processor.from_pretrained(
            self.hf_id, token=self.hf_token
        )
        self._model = ShieldGemma2ForImageClassification.from_pretrained(
            self.hf_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            token=self.hf_token,
        )
        self._model = load_to_device(self._model, device)
        self._model.eval()
        logger.info(f"{self.hf_id} reasoner loaded on {self._model.device} dtype={self._model.dtype}.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def reason(
        self,
        image_path: Union[str, Path],
        policy_text: str,
    ) -> ReasonerResult:
        if not self.is_loaded():
            self.load()

        import torch
        from PIL import Image

        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")

        # Run all 3 built-in policies in separate forward passes (avoid padding NaN)
        scores: dict[str, float] = {}
        for policy_key, cat_name in _POLICY_TO_CATEGORY.items():
            inputs = self._processor(
                images=image,
                policies=[policy_key],
                return_tensors="pt",
            ).to(self._model.device)

            with torch.inference_mode():
                outputs = self._model(**inputs)

            prob = float(outputs.probabilities[0, 0].float().cpu())
            scores[cat_name] = prob

            del inputs, outputs
            if self._model.device.type == "mps":
                torch.mps.empty_cache()

        latency = (time.perf_counter() - t0) * 1000

        max_score = max(scores.values())
        top_cat = max(scores, key=scores.__getitem__)

        if max_score >= 0.90:
            verdict = "BLOCK"
        elif max_score >= 0.50:
            verdict = "BLOCK"
        else:
            verdict = "ALLOW"
            top_cat = None

        score_lines = "  ".join(f"{cat}={v:.3f}" for cat, v in scores.items())
        if verdict == "ALLOW":
            rationale = (
                f"ShieldGemma-2 found no policy violations. "
                f"Scores: {score_lines}."
            )
        else:
            rationale = (
                f"ShieldGemma-2 detected potential violation in '{top_cat}' "
                f"(score={max_score:.3f}). All scores: {score_lines}."
            )

        return ReasonerResult(
            model=self.model_name,
            verdict=verdict,
            category=top_cat,
            confidence=max_score,
            rationale=rationale,
            raw_response=score_lines,
            latency_ms=latency,
        )
