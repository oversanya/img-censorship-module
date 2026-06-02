"""ShieldGemma-2 (google/shieldgemma-2-4b-it) classifier wrapper.

Uses ShieldGemma2ForImageClassification with ShieldGemma2Processor.
The processor's correct API: processor(images=img, policies=[key]) — one policy per call.
Passing a custom `text=` kwarg is ignored by ShieldGemma2Processor; it always generates
prompts from its built-in policy definitions.

outputs.probabilities shape per call: (1, 2)
  [:, 0] = P(Yes) = P(violation)
  [:, 1] = P(No)  = P(safe)

Built-in processor policy keys: "dangerous", "sexual", "violence"
Mapped to our taxonomy: extremism, sexual_explicit, violence_gore

Violence coverage note:
  SG2's "violence" policy targets gore/injury (biological violence).
  Bombs, explosions, and weapons score near 0 on "violence" but high on "dangerous".
  We therefore score violence_gore as max(violence, dangerous) to capture both types.
  Calibration on UnsafeBench (n=9 Violence images): "violence" policy P50=0.000,
  "dangerous" policy captures some (0.68–1.0). No false positives on safe images.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Union

from PIL import Image

from censorship.core.verdict import ClassifierResult
from censorship.core.device_utils import auto_device_dtype, load_to_device
from .base import ImageClassifier

logger = logging.getLogger(__name__)

# All three ShieldGemma-2 built-in policy keys to score per image
_ALL_POLICIES = ("sexual", "violence", "dangerous")


_POLICY_KEY_MAP: dict[str, str] = {
    "sexual_explicit": "sexual",
    "violence_gore":   "violence",
    "extremism":       "dangerous",
}


class ShieldGemma2Classifier(ImageClassifier):
    model_name = "shieldgemma-2-4b-it"
    supported_categories = list(_POLICY_KEY_MAP.keys())

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
        logger.info(f"Loading {self.hf_id} on {device} ({dtype}) ...")
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
        logger.info(f"{self.hf_id} loaded on {self._model.device} dtype={self._model.dtype}.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def _score_one_policy(self, image: "Image.Image", policy_key: str) -> float:
        """One forward pass for one policy. Returns P(Yes) = P(violation)."""
        import torch

        inputs = self._processor(
            images=image,
            policies=[policy_key],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # probabilities[0, 0] = P(Yes=violation), probabilities[0, 1] = P(No=safe)
        result = float(outputs.probabilities[0, 0].float().cpu())

        del inputs, outputs
        if self._model.device.type == "mps":
            torch.mps.empty_cache()

        return result

    def classify(self, image_path: Union[str, Path]) -> ClassifierResult:
        if not self.is_loaded():
            self.load()

        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")

        # Score all 3 built-in policies in a single pass each
        raw: dict[str, float] = {
            p: self._score_one_policy(image, p) for p in _ALL_POLICIES
        }

        # Map raw policy scores → taxonomy categories.
        # violence_gore uses max(violence, dangerous): SG2's "violence" policy focuses on
        # biological gore/injury (P50≈0 on UnsafeBench Violence images), while explosions
        # and weapons score under "dangerous". Taking the max covers both sub-types.
        scores: dict[str, float] = {
            "sexual_explicit": raw["sexual"],
            "violence_gore":   max(raw["violence"], raw["dangerous"]),
            "extremism":       raw["dangerous"],
        }

        latency = (time.perf_counter() - t0) * 1000
        triggered = [c for c, s in scores.items() if s >= 0.50]

        return ClassifierResult(
            model=self.model_name,
            scores=scores,
            is_unsafe=any(s >= 0.50 for s in scores.values()),
            triggered_categories=triggered,
            latency_ms=latency,
        )
