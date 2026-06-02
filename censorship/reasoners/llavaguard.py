"""LlavaGuard reasoner — supports both 0.5B and 7B variants.

Uses generic AutoProcessor + AutoModelForVision2Seq so it works with
any underlying architecture (Qwen2-VL for 0.5B, LLaVA-Next for 7B).

Default: AIML-TUDA/LlavaGuard-0.5B (~1 GB, runs on MPS/CPU).
"""

from __future__ import annotations

import re
import time
import logging
from pathlib import Path
from typing import Union

from censorship.core.verdict import ReasonerResult
from censorship.core.device_utils import auto_device_dtype, load_to_device
from .base import ImageReasoner

logger = logging.getLogger(__name__)

_DEFAULT_HF_ID = "AIML-TUDA/LlavaGuard-0.5B"

_SYSTEM_PROMPT = (
    "You are a content safety expert reviewing images for policy compliance. "
    "Analyze the provided image and determine if it violates the given safety policy. "
    "Respond in the following format:\n"
    "VERDICT: [SAFE or UNSAFE]\n"
    "CATEGORY: [harm category or NONE]\n"
    "CONFIDENCE: [0.0 to 1.0]\n"
    "RATIONALE: [detailed explanation of your assessment]"
)

_USER_TEMPLATE = (
    "Please review this image for the following policy:\n\n"
    "{policy_text}\n\n"
    "Provide your safety assessment."
)


def _parse_llavaguard_response(text: str) -> dict:
    """Parse structured LlavaGuard output into components."""
    result = {
        "verdict": "REVIEW",
        "category": None,
        "confidence": 0.5,
        "rationale": text.strip(),
    }

    verdict_match = re.search(r"VERDICT:\s*(SAFE|UNSAFE)", text, re.IGNORECASE)
    if verdict_match:
        result["verdict"] = "BLOCK" if verdict_match.group(1).upper() == "UNSAFE" else "ALLOW"

    category_match = re.search(r"CATEGORY:\s*([^\n]+)", text, re.IGNORECASE)
    if category_match:
        cat = category_match.group(1).strip()
        result["category"] = None if cat.upper() == "NONE" else cat.lower()

    conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", text, re.IGNORECASE)
    if conf_match:
        try:
            result["confidence"] = float(conf_match.group(1))
        except ValueError:
            pass

    rationale_match = re.search(r"RATIONALE:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if rationale_match:
        result["rationale"] = rationale_match.group(1).strip()

    return result


class LlavaGuardReasoner(ImageReasoner):
    """VLM reasoner using LlavaGuard. Default: 0.5B (~1 GB, MPS-compatible)."""

    def __init__(
        self,
        hf_id: str = _DEFAULT_HF_ID,
        torch_dtype: str = "float16",
        hf_token: str | None = None,
        max_new_tokens: int = 512,
    ):
        self.hf_id = hf_id
        self.torch_dtype = torch_dtype
        self.hf_token = hf_token
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        # Derive display name from model ID (e.g. "LlavaGuard-0.5B")
        self.model_name = hf_id.split("/")[-1].lower()

    def load(self) -> None:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        device, dtype = auto_device_dtype(self.torch_dtype)
        logger.info(f"Loading {self.hf_id} on {device} ({dtype}) ...")
        self._processor = AutoProcessor.from_pretrained(
            self.hf_id, token=self.hf_token
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.hf_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            token=self.hf_token,
        )
        self._model = load_to_device(self._model, device)
        self._model.eval()
        logger.info(f"{self.hf_id} loaded on {self._model.device}.")

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

        conversation = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _USER_TEMPLATE.format(policy_text=policy_text)},
                ],
            },
        ]

        prompt = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self._processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated = self._processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        parsed = _parse_llavaguard_response(generated)
        latency = (time.perf_counter() - t0) * 1000

        return ReasonerResult(
            model=self.model_name,
            verdict=parsed["verdict"],
            category=parsed["category"],
            confidence=parsed["confidence"],
            rationale=parsed["rationale"],
            raw_response=generated,
            latency_ms=latency,
        )
