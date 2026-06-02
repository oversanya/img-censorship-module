"""LlamaGuard prompt guard — default: Llama-Guard-3-1B (~2 GB, MPS-compatible).

LlamaGuard outputs a structured response:
  safe     → ALLOW
  unsafe\n<S1>  → BLOCK with category code

Model comparison:
  Llama-Guard-3-1B   — 1B params,  ~2 GB,  fast on MPS/CPU  (default)
  Llama-Guard-4-12B  — 12B params, ~24 GB, needs GPU/large RAM
"""

from __future__ import annotations

import re
import time
import logging

from censorship.core.verdict import PromptGuardResult
from censorship.core.device_utils import auto_device_dtype, load_to_device
from .base import PromptGuard

logger = logging.getLogger(__name__)

_HF_ID = "meta-llama/Llama-Guard-3-1B"  # 1B ← was 12B

# LlamaGuard category codes → taxonomy IDs
_CATEGORY_MAP = {
    "S1": "violence_gore",
    "S2": "sexual_explicit",
    "S3": "hate_speech",
    "S4": "extremism",
    "S5": "financial_fraud",
    "S6": "personal_data",
    "S7": "csam",
    "S8": "extremism",
    "S9": "violence_gore",
    "S10": "violence_gore",
    "S11": "violence_gore",
    "S12": "personal_data",
    "S13": "financial_fraud",
    "S14": "csam",
}

# Bank-specific policy for LlamaGuard
_POLICY = """The following types of content are prohibited:
1. Sexual content or nudity
2. Violence, gore, or graphic injury
3. Hate speech, discrimination, or extremist symbols
4. Terrorist propaganda or incitement to violence
5. Child sexual abuse material (CSAM) — zero tolerance
6. Personal data exposure (passport, bank cards, PII)
7. Financial fraud materials (forged documents, phishing)"""


class LlamaGuard4PromptGuard(PromptGuard):
    model_name = "llamaguard-3-1b"

    def __init__(
        self,
        hf_id: str = _HF_ID,
        torch_dtype: str = "float16",
        hf_token: str | None = None,
    ):
        self.hf_id = hf_id
        self.torch_dtype = torch_dtype
        self.hf_token = hf_token
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device, dtype = auto_device_dtype(self.torch_dtype)
        logger.info(f"Loading {self.hf_id} on {device} ({dtype}) ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, token=self.hf_token
        )
        self._model = AutoModelForCausalLM.from_pretrained(
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

    def check(self, text: str) -> PromptGuardResult:
        if not self.is_loaded():
            self.load()

        import torch

        t0 = time.perf_counter()

        conversation = [
            {"role": "user", "content": f"{_POLICY}\n\nUser prompt: {text}"},
        ]

        input_ids = self._tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            output[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        latency = (time.perf_counter() - t0) * 1000

        # Parse LlamaGuard response format
        if response.lower().startswith("safe"):
            return PromptGuardResult(
                model=self.model_name,
                verdict="ALLOW",
                category=None,
                confidence=0.95,
                rationale="Prompt classified as safe by LlamaGuard-4.",
                latency_ms=latency,
            )

        # unsafe\n<S_code>
        category = None
        cat_match = re.search(r"S(\d+)", response)
        if cat_match:
            code = f"S{cat_match.group(1)}"
            category = _CATEGORY_MAP.get(code, "unknown")

        return PromptGuardResult(
            model=self.model_name,
            verdict="BLOCK",
            category=category,
            confidence=0.90,
            rationale=f"Prompt classified as unsafe by LlamaGuard-4. Raw: {response}",
            latency_ms=latency,
        )
