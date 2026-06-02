"""ShieldGemma text guard via Ollama (local, no token required).

Requires Ollama running locally with `shieldgemma:2b` pulled.
"""

from __future__ import annotations

import json
import time
import logging

import requests

from censorship.core.verdict import PromptGuardResult
from .base import PromptGuard

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://localhost:11434"

_SYSTEM_PROMPT = """You are a content safety classifier for a bank AI system.
Determine if the following user prompt requests prohibited content.

Prohibited categories:
- Sexually explicit content or nudity
- Violence, gore, or graphic injury
- Hate speech or extremist content
- Terrorist or extremist propaganda
- Child sexual abuse material (CSAM)
- Exposure of personal/financial data
- Financial fraud or forgery

Respond in this exact format:
VERDICT: [SAFE or UNSAFE]
CATEGORY: [category name or NONE]
RATIONALE: [one sentence explanation]"""


class ShieldGemmaTextGuard(PromptGuard):
    model_name = "shieldgemma-text-2b"

    def __init__(self, ollama_host: str = _DEFAULT_HOST, model_id: str = "shieldgemma:2b"):
        self.ollama_host = ollama_host
        self.model_id = model_id
        self._available = False

    def load(self) -> None:
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            self._available = any(self.model_id in m for m in models)
            if not self._available:
                logger.warning(
                    f"Ollama model '{self.model_id}' not found. "
                    f"Run: ollama pull {self.model_id}"
                )
        except Exception as e:
            logger.warning(f"Ollama not available at {self.ollama_host}: {e}")

    def is_loaded(self) -> bool:
        return self._available

    def check(self, text: str) -> PromptGuardResult:
        if not self.is_loaded():
            self.load()

        t0 = time.perf_counter()

        try:
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"User prompt: {text}"},
                ],
                "stream": False,
            }
            resp = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            response_text = resp.json()["message"]["content"]
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(f"Ollama request failed: {e}")
            return PromptGuardResult(
                model=self.model_name,
                verdict="REVIEW",
                category=None,
                confidence=0.0,
                rationale=f"Ollama unavailable: {e}",
                latency_ms=latency,
            )

        latency = (time.perf_counter() - t0) * 1000

        verdict = "ALLOW"
        category = None
        rationale = response_text.strip()

        for line in response_text.splitlines():
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                val = line.split(":", 1)[1].strip().upper()
                verdict = "BLOCK" if "UNSAFE" in val else "ALLOW"
            elif line.upper().startswith("CATEGORY:"):
                val = line.split(":", 1)[1].strip()
                if val.upper() != "NONE":
                    category = val.lower()
            elif line.upper().startswith("RATIONALE:"):
                rationale = line.split(":", 1)[1].strip()

        return PromptGuardResult(
            model=self.model_name,
            verdict=verdict,
            category=category,
            confidence=0.85 if verdict == "BLOCK" else 0.90,
            rationale=rationale,
            latency_ms=latency,
        )
