"""Prompt guard registry."""

from __future__ import annotations

import os
from typing import Any

from .base import PromptGuard


def _get_llamaguard4(config: dict, **kwargs) -> PromptGuard:
    from .llamaguard4 import LlamaGuard4PromptGuard
    return LlamaGuard4PromptGuard(
        hf_id=config.get("hf_id", "meta-llama/Llama-Guard-3-1B"),
        torch_dtype=config.get("torch_dtype", "float16"),
        hf_token=kwargs.get("hf_token") or os.environ.get("HF_TOKEN"),
    )


def _get_shieldgemma_text(config: dict, **kwargs) -> PromptGuard:
    from .shieldgemma_text import ShieldGemmaTextGuard
    return ShieldGemmaTextGuard(
        ollama_host=config.get("ollama_host", "http://localhost:11434"),
        model_id=config.get("ollama_id", "shieldgemma:2b"),
    )


PROMPT_GUARD_REGISTRY: dict[str, Any] = {
    "llamaguard4": _get_llamaguard4,
    "shieldgemma_text": _get_shieldgemma_text,
}


def get_prompt_guard(name: str, config: dict | None = None, **kwargs) -> PromptGuard:
    if name not in PROMPT_GUARD_REGISTRY:
        raise ValueError(
            f"Unknown prompt guard '{name}'. Available: {list(PROMPT_GUARD_REGISTRY)}"
        )
    return PROMPT_GUARD_REGISTRY[name](config or {}, **kwargs)
