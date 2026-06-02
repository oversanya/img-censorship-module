"""Reasoner registry — factory for ImageReasoner instances."""

from __future__ import annotations

import os
from typing import Any

from .base import ImageReasoner


def _get_llavaguard(config: dict, **kwargs) -> ImageReasoner:
    from .llavaguard import LlavaGuardReasoner
    return LlavaGuardReasoner(
        hf_id=config.get("hf_id", "AIML-TUDA/LlavaGuard-0.5B"),
        torch_dtype=config.get("torch_dtype", "float16"),
        hf_token=kwargs.get("hf_token") or os.environ.get("HF_TOKEN"),
    )


def _get_shieldgemma2_reason(config: dict, **kwargs) -> ImageReasoner:
    from .shieldgemma2_reason import ShieldGemma2Reasoner
    return ShieldGemma2Reasoner(
        hf_id=config.get("hf_id", "google/shieldgemma-2-4b-it"),
        torch_dtype=config.get("torch_dtype", "bfloat16"),
        hf_token=kwargs.get("hf_token") or os.environ.get("HF_TOKEN"),
    )


REASONER_REGISTRY: dict[str, Any] = {
    "llavaguard": _get_llavaguard,
    "shieldgemma2_reason": _get_shieldgemma2_reason,
}


def get_reasoner(name: str, config: dict | None = None, **kwargs) -> ImageReasoner:
    if name not in REASONER_REGISTRY:
        raise ValueError(
            f"Unknown reasoner '{name}'. Available: {list(REASONER_REGISTRY)}"
        )
    return REASONER_REGISTRY[name](config or {}, **kwargs)
