"""Classifier registry — factory for ImageClassifier instances."""

from __future__ import annotations

import os
from typing import Any

from .base import ImageClassifier

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


# Lazy imports to avoid loading ML frameworks at import time

def _get_shieldgemma2(config: dict, **kwargs) -> ImageClassifier:
    from .shieldgemma2 import ShieldGemma2Classifier
    return ShieldGemma2Classifier(
        hf_id=config.get("hf_id", "google/shieldgemma-2-4b-it"),
        torch_dtype=config.get("torch_dtype", "bfloat16"),
        hf_token=kwargs.get("hf_token") or os.environ.get("HF_TOKEN"),
    )


def _get_nudenet(config: dict, **kwargs) -> ImageClassifier:
    from .nudenet import NudeNetClassifier
    return NudeNetClassifier()


def _get_q16(config: dict, **kwargs) -> ImageClassifier:
    from .q16 import Q16Classifier
    return Q16Classifier(
        hf_id=config.get("hf_id"),
        hf_token=kwargs.get("hf_token") or os.environ.get("HF_TOKEN"),
    )


CLASSIFIER_REGISTRY: dict[str, Any] = {
    "shieldgemma2": _get_shieldgemma2,
    "nudenet": _get_nudenet,
    "q16": _get_q16,
}


def get_classifier(name: str, config: dict | None = None, **kwargs) -> ImageClassifier:
    """
    Factory: returns an unloaded ImageClassifier by name.

    Args:
        name: classifier name (shieldgemma2 | nudenet | q16)
        config: model config dict from models.yaml
        **kwargs: extra args (e.g., hf_token)
    """
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{name}'. Available: {list(CLASSIFIER_REGISTRY)}"
        )
    return CLASSIFIER_REGISTRY[name](config or {}, **kwargs)
