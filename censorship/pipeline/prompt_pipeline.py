"""Text-only prompt safety pipeline."""

from __future__ import annotations

import time
import logging
import os
from pathlib import Path

import yaml

from censorship.core.verdict import PromptGuardResult, _now_iso
from censorship.prompt_guard.base import PromptGuard
from censorship.prompt_guard.registry import get_prompt_guard

logger = logging.getLogger(__name__)


class PromptPipeline:
    """
    Standalone text-prompt safety check.

    Usage:
        pipeline = PromptPipeline.from_config("config/models.yaml")
        result = pipeline.check("Generate an image of ...")
    """

    def __init__(self, guard: PromptGuard):
        self.guard = guard

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config/models.yaml",
        guard: str = "llamaguard4",
        hf_token: str | None = None,
    ) -> "PromptPipeline":
        config_path = Path(config_path)
        with open(config_path) as f:
            models_cfg = yaml.safe_load(f)

        hf_token = hf_token or os.environ.get("HF_TOKEN")
        guard_cfg = models_cfg.get("prompt_guards", {}).get(guard, {})
        prompt_guard = get_prompt_guard(guard, guard_cfg, hf_token=hf_token)

        return cls(guard=prompt_guard)

    def check(self, text: str) -> PromptGuardResult:
        return self.guard.check(text)
