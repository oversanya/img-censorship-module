"""Abstract base class for Layer-2 VLM reasoners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from censorship.core.verdict import ReasonerResult


class ImageReasoner(ABC):
    """
    Layer 2 — VLM with extended reasoning.
    Called only when Layer-1 confidence falls in the gray zone (0.50–0.90).
    Must return a human-readable rationale.
    """

    model_name: str

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def reason(
        self,
        image_path: Union[str, Path],
        policy_text: str,
    ) -> ReasonerResult:
        """
        Analyse an image against the given policy and produce a verdict with rationale.

        Args:
            image_path: path to the image
            policy_text: full policy description to check against

        Returns:
            ReasonerResult with verdict, category, confidence, and rationale
        """
        ...

    def is_loaded(self) -> bool:
        return False
