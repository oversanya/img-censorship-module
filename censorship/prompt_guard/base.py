"""Abstract base class for prompt guard modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from censorship.core.verdict import PromptGuardResult


class PromptGuard(ABC):
    """Checks a text prompt for policy violations before image generation."""

    model_name: str

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        ...

    @abstractmethod
    def check(self, text: str) -> PromptGuardResult:
        """
        Check a text prompt for unsafe content.

        Returns:
            PromptGuardResult with verdict (ALLOW/BLOCK), category, and rationale
        """
        ...

    def is_loaded(self) -> bool:
        return False
