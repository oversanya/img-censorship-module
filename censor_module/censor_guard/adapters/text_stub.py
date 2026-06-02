from __future__ import annotations

from censor_guard.schemas import SignalResult


class TextGuardStub:
    name = "text_guard_stub"

    def moderate(self, text: str | None) -> SignalResult:
        if not text:
            return SignalResult(
                name=self.name,
                status="skipped",
                reason="No prompt text supplied.",
            )
        return SignalResult(
            name=self.name,
            status="ok",
            reason="Placeholder adapter: prompt text is treated as safe in MVP.",
            raw={"mode": "stub", "text_length": len(text)},
        )

