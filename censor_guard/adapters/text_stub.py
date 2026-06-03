from __future__ import annotations

from censor_guard.schemas import SignalResult


class TextGuardStub:
    """ЗАГЛУШКА текстового гарда. Сейчас любой непустой текст считается
    безопасным (status="ok" без категорий). Это означает, что ни промпт, ни
    извлечённый OCR-текст фактически НЕ модерируются — контракт адаптера готов,
    а реальная проверка должна быть подключена позже (см. PLAN.md)."""

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

