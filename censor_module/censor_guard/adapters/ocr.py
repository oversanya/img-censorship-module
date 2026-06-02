from __future__ import annotations

from pathlib import Path

from censor_guard.schemas import SignalResult


class OCRAdapter:
    name = "ocr_adapter"

    def __init__(self, enabled: bool = True, tesseract_cmd: str | None = None) -> None:
        self.enabled = enabled
        self.tesseract_cmd = tesseract_cmd or None

    def extract(self, image) -> SignalResult:
        if not self.enabled:
            return SignalResult(name=self.name, status="skipped", reason="OCR disabled by configuration.")
        try:
            import pytesseract
        except ImportError:
            return SignalResult(
                name=self.name,
                status="skipped",
                reason="pytesseract is not installed.",
            )

        if self.tesseract_cmd:
            tesseract_path = Path(self.tesseract_cmd)
            if tesseract_path.exists():
                pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)

        try:
            text = pytesseract.image_to_string(image).strip()
        except Exception as exc:  # pragma: no cover - backend-specific failures
            return SignalResult(
                name=self.name,
                status="error",
                reason=f"OCR backend failed: {exc}",
            )

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return SignalResult(
                name=self.name,
                status="ok",
                reason="No readable text found in image.",
            )
        return SignalResult(
            name=self.name,
            status="ok",
            text=lines,
            reason="Extracted text from image via OCR.",
            raw={"line_count": len(lines)},
        )
