from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from censor_guard.schemas import SignalResult


class OCRAdapter:
    name = "ocr_adapter"
    ocr_languages = ("rus", "eng", "rus+eng")

    def __init__(
        self,
        enabled: bool = True,
        tesseract_cmd: str | None = None,
        tessdata_dir: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.tesseract_cmd = tesseract_cmd or None
        self.tessdata_dir = tessdata_dir or None

    def _resolve_tessdata_dir(self) -> Path | None:
        if self.tessdata_dir:
            candidate = Path(self.tessdata_dir)
            if candidate.exists():
                return candidate
        if not self.tesseract_cmd:
            return None
        tesseract_path = Path(self.tesseract_cmd)
        candidate = tesseract_path.parent / "tessdata"
        if candidate.exists():
            return candidate
        return None

    def _available_languages(self) -> set[str]:
        tessdata_dir = self._resolve_tessdata_dir()
        if tessdata_dir is None:
            return set()
        return {path.stem for path in tessdata_dir.glob("*.traineddata")}

    def _extract_with_lang(self, pytesseract: Any, image, lang: str) -> tuple[str, float]:
        tessdata_dir = self._resolve_tessdata_dir()
        if tessdata_dir is not None:
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )
        words = []
        confidences = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            token = (text or "").strip()
            if not token:
                continue
            words.append(token)
            try:
                conf_value = float(conf)
            except (TypeError, ValueError):
                continue
            if conf_value >= 0:
                confidences.append(conf_value)
        mean_confidence = sum(confidences) / len(confidences) if confidences else -1.0
        return "\n".join(words).strip(), mean_confidence

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

        available_languages = self._available_languages()
        configured_languages = []
        for lang in self.ocr_languages:
            parts = lang.split("+")
            if not available_languages or all(part in available_languages for part in parts):
                configured_languages.append(lang)
        skipped_languages = [lang for lang in self.ocr_languages if lang not in configured_languages]

        if not configured_languages:
            missing = ", ".join(self.ocr_languages)
            found = ", ".join(sorted(available_languages)) or "none"
            return SignalResult(
                name=self.name,
                status="error",
                reason=(
                    f"OCR language data is missing. Required one of: {missing}. "
                    f"Found installed languages: {found}."
                ),
                raw={
                    "requested_languages": list(self.ocr_languages),
                    "available_languages": sorted(available_languages),
                    "tessdata_dir": str(self._resolve_tessdata_dir()) if self._resolve_tessdata_dir() else None,
                },
            )

        try:
            candidates = []
            for lang in configured_languages:
                text, confidence = self._extract_with_lang(pytesseract, image, lang)
                candidates.append((confidence, text, lang))
        except Exception as exc:  # pragma: no cover - backend-specific failures
            return SignalResult(
                name=self.name,
                status="error",
                reason=f"OCR backend failed for languages {', '.join(configured_languages)}: {exc}",
            )

        _, text, selected_lang = max(candidates, key=lambda item: (item[0], len(item[1])))
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return SignalResult(
                name=self.name,
                status="ok",
                reason="No readable text found in image.",
                raw={
                    "selected_lang": selected_lang,
                    "requested_languages": list(self.ocr_languages),
                    "available_languages": sorted(available_languages),
                    "skipped_languages": skipped_languages,
                },
            )
        return SignalResult(
            name=self.name,
            status="ok",
            text=lines,
            reason=f"Extracted text from image via OCR ({selected_lang}).",
            raw={
                "line_count": len(lines),
                "selected_lang": selected_lang,
                "requested_languages": list(self.ocr_languages),
                "available_languages": sorted(available_languages),
                "skipped_languages": skipped_languages,
                "candidates": [
                    {"lang": lang, "confidence": confidence, "text": candidate_text}
                    for confidence, candidate_text, lang in candidates
                ],
            },
        )
