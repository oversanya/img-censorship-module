from typing import List

from img_censor.detectors.base import Detector
from img_censor.detectors.heuristics import PromptKeywordGuard
from img_censor.io import load_image
from img_censor.schemas import Finding, GuardRequest


class OcrTextDetector(Detector):
    name = "ocr_text"

    def run(self, request: GuardRequest) -> List[Finding]:
        if not self.enabled:
            return []

        findings: List[Finding] = []
        for stage, image_path in (
            ("input_image", request.input_image),
            ("output_image", request.output_image),
        ):
            if not image_path or not self.should_run_on(stage):
                continue
            text = self._extract_text(image_path)
            if not text:
                continue
            findings.extend(self._classify_text(stage, text))
        return findings

    def _extract_text(self, image_path: str) -> str:
        try:
            import pytesseract
        except ImportError:
            return ""

        image = load_image(image_path)
        languages = self.config.get("languages", "rus+eng")
        try:
            return pytesseract.image_to_string(image, lang=languages).strip()
        except Exception:
            return ""

    def _classify_text(self, stage: str, text: str) -> List[Finding]:
        keyword_guard = PromptKeywordGuard(
            {"enabled": True, "stages": ["prompt"], "block_threshold": 1.0},
            self.runtime_config,
        )
        findings = keyword_guard.run(GuardRequest(prompt=text))
        converted = []
        for finding in findings:
            converted.append(
                Finding(
                    detector=self.name,
                    stage=stage,
                    category=finding.category,
                    score=finding.score,
                    verdict=finding.verdict,
                    rationale="OCR text embedded in image matched prohibited policy keywords.",
                    raw={
                        **finding.raw,
                        "extracted_text": text[:1000],
                    },
                )
            )
        return converted


def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True
