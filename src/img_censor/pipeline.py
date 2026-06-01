import time
from typing import Dict, List

from img_censor.aggregator import aggregate
from img_censor.detectors import (
    ClipZeroShotDetector,
    LlavaGuardDetector,
    NsfwVitDetector,
    OcrTextDetector,
    PromptKeywordGuard,
    PromptToxicityDetector,
    PromptZeroShotDetector,
    QrCodeDetector,
    ShieldGemmaDetector,
)
from img_censor.review_queue import maybe_enqueue_review
from img_censor.schemas import Finding, GuardRequest, GuardResult


DETECTOR_CLASSES = {
    "prompt_keywords": PromptKeywordGuard,
    "prompt_toxicity": PromptToxicityDetector,
    "prompt_zero_shot": PromptZeroShotDetector,
    "nsfw_vit": NsfwVitDetector,
    "ocr_text": OcrTextDetector,
    "qr_code": QrCodeDetector,
    "llavaguard": LlavaGuardDetector,
    "clip_zero_shot": ClipZeroShotDetector,
    "shieldgemma": ShieldGemmaDetector,
}


class ImageCensorPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.runtime_config = config.get("runtime", {})
        self.decision_config = config.get("decision", {})
        self.detectors = self._build_detectors(config.get("detectors", {}))

    def _build_detectors(self, detectors_config: Dict) -> List:
        detectors = []
        for name, detector_config in detectors_config.items():
            detector_class = DETECTOR_CLASSES.get(name)
            if detector_class is None:
                continue
            detectors.append(detector_class(detector_config, self.runtime_config))
        return detectors

    def check(self, request: GuardRequest) -> GuardResult:
        findings: List[Finding] = []
        latency_ms = {}
        for detector in self.detectors:
            started_at = time.perf_counter()
            findings.extend(detector.run(request))
            latency_ms[detector.name] = round((time.perf_counter() - started_at) * 1000, 3)

        result = aggregate(findings, self.decision_config)
        result.audit["latency_ms"] = latency_ms
        result.audit["request_id"] = request.request_id
        result.audit["scenario"] = request.scenario
        maybe_enqueue_review(result, request, self.decision_config.get("review_queue_path"))
        return result

    def describe(self) -> Dict:
        return {
            "detectors": [
                {
                    "name": detector.name,
                    "enabled": detector.enabled,
                    "stages": sorted(detector.stages),
                    "model_id": detector.config.get("model_id"),
                }
                for detector in self.detectors
            ],
            "decision": self.decision_config,
        }
