from typing import Dict, List

from img_censor.aggregator import aggregate
from img_censor.detectors import (
    ClipZeroShotDetector,
    LlavaGuardDetector,
    NsfwVitDetector,
    PromptKeywordGuard,
    PromptToxicityDetector,
    PromptZeroShotDetector,
    ShieldGemmaDetector,
)
from img_censor.schemas import Finding, GuardRequest, GuardResult


DETECTOR_CLASSES = {
    "prompt_keywords": PromptKeywordGuard,
    "prompt_toxicity": PromptToxicityDetector,
    "prompt_zero_shot": PromptZeroShotDetector,
    "nsfw_vit": NsfwVitDetector,
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
        for detector in self.detectors:
            findings.extend(detector.run(request))
        return aggregate(findings, self.decision_config)

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
