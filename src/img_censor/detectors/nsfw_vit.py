from typing import List

from img_censor.detectors.base import Detector
from img_censor.io import load_image
from img_censor.schemas import Finding, GuardRequest, Verdict


class NsfwVitDetector(Detector):
    name = "nsfw_vit"

    def __init__(self, config: dict, runtime_config: dict):
        super().__init__(config, runtime_config)
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline

            cache_dir = self.runtime_config.get("cache_dir")
            kwargs = {"model_kwargs": {"cache_dir": cache_dir}} if cache_dir else {}
            self._pipeline = pipeline("image-classification", model=self.config["model_id"], **kwargs)
        return self._pipeline

    def run(self, request: GuardRequest) -> List[Finding]:
        if not self.enabled:
            return []

        findings = []
        for stage, image_path in (
            ("input_image", request.input_image),
            ("output_image", request.output_image),
        ):
            if not image_path or not self.should_run_on(stage):
                continue
            image = load_image(image_path)
            predictions = self._load()(image)
            findings.extend(self._find_unsafe(stage, predictions))
        return findings

    def _find_unsafe(self, stage: str, predictions: list) -> List[Finding]:
        unsafe_labels = {label.lower() for label in self.config.get("unsafe_labels", [])}
        findings = []
        for prediction in predictions:
            label = str(prediction.get("label", "")).lower().replace("-", "_")
            normalized = label.replace("_", " ")
            if label in unsafe_labels or normalized in unsafe_labels:
                score = float(prediction.get("score", 0.0))
                verdict = self.threshold_verdict(score)
                if verdict != Verdict.ALLOW:
                    findings.append(
                        Finding(
                            detector=self.name,
                            stage=stage,
                            category="sexual",
                            score=score,
                            verdict=verdict,
                            rationale="Fast NSFW classifier detected unsafe visual content.",
                            raw={"prediction": prediction, "model_id": self.config["model_id"]},
                        )
                    )
        return findings
