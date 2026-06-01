from typing import List, Optional

from img_censor.detectors.base import Detector
from img_censor.schemas import Finding, GuardRequest, Verdict


class PromptZeroShotDetector(Detector):
    name = "prompt_zero_shot"

    def __init__(self, config: dict, runtime_config: dict):
        super().__init__(config, runtime_config)
        self._classifier = None

    def _load(self):
        if self._classifier is None:
            from transformers import pipeline

            model_id = self.config["model_id"]
            cache_dir = self.runtime_config.get("cache_dir")
            kwargs = {"model_kwargs": {"cache_dir": cache_dir}} if cache_dir else {}
            self._classifier = pipeline("zero-shot-classification", model=model_id, device=-1, **kwargs)
        return self._classifier

    def run(self, request: GuardRequest) -> List[Finding]:
        if not self.enabled or not request.prompt:
            return []

        labels = self.config.get("labels", {})
        if not labels:
            return []

        classifier = self._load()
        label_names = list(labels.keys())
        label_texts = [labels[name] for name in label_names]
        result = classifier(request.prompt, candidate_labels=label_texts, multi_label=True)

        findings = []
        for label_text, score in zip(result["labels"], result["scores"]):
            category = self._category_for_label_text(label_text, labels)
            verdict = self.threshold_verdict(float(score))
            if verdict != Verdict.ALLOW:
                findings.append(
                    Finding(
                        detector=self.name,
                        stage="prompt",
                        category=category,
                        score=float(score),
                        verdict=verdict,
                        rationale="Prompt is semantically close to a prohibited request class.",
                        raw={"label": label_text, "model_id": self.config["model_id"]},
                    )
                )
        return findings

    @staticmethod
    def _category_for_label_text(label_text: str, labels: dict) -> str:
        for category, configured_text in labels.items():
            if configured_text == label_text:
                return category
        return "unknown"
