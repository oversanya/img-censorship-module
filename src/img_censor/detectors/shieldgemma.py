from typing import List

from img_censor.detectors.base import Detector
from img_censor.io import load_image, move_batch_to_device, select_device_and_dtype
from img_censor.schemas import Finding, GuardRequest, Verdict


class ShieldGemmaDetector(Detector):
    name = "shieldgemma"

    def __init__(self, config: dict, runtime_config: dict):
        super().__init__(config, runtime_config)
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None

    def _load(self):
        if self._model is None:
            from transformers import AutoProcessor, ShieldGemma2ForImageClassification

            model_id = self.config["model_id"]
            cache_dir = self.runtime_config.get("cache_dir")
            self._device, self._dtype = select_device_and_dtype(self.runtime_config)
            self._processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self._model = ShieldGemma2ForImageClassification.from_pretrained(
                model_id,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
            ).eval()
            self._model.to(self._device)
        return self._model, self._processor, self._device, self._dtype

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
            findings.extend(self._classify(stage, image_path))
        return findings

    def _classify(self, stage: str, image_path: str) -> List[Finding]:
        import torch

        model, processor, device, dtype = self._load()
        image = load_image(image_path)
        policies = self.config.get("policies", ["dangerous", "sexual", "violence"])
        inputs = processor(images=[image], policies=policies, return_tensors="pt")
        inputs = move_batch_to_device(inputs, device, dtype)

        with torch.inference_mode():
            outputs = model(**inputs)

        probabilities = outputs.probabilities.detach().float().cpu()
        findings = []
        for policy, pair in zip(policies, probabilities):
            yes_probability = float(pair[0])
            verdict = self.threshold_verdict(yes_probability)
            if verdict != Verdict.ALLOW:
                findings.append(
                    Finding(
                        detector=self.name,
                        stage=stage,
                        category=policy,
                        score=yes_probability,
                        verdict=verdict,
                        rationale="ShieldGemma classified the image as violating this policy.",
                        raw={
                            "policy": policy,
                            "yes_probability": yes_probability,
                            "no_probability": float(pair[1]),
                            "model_id": self.config["model_id"],
                        },
                    )
                )
        return findings
