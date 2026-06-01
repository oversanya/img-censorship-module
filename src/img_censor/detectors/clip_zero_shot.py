from typing import List

from img_censor.detectors.base import Detector
from img_censor.io import load_image, move_batch_to_device, select_device_and_dtype
from img_censor.schemas import Finding, GuardRequest, Verdict


class ClipZeroShotDetector(Detector):
    name = "clip_zero_shot"

    def __init__(self, config: dict, runtime_config: dict):
        super().__init__(config, runtime_config)
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor

            model_id = self.config["model_id"]
            cache_dir = self.runtime_config.get("cache_dir")
            self._device, self._dtype = select_device_and_dtype(self.runtime_config)
            self._processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self._model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir).eval()
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

        labels = self.config.get("labels", {})
        if not labels:
            return []

        model, processor, device, dtype = self._load()
        image = load_image(image_path)
        label_names = list(labels.keys())
        prompts = [labels[name] for name in label_names]
        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
        inputs = move_batch_to_device(inputs, device, dtype)

        with torch.inference_mode():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].detach().float().cpu()

        findings = []
        for category, score in zip(label_names, probs):
            score_value = float(score)
            verdict = self.threshold_verdict(score_value)
            if verdict != Verdict.ALLOW:
                findings.append(
                    Finding(
                        detector=self.name,
                        stage=stage,
                        category=category,
                        score=score_value,
                        verdict=verdict,
                        rationale="CLIP zero-shot visual label matched a risk prompt.",
                        raw={"label_prompt": labels[category], "model_id": self.config["model_id"]},
                    )
                )
        return findings
