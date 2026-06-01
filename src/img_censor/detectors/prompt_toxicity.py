from typing import List

from img_censor.detectors.base import Detector
from img_censor.schemas import Finding, GuardRequest, Verdict


class PromptToxicityDetector(Detector):
    name = "prompt_toxicity"

    def __init__(self, config: dict, runtime_config: dict):
        super().__init__(config, runtime_config)
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_id = self.config["model_id"]
            cache_dir = self.runtime_config.get("cache_dir")
            local_files_only = bool(self.config.get("local_files_only", True))
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                ).eval()
            except OSError as error:
                raise RuntimeError(
                    f"Prompt model {model_id} is not available locally. "
                    "Run scripts/download_prompt_model.sh first or set local_files_only: false in config."
                ) from error
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
            self._model.to(self._device)
        return self._model, self._tokenizer, self._device

    def run(self, request: GuardRequest) -> List[Finding]:
        if not self.enabled or not request.prompt:
            return []

        import torch

        model, tokenizer, device = self._load()
        inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(**inputs).logits)[0].detach().float().cpu()

        label_scores = labels_from_probabilities(model.config.id2label, probabilities.tolist())
        toxicity_score = aggregate_toxicity(label_scores)
        verdict = self.threshold_verdict(toxicity_score)
        if verdict == Verdict.ALLOW:
            return []

        category = category_from_labels(label_scores)
        return [
            Finding(
                detector=self.name,
                stage="prompt",
                category=category,
                score=toxicity_score,
                verdict=verdict,
                rationale="Prompt toxicity classifier detected toxic or inappropriate request text.",
                raw={
                    "model_id": self.config["model_id"],
                    "label_scores": {key: round(value, 6) for key, value in label_scores.items()},
                },
            )
        ]


def labels_from_probabilities(id2label: dict, probabilities: List[float]) -> dict:
    return {
        str(id2label.get(index, index)).lower(): float(probability)
        for index, probability in enumerate(probabilities)
    }


def aggregate_toxicity(label_scores: dict) -> float:
    non_toxic = label_scores.get("non-toxic", label_scores.get("non_toxic", 1.0))
    toxic_labels = [
        score
        for label, score in label_scores.items()
        if label not in {"non-toxic", "non_toxic", "neutral"}
    ]
    max_toxic = max(toxic_labels) if toxic_labels else 0.0
    return max_toxic * (1 - non_toxic)


def category_from_labels(label_scores: dict) -> str:
    harmful_scores = {
        label: score
        for label, score in label_scores.items()
        if label not in {"non-toxic", "non_toxic", "neutral"}
    }
    if not harmful_scores:
        return "toxic_inappropriate"

    dominant_label = max(harmful_scores, key=harmful_scores.get)
    if dominant_label in {"dangerous", "threat"}:
        return "dangerous"
    return "toxic_inappropriate"
