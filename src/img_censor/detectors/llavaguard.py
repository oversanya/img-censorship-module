import json
import re
from typing import List

from img_censor.detectors.base import Detector
from img_censor.io import load_image, move_batch_to_device, select_device_and_dtype
from img_censor.policy import LLAVAGUARD_CATEGORY_MAP, LLAVAGUARD_POLICY
from img_censor.schemas import Finding, GuardRequest, Verdict


class LlavaGuardDetector(Detector):
    name = "llavaguard"

    def __init__(self, config: dict, runtime_config: dict):
        super().__init__(config, runtime_config)
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None

    def _load(self):
        if self._model is None:
            from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

            model_id = self.config["model_id"]
            cache_dir = self.runtime_config.get("cache_dir")
            self._device, self._dtype = select_device_and_dtype(self.runtime_config)
            self._processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self._model = LlavaOnevisionForConditionalGeneration.from_pretrained(
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
            findings.append(self._classify(stage, image_path))
        return [finding for finding in findings if finding.verdict != Verdict.ALLOW]

    def _classify(self, stage: str, image_path: str) -> Finding:
        model, processor, device, dtype = self._load()
        image = load_image(image_path)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": LLAVAGUARD_POLICY},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=text_prompt, images=image, return_tensors="pt")
        inputs = move_batch_to_device(inputs, device, dtype)

        generation_config = {
            "max_new_tokens": int(self.config.get("max_new_tokens", 220)),
            "do_sample": False,
            "use_cache": True,
        }

        output = model.generate(**inputs, **generation_config)
        decoded = processor.decode(output[0], skip_special_tokens=True)
        parsed = parse_llavaguard_json(decoded)

        rating = str(parsed.get("rating", "Safe"))
        category_text = str(parsed.get("category", "NA: None applying"))
        rationale = str(parsed.get("rationale", "LlavaGuard returned no rationale."))
        category = normalize_llavaguard_category(category_text)

        unsafe = rating.lower() == "unsafe"
        score = 1.0 if unsafe else 0.0
        verdict = self.threshold_verdict(score)
        return Finding(
            detector=self.name,
            stage=stage,
            category=category,
            score=score,
            verdict=verdict,
            rationale=rationale,
            raw={
                "rating": rating,
                "category_text": category_text,
                "model_id": self.config["model_id"],
                "decoded": decoded,
            },
        )


def parse_llavaguard_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    rating_match = re.search(r'"?rating"?\s*:\s*"?(Unsafe|Safe)"?', text, flags=re.IGNORECASE)
    category_match = re.search(r'"?category"?\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    rationale_match = re.search(r'"?rationale"?\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    return {
        "rating": rating_match.group(1) if rating_match else "Safe",
        "category": category_match.group(1) if category_match else "NA: None applying",
        "rationale": rationale_match.group(1) if rationale_match else text[-500:],
    }


def normalize_llavaguard_category(category_text: str) -> str:
    for prefix, category in LLAVAGUARD_CATEGORY_MAP.items():
        if category_text.strip().startswith(prefix):
            return category
    return "unknown"
