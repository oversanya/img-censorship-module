from typing import Dict, Optional

from img_censor.generator import create_mock_generated_image
from img_censor.pipeline import ImageCensorPipeline
from img_censor.schemas import GuardRequest, GuardResult, Verdict


class HackathonCensorService:
    def __init__(self, pipeline: ImageCensorPipeline):
        self.pipeline = pipeline

    def check_prompt(self, prompt: str, request_id: Optional[str] = None) -> GuardResult:
        return self.pipeline.check(GuardRequest(prompt=prompt, request_id=request_id))

    def check_input_image(self, image_path: str, request_id: Optional[str] = None) -> GuardResult:
        return self.pipeline.check(GuardRequest(input_image=image_path, request_id=request_id))

    def check_output_image(self, image_path: str, request_id: Optional[str] = None) -> GuardResult:
        return self.pipeline.check(GuardRequest(output_image=image_path, request_id=request_id))

    def check_input_gate(
        self,
        prompt: Optional[str],
        input_image: Optional[str],
        request_id: Optional[str] = None,
    ) -> GuardResult:
        return self.pipeline.check(GuardRequest(prompt=prompt, input_image=input_image, request_id=request_id))

    def full_flow(
        self,
        prompt: Optional[str],
        input_image: Optional[str] = None,
        generated_image: Optional[str] = None,
        request_id: Optional[str] = None,
        use_mock_generator: bool = True,
    ) -> Dict:
        input_gate = self.check_input_gate(prompt=prompt, input_image=input_image, request_id=request_id)
        if input_gate.verdict != Verdict.ALLOW:
            return {
                "verdict": input_gate.verdict.value,
                "blocked_stage": "input_gate",
                "reason": "Request was stopped before generation.",
                "input_gate": input_gate.to_dict(),
                "generation": {"skipped": True},
                "output_gate": None,
            }

        image_for_output_check = generated_image
        generation = {
            "skipped": False,
            "mode": "provided_generated_image" if generated_image else "mock_generator",
            "image_path": generated_image,
        }

        if image_for_output_check is None and use_mock_generator:
            image_for_output_check = create_mock_generated_image(prompt)
            generation["image_path"] = image_for_output_check

        if image_for_output_check is None:
            return {
                "verdict": Verdict.REVIEW.value,
                "blocked_stage": "generation",
                "reason": "No generated image was supplied and mock generation is disabled.",
                "input_gate": input_gate.to_dict(),
                "generation": generation,
                "output_gate": None,
            }

        output_gate = self.check_output_image(image_for_output_check, request_id=request_id)
        return {
            "verdict": output_gate.verdict.value,
            "blocked_stage": None if output_gate.verdict == Verdict.ALLOW else "output_gate",
            "reason": "Generated image passed output censor." if output_gate.verdict == Verdict.ALLOW else output_gate.rationale,
            "input_gate": input_gate.to_dict(),
            "generation": generation,
            "output_gate": output_gate.to_dict(),
        }

