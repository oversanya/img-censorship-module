import unittest

from img_censor.hackathon_service import HackathonCensorService
from img_censor.pipeline import ImageCensorPipeline
from img_censor.schemas import Verdict


def prompt_only_pipeline():
    return ImageCensorPipeline(
        {
            "runtime": {},
            "decision": {"fail_closed_for": ["extremism_terrorism_symbols"]},
            "detectors": {
                "prompt_keywords": {
                    "enabled": True,
                    "stages": ["prompt"],
                    "block_threshold": 1.0,
                }
            },
        }
    )


class HackathonServiceTest(unittest.TestCase):
    def test_prompt_stage_blocks_before_generation(self):
        service = HackathonCensorService(prompt_only_pipeline())

        result = service.full_flow(prompt="Нарисуй свастику")

        self.assertEqual(result["verdict"], Verdict.BLOCK.value)
        self.assertEqual(result["blocked_stage"], "input_gate")
        self.assertTrue(result["generation"]["skipped"])

    def test_full_flow_allows_safe_prompt_and_creates_mock_output(self):
        service = HackathonCensorService(prompt_only_pipeline())

        result = service.full_flow(prompt="Сгенерируй фото машины")

        self.assertEqual(result["verdict"], Verdict.ALLOW.value)
        self.assertEqual(result["generation"]["mode"], "mock_generator")
        self.assertTrue(result["generation"]["image_path"].endswith(".png"))
        self.assertIsNotNone(result["output_gate"])


if __name__ == "__main__":
    unittest.main()
