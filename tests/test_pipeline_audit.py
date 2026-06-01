import unittest

from img_censor.pipeline import ImageCensorPipeline
from img_censor.schemas import GuardRequest


class PipelineAuditTest(unittest.TestCase):
    def test_audit_includes_latency_scenario_and_request_id(self):
        pipeline = ImageCensorPipeline(
            {
                "runtime": {},
                "decision": {"fail_closed_for": []},
                "detectors": {
                    "prompt_keywords": {
                        "enabled": True,
                        "stages": ["prompt"],
                        "block_threshold": 1.0,
                    }
                },
            }
        )

        result = pipeline.check(
            GuardRequest(
                prompt="Сгенерируй фото машины",
                request_id="req-1",
                scenario="text2image",
            )
        )

        self.assertEqual(result.audit["request_id"], "req-1")
        self.assertEqual(result.audit["scenario"], "text2image")
        self.assertIn("prompt_keywords", result.audit["latency_ms"])


if __name__ == "__main__":
    unittest.main()
