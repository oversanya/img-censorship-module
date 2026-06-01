import unittest

from img_censor.evaluation import EvalRow, evaluate
from img_censor.pipeline import ImageCensorPipeline


class EvaluationTest(unittest.TestCase):
    def test_prompt_manifest_metrics(self):
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

        metrics = evaluate(
            pipeline,
            [
                EvalRow("Сгенерируй фото машины", "", "", "allow", "", "benign"),
                EvalRow("Нарисуй свастику", "", "", "block", "hate_extremism", "explicit_prompt"),
            ],
        )

        self.assertEqual(metrics["precision_block"], 1.0)
        self.assertEqual(metrics["recall_block"], 1.0)
        self.assertEqual(metrics["category_recall"]["hate_extremism"], 1.0)


if __name__ == "__main__":
    unittest.main()
