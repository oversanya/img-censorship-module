from __future__ import annotations

import json
import unittest
from pathlib import Path

from censor_guard.pipeline import GuardrailPipeline
from censor_guard.schemas import ModerationRequest


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "forbidden_prompts_ru.jsonl"


class ForbiddenPromptsDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = GuardrailPipeline()
        cls.items = [
            json.loads(line)
            for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_dataset_is_not_empty(self) -> None:
        self.assertGreaterEqual(len(self.items), 60)

    def test_forbidden_prompts_are_not_allowed(self) -> None:
        misses = []
        for item in self.items:
            response = self.pipeline.moderate(
                ModerationRequest(
                    scenario="text2image",
                    stage="input",
                    prompt=item["prompt"],
                )
            )
            if response.verdict == "allow" or item["category"] not in response.categories:
                misses.append(
                    {
                        "id": item["id"],
                        "expected": item["category"],
                        "verdict": response.verdict,
                        "actual": response.categories,
                    }
                )

        self.assertEqual(misses, [])


if __name__ == "__main__":
    unittest.main()
