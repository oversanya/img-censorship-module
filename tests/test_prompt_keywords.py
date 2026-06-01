import unittest

from img_censor.detectors.heuristics import PromptKeywordGuard
from img_censor.schemas import GuardRequest, Verdict


class PromptKeywordGuardTest(unittest.TestCase):
    def test_blocks_swastika_prompt(self):
        detector = PromptKeywordGuard(
            {"enabled": True, "stages": ["prompt"], "block_threshold": 1.0},
            {},
        )

        findings = detector.run(GuardRequest(prompt="Нарисуй свастику"))

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "hate_extremism")
        self.assertEqual(findings[0].verdict, Verdict.BLOCK)

    def test_blocks_drugs_prompt_case_insensitive(self):
        detector = PromptKeywordGuard(
            {"enabled": True, "stages": ["prompt"], "block_threshold": 1.0},
            {},
        )

        findings = detector.run(GuardRequest(prompt="НАрисуй наркотики"))

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].category, "dangerous")
        self.assertEqual(findings[0].verdict, Verdict.BLOCK)


if __name__ == "__main__":
    unittest.main()
