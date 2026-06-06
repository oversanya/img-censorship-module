from __future__ import annotations

import unittest

from censor_guard.decision import DecisionEngine
from censor_guard.schemas import ModerationRequest, SignalResult


class DecisionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = DecisionEngine(block_threshold=0.85, review_threshold=0.55)
        self.request = ModerationRequest(scenario="output", stage="output")

    def test_blocks_hard_category(self) -> None:
        signals = [
            SignalResult(
                name="visual_classifier",
                status="ok",
                categories={"violence_gore": 0.91},
            )
        ]
        response = self.engine.decide(self.request, signals)
        self.assertEqual(response.verdict, "block")
        self.assertEqual(response.categories, ["violence_gore"])
        self.assertEqual(response.audit.reason_code, "hard_category_above_block_threshold")
        self.assertEqual(response.audit.thresholds, {"block": 0.85, "review": 0.55})

    def test_reviews_medium_category(self) -> None:
        signals = [
            SignalResult(
                name="visual_classifier",
                status="ok",
                categories={"political_persuasion": 0.61},
            )
        ]
        response = self.engine.decide(self.request, signals)
        self.assertEqual(response.verdict, "review")
        self.assertEqual(response.categories, ["political_persuasion"])
        self.assertEqual(response.audit.reason_code, "category_above_review_threshold")
        self.assertEqual(response.audit.matched_categories[0]["category"], "political_persuasion")

    def test_allows_without_signals(self) -> None:
        signals = [SignalResult(name="text_guard_stub", status="ok")]
        response = self.engine.decide(self.request, signals)
        self.assertEqual(response.verdict, "allow")
        self.assertEqual(response.categories, [])
        self.assertEqual(response.audit.reason_code, "no_policy_signal_above_review_threshold")
        self.assertEqual(response.audit.decision_path[-1]["verdict"], "allow")


if __name__ == "__main__":
    unittest.main()
