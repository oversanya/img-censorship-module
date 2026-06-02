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

    def test_allows_without_signals(self) -> None:
        signals = [SignalResult(name="text_guard_stub", status="ok")]
        response = self.engine.decide(self.request, signals)
        self.assertEqual(response.verdict, "allow")
        self.assertEqual(response.categories, [])


if __name__ == "__main__":
    unittest.main()
