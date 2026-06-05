from __future__ import annotations

import unittest

from censor_guard.decision import DecisionEngine
from censor_guard.fusion import FUSION_SIGNAL_NAME
from censor_guard.schemas import ModerationRequest, SignalResult


def fusion_signal(scores, agreement=None, sources=None) -> SignalResult:
    """Собирает сигнал policy_fusion как его строит PolicyJudge."""
    agreement = agreement or {code: 1 for code in scores}
    sources = sources or {code: ["visual_classifier"] for code in scores}
    contributions = {
        code: [{"sensor": s, "score": scores[code], "weight": 1.0} for s in sources[code]]
        for code in scores
    }
    return SignalResult(
        name=FUSION_SIGNAL_NAME,
        status="ok",
        categories=dict(scores),
        raw={"contributions": contributions, "agreement": agreement},
    )


class DecisionWithFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = DecisionEngine(block_threshold=0.85, review_threshold=0.55)
        self.request = ModerationRequest(scenario="output", stage="output")

    def test_hard_category_blocks_from_fusion(self) -> None:
        signals = [fusion_signal({"sexual": 0.9})]
        response = self.engine.decide(self.request, signals)
        self.assertEqual(response.verdict, "block")
        self.assertEqual(response.categories, ["sexual"])
        self.assertEqual(response.confidence, 0.9)  # честная оценка, без раздувания

    def test_soft_category_needs_two_sensors_to_block(self) -> None:
        # Одиночный сенсор по soft-категории → только review, даже выше block-порога.
        single = [fusion_signal({"political_persuasion": 0.9}, agreement={"political_persuasion": 1})]
        self.assertEqual(self.engine.decide(self.request, single).verdict, "review")

        # Согласие двух сенсоров → блок.
        double = [
            fusion_signal(
                {"political_persuasion": 0.9},
                agreement={"political_persuasion": 2},
                sources={"political_persuasion": ["visual_classifier", "text_guard"]},
            )
        ]
        self.assertEqual(self.engine.decide(self.request, double).verdict, "block")

    def test_soft_category_blocks_when_shieldgemma_confirms(self) -> None:
        signals = [
            fusion_signal(
                {"harassment": 0.9},
                agreement={"harassment": 1},
                sources={"harassment": ["policy_judge_shieldgemma"]},
            )
        ]
        self.assertEqual(self.engine.decide(self.request, signals).verdict, "block")

    def test_evidence_reports_contributing_sensors(self) -> None:
        signals = [
            fusion_signal(
                {"sexual": 0.9},
                sources={"sexual": ["visual_classifier", "explicit_content_detector"]},
            )
        ]
        response = self.engine.decide(self.request, signals)
        self.assertEqual(
            response.evidence["sexual"],
            ["explicit_content_detector", "visual_classifier"],
        )


if __name__ == "__main__":
    unittest.main()
