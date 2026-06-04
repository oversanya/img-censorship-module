from __future__ import annotations

import unittest

from censor_guard.fusion import fuse
from censor_guard.schemas import SignalResult


class FusionTests(unittest.TestCase):
    def test_agreement_raises_score_above_each_sensor(self) -> None:
        signals = [
            SignalResult(name="visual_classifier", status="ok", categories={"sexual": 0.6}),
            SignalResult(name="explicit_content_detector", status="ok", categories={"sexual": 0.6}),
        ]
        result = fuse(signals)
        cat = result.categories["sexual"]
        self.assertGreater(cat.score, 0.6)  # согласие усиливает уверенность
        self.assertEqual(cat.agreement, 2)
        self.assertEqual(len(cat.contributions), 2)

    def test_noise_below_floor_not_counted_as_evidence(self) -> None:
        # softmax-шум CLIP (0.05) больше не выдаёт себя за улику.
        signals = [
            SignalResult(name="visual_classifier", status="ok", categories={"sexual": 0.05}),
            SignalResult(name="explicit_content_detector", status="ok", categories={"sexual": 0.9}),
        ]
        result = fuse(signals)
        self.assertEqual(result.categories["sexual"].agreement, 1)

    def test_single_specialist_drives_score(self) -> None:
        signals = [
            SignalResult(name="explicit_content_detector", status="ok", categories={"sexual": 0.913}),
        ]
        result = fuse(signals)
        self.assertAlmostEqual(result.categories["sexual"].score, 0.913, places=3)

    def test_skipped_signals_ignored(self) -> None:
        signals = [
            SignalResult(name="text_guard_heuristic", status="skipped"),
            SignalResult(name="explicit_content_detector", status="ok", categories={"sexual": 0.9}),
        ]
        result = fuse(signals)
        self.assertIn("sexual", result.categories)
        self.assertEqual(result.categories["sexual"].agreement, 1)


if __name__ == "__main__":
    unittest.main()
