from __future__ import annotations

import unittest

from censor_guard.calibration import calibrate_against_safe
from censor_guard.taxonomy import SAFE_VISUAL_LABEL_SET, VISUAL_LABEL_TO_CODE


class CalibrationTests(unittest.TestCase):
    def _calibrate(self, raw: dict[str, float], floor: float = 0.5):
        return calibrate_against_safe(
            raw_scores=raw,
            label_to_code=VISUAL_LABEL_TO_CODE,
            safe_labels=SAFE_VISUAL_LABEL_SET,
            floor=floor,
        )

    def test_benign_image_collapses_to_near_zero(self) -> None:
        # Котик: "sexual" едва обгоняет safe → калиброванная оценка должна быть ~0,
        # хотя сырой softmax 0.27 выглядел «опасно».
        raw = {"sexual or explicit content": 0.275, "safe everyday image": 0.259}
        result = self._calibrate(raw)
        self.assertLess(result.scores["sexual"], 0.1)

    def test_unsafe_image_scores_high(self) -> None:
        raw = {"sexual or explicit content": 0.6, "safe everyday image": 0.05}
        result = self._calibrate(raw)
        self.assertGreater(result.scores["sexual"], 0.8)

    def test_uses_strongest_safe_anchor(self) -> None:
        # Даже если первый якорь слабый, берём максимум среди safe-формулировок.
        raw = {
            "sexual or explicit content": 0.4,
            "safe everyday image": 0.05,
            "neutral ordinary photo": 0.5,
        }
        result = self._calibrate(raw)
        self.assertEqual(result.safe_score, 0.5)
        self.assertLess(result.scores["sexual"], 0.1)


if __name__ == "__main__":
    unittest.main()
