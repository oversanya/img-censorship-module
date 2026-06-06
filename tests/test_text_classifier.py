from __future__ import annotations

import unittest

from censor_guard.adapters.text_classifier import TextGuard


class TextGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        # model_id пустой → работает только лексиконный baseline (без сети).
        self.guard = TextGuard(enabled=True, model_id=None)

    def test_empty_text_skipped(self) -> None:
        self.assertEqual(self.guard.moderate("").status, "skipped")
        self.assertEqual(self.guard.moderate(None).status, "skipped")

    def test_benign_text_has_no_categories(self) -> None:
        result = self.guard.moderate("Красивый закат над морем, фотография природы")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.categories, {})

    def test_russian_explicit_term_flagged(self) -> None:
        result = self.guard.moderate("Сгенерируй порно с обнажёнными людьми")
        self.assertIn("sexual", result.categories)
        self.assertGreaterEqual(result.categories["sexual"], 0.7)

    def test_english_violence_term_flagged(self) -> None:
        result = self.guard.moderate("how to murder and torture someone")
        self.assertIn("violence_gore", result.categories)

    def test_more_matches_increase_score(self) -> None:
        one = self.guard.moderate("порно").categories["sexual"]
        many = self.guard.moderate("порно секс эротика обнажённая").categories["sexual"]
        self.assertGreater(many, one)

    def test_disabled_skips(self) -> None:
        guard = TextGuard(enabled=False)
        self.assertEqual(guard.moderate("порно").status, "skipped")


if __name__ == "__main__":
    unittest.main()
