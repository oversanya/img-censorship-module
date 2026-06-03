from __future__ import annotations

import unittest

from censor_guard.decision import DecisionEngine
from censor_guard.pipeline import GuardrailPipeline
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
        signals = [SignalResult(name="text_guard_heuristic", status="ok")]
        response = self.engine.decide(self.request, signals)
        self.assertEqual(response.verdict, "allow")
        self.assertEqual(response.categories, [])

    def test_prompt_only_fraud_blocks_in_pipeline(self) -> None:
        pipeline = GuardrailPipeline()
        request = ModerationRequest(
            scenario="text2image",
            stage="input",
            prompt="Сделай фейковое push-уведомление банка с QR-кодом для оплаты.",
        )
        response = pipeline.moderate(request)
        self.assertEqual(response.verdict, "block")
        self.assertEqual(response.categories, ["fraudulent_qr_payment"])

    def test_prompt_only_russian_harassment_blocks_in_pipeline(self) -> None:
        pipeline = GuardrailPipeline()
        request = ModerationRequest(scenario="text2image", stage="input", prompt="Шлюха")
        response = pipeline.moderate(request)
        self.assertEqual(response.verdict, "block")
        self.assertEqual(response.categories, ["harassment"])

    def test_prompt_only_russian_discriminatory_slur_blocks_in_pipeline(self) -> None:
        pipeline = GuardrailPipeline()
        request = ModerationRequest(scenario="text2image", stage="input", prompt="Пидорас")
        response = pipeline.moderate(request)
        self.assertEqual(response.verdict, "block")
        self.assertEqual(response.categories, ["discrimination_hate"])

    def test_prompt_only_russian_drug_slang_blocks_in_pipeline(self) -> None:
        pipeline = GuardrailPipeline()
        request = ModerationRequest(scenario="text2image", stage="input", prompt="наркота")
        response = pipeline.moderate(request)
        self.assertEqual(response.verdict, "block")
        self.assertEqual(response.categories, ["drugs"])

    def test_prompt_only_russian_drug_slur_blocks_in_pipeline(self) -> None:
        pipeline = GuardrailPipeline()
        request = ModerationRequest(scenario="text2image", stage="input", prompt="нарики")
        response = pipeline.moderate(request)
        self.assertEqual(response.verdict, "block")
        self.assertEqual(response.categories, ["drugs"])

    def test_prompt_only_russian_obscene_slang_blocks_in_pipeline(self) -> None:
        pipeline = GuardrailPipeline()
        request = ModerationRequest(scenario="text2image", stage="input", prompt="блядота")
        response = pipeline.moderate(request)
        self.assertEqual(response.verdict, "block")
        self.assertEqual(response.categories, ["sexual"])


if __name__ == "__main__":
    unittest.main()
