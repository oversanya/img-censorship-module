import unittest

from img_censor.aggregator import aggregate
from img_censor.schemas import Finding, Verdict


class AggregatorTest(unittest.TestCase):
    def test_allow_without_findings(self):
        result = aggregate([], {"fail_closed_for": []})
        self.assertEqual(result.verdict, Verdict.ALLOW)
        self.assertIn("policy_version", result.audit)

    def test_blocks_fail_closed_category_even_if_review(self):
        finding = Finding(
            detector="test",
            stage="output_image",
            category="personal_financial_data",
            score=0.4,
            verdict=Verdict.REVIEW,
            rationale="test",
        )
        result = aggregate([finding], {"fail_closed_for": ["personal_financial_data"]})
        self.assertEqual(result.verdict, Verdict.BLOCK)
        self.assertEqual(result.audit["category_evidence"]["personal_financial_data"]["severity"], "critical")

    def test_uses_max_detector_verdict(self):
        findings = [
            Finding("a", "prompt", "sexual", 0.4, Verdict.REVIEW, "test"),
            Finding("b", "output_image", "violence_gore", 0.8, Verdict.BLOCK, "test"),
        ]
        result = aggregate(findings, {"fail_closed_for": []})
        self.assertEqual(result.verdict, Verdict.BLOCK)
        self.assertEqual(result.categories, ["sexual", "violence_gore"])


if __name__ == "__main__":
    unittest.main()
