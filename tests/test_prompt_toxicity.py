import unittest

from img_censor.detectors.prompt_toxicity import aggregate_toxicity, category_from_labels


class PromptToxicityTest(unittest.TestCase):
    def test_aggregate_uses_dangerous_label(self):
        score = aggregate_toxicity(
            {
                "non-toxic": 0.95,
                "insult": 0.01,
                "obscenity": 0.01,
                "threat": 0.02,
                "dangerous": 0.9,
            }
        )

        self.assertLess(score, 0.1)

    def test_aggregate_blocks_when_non_toxic_is_low(self):
        score = aggregate_toxicity(
            {
                "non-toxic": 0.05,
                "insult": 0.01,
                "obscenity": 0.01,
                "threat": 0.02,
                "dangerous": 0.9,
            }
        )

        self.assertGreaterEqual(score, 0.8)

    def test_category_from_dangerous_label(self):
        category = category_from_labels({"dangerous": 0.9, "threat": 0.1})
        self.assertEqual(category, "dangerous")

    def test_category_from_general_toxicity(self):
        category = category_from_labels({"insult": 0.9, "dangerous": 0.8})
        self.assertEqual(category, "toxic_inappropriate")


if __name__ == "__main__":
    unittest.main()
