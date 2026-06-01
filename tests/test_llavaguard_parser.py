import unittest

from img_censor.detectors.llavaguard import normalize_llavaguard_category, parse_llavaguard_json


class LlavaGuardParserTest(unittest.TestCase):
    def test_parse_json_from_verbose_output(self):
        parsed = parse_llavaguard_json(
            'assistant\n{"rating": "Unsafe", "category": "O2: Violence, Harm, or Cruelty", "rationale": "blood"}'
        )
        self.assertEqual(parsed["rating"], "Unsafe")
        self.assertEqual(parsed["category"], "O2: Violence, Harm, or Cruelty")

    def test_normalize_category(self):
        category = normalize_llavaguard_category("O8: Personal or Financial Data")
        self.assertEqual(category, "personal_financial_data")


if __name__ == "__main__":
    unittest.main()

