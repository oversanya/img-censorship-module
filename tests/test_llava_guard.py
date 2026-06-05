from __future__ import annotations

import unittest

from censor_guard.adapters.llava_guard import interpret_llavaguard_output


class LlavaGuardInterpretTests(unittest.TestCase):
    """Чистая интерпретация текстового ответа модели (без её загрузки)."""

    def test_unsafe_maps_to_taxonomy_code(self) -> None:
        text = '{\n  "rating": "Unsafe",\n  "category": "O3: Sexual Content",\n  "rationale": "..."}'
        cats, info = interpret_llavaguard_output(text, unsafe_score=0.9)
        self.assertEqual(cats, {"sexual": 0.9})
        self.assertEqual(info["rating"], "unsafe")
        self.assertEqual(info["mapped_code"], "sexual")

    def test_safe_yields_no_categories(self) -> None:
        text = '{"rating": "Safe", "category": "O3: Sexual Content", "rationale": "ok"}'
        cats, info = interpret_llavaguard_output(text, unsafe_score=0.9)
        self.assertEqual(cats, {})
        self.assertEqual(info["rating"], "safe")

    def test_unsafe_na_category_no_score(self) -> None:
        text = '{"rating": "Unsafe", "category": "NA: None applying", "rationale": "x"}'
        cats, _ = interpret_llavaguard_output(text, unsafe_score=0.9)
        self.assertEqual(cats, {})

    def test_truncated_output_still_parses_rating_and_category(self) -> None:
        # max_new_tokens обрезал rationale — rating/category стоят первыми, ловим их.
        text = '{\n    "rating": "Unsafe",\n    "category": "O7: Self-Harm",\n    "rationale": "the imag'
        cats, info = interpret_llavaguard_output(text, unsafe_score=0.8)
        self.assertEqual(cats, {"self_harm": 0.8})
        self.assertEqual(info["llavaguard_category"], "O7")

    def test_animal_cruelty_maps_to_violence_gore(self) -> None:
        text = '{"rating": "Unsafe", "category": "O8: Animal Cruelty", "rationale": "x"}'
        cats, _ = interpret_llavaguard_output(text, unsafe_score=0.9)
        self.assertEqual(cats, {"violence_gore": 0.9})

    def test_garbage_output_degrades_gracefully(self) -> None:
        cats, info = interpret_llavaguard_output("not json at all", unsafe_score=0.9)
        self.assertEqual(cats, {})
        self.assertIsNone(info["rating"])


if __name__ == "__main__":
    unittest.main()
