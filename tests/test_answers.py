import unittest

from math_tta.answers import extract_answer, normalize_answer


class AnswerTests(unittest.TestCase):
    def test_extracts_boxed_answer_with_nested_braces(self) -> None:
        self.assertEqual(
            extract_answer(r"Reasoning. Therefore \boxed{\frac{1}{2}}"),
            r"\frac{1}{2}",
        )

    def test_extracts_last_final_answer(self) -> None:
        self.assertEqual(
            extract_answer("Final answer: 3\nWait, final answer: 12."),
            "12",
        )

    def test_normalizes_fraction_and_decimal_equivalently(self) -> None:
        self.assertEqual(normalize_answer("6/4"), normalize_answer("1.5"))

    def test_returns_none_without_explicit_final_answer(self) -> None:
        self.assertIsNone(extract_answer("I think the answer may be 42."))


if __name__ == "__main__":
    unittest.main()
