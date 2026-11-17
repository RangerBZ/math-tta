import json
import tempfile
import unittest
from pathlib import Path

from math_tta.data import load_candidates


class DataTests(unittest.TestCase):
    def test_loads_empty_attempts_and_optional_token_logprobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "p1",
                        "question": "question",
                        "completion": "",
                        "token_logprobs": [-0.5, -1.25],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = load_candidates(path)

        self.assertEqual(candidates[0].completion, "")
        self.assertEqual(candidates[0].token_logprobs, (-0.5, -1.25))

    def test_rejects_non_finite_token_logprobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            path.write_text(
                '{"id":"p1","question":"q","completion":"x",'
                '"token_logprobs":[NaN]}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "finite numbers"):
                load_candidates(path)

    def test_loads_exact_completion_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "p1",
                        "question": "question",
                        "completion": "answer",
                        "completion_token_ids": [10, 11, 99],
                        "prompt_token_ids": [1, 2, 3],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            candidates = load_candidates(path)

        self.assertEqual(candidates[0].completion_token_ids, (10, 11, 99))
        self.assertEqual(candidates[0].prompt_token_ids, (1, 2, 3))

    def test_rejects_non_integer_completion_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            path.write_text(
                '{"id":"p1","question":"q","completion":"x",'
                '"completion_token_ids":[10.5]}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "list of integers"):
                load_candidates(path)

    def test_rejects_boolean_completion_token_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            path.write_text(
                '{"id":"p1","question":"q","completion":"x",'
                '"completion_token_ids":[true]}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "list of integers"):
                load_candidates(path)


if __name__ == "__main__":
    unittest.main()
