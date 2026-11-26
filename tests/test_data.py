import json
import tempfile
import unittest
from pathlib import Path

from math_tta.data import load_candidates, prepare_svamp


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


    def test_prepare_svamp_converts_body_question_and_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "SVAMP.json"
            output = Path(directory) / "svamp.jsonl"
            source.write_text(
                json.dumps(
                    [
                        {
                            "Body": "A box has 3 pens.",
                            "Question": "How many pens are there?",
                            "Answer": 3,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(prepare_svamp(source, output), 1)
            row = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(row["id"], "svamp-00000")
            self.assertEqual(
                row["question"], "A box has 3 pens. How many pens are there?"
            )
            self.assertEqual(row["answer"], "3")

    def test_prepare_svamp_rejects_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "SVAMP.json"
            output = Path(directory) / "svamp.jsonl"
            source.write_text(
                json.dumps([{"Body": "missing question", "Answer": 1}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Body, Question, and Answer"):
                prepare_svamp(source, output)


if __name__ == "__main__":
    unittest.main()
