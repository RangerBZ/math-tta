import unittest

from math_tta.data import Problem
from math_tta.training import _training_collator, run_self_training
from math_tta.verification import VerifiedCandidate


class FakeTorch:
    long = "long"

    @staticmethod
    def tensor(values, dtype):
        return values


class FakeTokenizer:
    chat_template = None
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text.split())))}


class TrainingTests(unittest.TestCase):
    def test_collator_uses_max_length_for_assistant_targets(self) -> None:
        tokenizer = FakeTokenizer()
        example = VerifiedCandidate(
            id="p1",
            question="question",
            completion="answer",
            gold_answer="answer",
            predicted_answer="answer",
            is_correct=True,
        )

        collate = _training_collator(tokenizer, FakeTorch, max_length=100)
        batch = collate([example])

        self.assertEqual(len(batch["input_ids"][0]), len(batch["labels"][0]))
        self.assertIn(-100, batch["labels"][0])

    def test_refuses_gsm8k_test_split_before_loading_model_dependencies(self) -> None:
        with self.assertRaisesRegex(ValueError, "test-split ids"):
            run_self_training(
                [Problem("test-00001", "question", "answer")],
                model_id="unused",
                output_dir="unused",
            )

    def test_rejects_duplicate_problem_ids_before_loading_dependencies(self) -> None:
        problems = [
            Problem("train-00001", "question one", "1"),
            Problem("train-00001", "question two", "2"),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate ids"):
            run_self_training(
                problems,
                model_id="unused",
                output_dir="unused",
            )


if __name__ == "__main__":
    unittest.main()
