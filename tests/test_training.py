import unittest

from math_tta.data import Problem
from math_tta.training import run_self_training


class TrainingTests(unittest.TestCase):
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
