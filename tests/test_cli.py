from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from math_tta.cli import _parser, main
from math_tta.data import Problem


class CliTests(unittest.TestCase):
    def test_grpo_problem_limit_reaches_training(self) -> None:
        problems = [
            Problem(f"q{index}", f"question {index}", "1")
            for index in range(4)
        ]
        with (
            patch("math_tta.cli.load_problems", return_value=problems),
            patch("math_tta.cli.run_grpo_training", return_value=[]) as train,
            redirect_stdout(StringIO()),
        ):
            result = main(
                [
                    "grpo",
                    "--input",
                    "train.jsonl",
                    "--output-dir",
                    "output",
                    "--limit-problems",
                    "2",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(len(train.call_args.args[0]), 2)

    def test_grpo_accepts_sampling_device_count(self) -> None:
        arguments = _parser().parse_args(
            [
                "grpo",
                "--input",
                "train.jsonl",
                "--output-dir",
                "output",
                "--sampling-devices",
                "2",
            ]
        )

        self.assertEqual(arguments.sampling_devices, 2)

    def test_grpo_accepts_pseudo_label_reward_mode(self) -> None:
        arguments = _parser().parse_args(
            [
                "grpo",
                "--input",
                "train.jsonl",
                "--output-dir",
                "output",
                "--reward-mode",
                "pseudo-label",
                "--min-consensus",
                "0.75",
            ]
        )

        self.assertEqual(arguments.reward_mode, "pseudo-label")
        self.assertEqual(arguments.min_consensus, 0.75)


if __name__ == "__main__":
    unittest.main()
