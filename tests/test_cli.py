from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import MagicMock, call, patch

from math_tta.cli import _parser, main
from math_tta.data import Candidate, Problem


class CliTests(unittest.TestCase):
    def test_generate_progress_updates_once_per_problem(self) -> None:
        problems = [
            Problem("q1", "question 1"),
            Problem("q2", "question 2"),
        ]
        candidates = [
            Candidate("q1", "question 1", "answer 1"),
            Candidate("q1", "question 1", "answer 2"),
            Candidate("q2", "question 2", "answer 1"),
            Candidate("q2", "question 2", "answer 2"),
        ]
        progress_bar = MagicMock()
        progress_bar.__enter__.return_value = progress_bar
        progress = MagicMock(return_value=progress_bar)

        with (
            patch("math_tta.cli.load_problems", return_value=problems),
            patch(
                "math_tta.cli.generate_candidates",
                return_value=iter(candidates),
            ),
            patch("math_tta.cli.load_tqdm", return_value=progress),
            patch(
                "math_tta.cli.write_jsonl",
                side_effect=lambda _, records: list(records),
            ),
            redirect_stdout(StringIO()),
        ):
            result = main(
                [
                    "generate",
                    "--input",
                    "test.jsonl",
                    "--output",
                    "candidates.jsonl",
                    "--model",
                    "unused",
                    "--samples",
                    "2",
                ]
            )

        self.assertEqual(result, 0)
        progress.assert_called_once_with(
            total=2,
            desc="Generating",
            unit="problem",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        self.assertEqual(
            progress_bar.update.call_args_list,
            [call(1), call(1)],
        )

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

    def test_grpo_tta1_options_reach_training(self) -> None:
        with (
            patch(
                "math_tta.cli.load_problems",
                return_value=[Problem("q1", "question")],
            ),
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
                    "--reward-mode",
                    "pseudo-label",
                    "--sampling-policy",
                    "tta-1",
                    "--samples",
                    "4",
                    "--max-samples",
                    "8",
                    "--tta1-threshold",
                    "0.8",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(train.call_args.kwargs["sampling_policy"], "tta-1")
        self.assertEqual(train.call_args.kwargs["max_samples"], 8)
        self.assertEqual(train.call_args.kwargs["tta1_threshold"], 0.8)

    def test_grpo_accepts_tta2_sampling_policy(self) -> None:
        arguments = _parser().parse_args(
            [
                "grpo",
                "--input",
                "train.jsonl",
                "--output-dir",
                "output",
                "--reward-mode",
                "pseudo-label",
                "--sampling-policy",
                "tta-2",
            ]
        )

        self.assertEqual(arguments.sampling_policy, "tta-2")

    def test_grpo_forwards_tta2_sampling_policy(self) -> None:
        with (
            patch(
                "math_tta.cli.load_problems",
                return_value=[Problem("q1", "question")],
            ),
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
                    "--reward-mode",
                    "pseudo-label",
                    "--sampling-policy",
                    "tta-2",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(train.call_args.kwargs["sampling_policy"], "tta-2")

    def test_grpo_accepts_and_forwards_tta3_sampling_policy(self) -> None:
        arguments = _parser().parse_args(
            [
                "grpo",
                "--input",
                "train.jsonl",
                "--output-dir",
                "output",
                "--reward-mode",
                "pseudo-label",
                "--sampling-policy",
                "tta-3",
            ]
        )
        self.assertEqual(arguments.sampling_policy, "tta-3")

        with (
            patch(
                "math_tta.cli.load_problems",
                return_value=[Problem("q1", "question", "#### 1")],
            ),
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
                    "--reward-mode",
                    "pseudo-label",
                    "--sampling-policy",
                    "tta-3",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(train.call_args.kwargs["sampling_policy"], "tta-3")

    def test_grpo_accepts_and_forwards_tta4_sampling_policy(self) -> None:
        arguments = _parser().parse_args(
            [
                "grpo",
                "--input",
                "train.jsonl",
                "--output-dir",
                "output",
                "--reward-mode",
                "pseudo-label",
                "--sampling-policy",
                "tta-4",
            ]
        )
        self.assertEqual(arguments.sampling_policy, "tta-4")

        with (
            patch(
                "math_tta.cli.load_problems",
                return_value=[
                    Problem("train-00001", "question 1", "#### 1"),
                    Problem("train-00002", "question 2", "#### 2"),
                ],
            ),
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
                    "--reward-mode",
                    "pseudo-label",
                    "--sampling-policy",
                    "tta-4",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(train.call_args.kwargs["sampling_policy"], "tta-4")


if __name__ == "__main__":
    unittest.main()
