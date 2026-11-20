import unittest

from math_tta.cli import _parser


class CliTests(unittest.TestCase):
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
