import unittest
from io import StringIO
from unittest.mock import patch

from math_tta.data import Candidate, Problem, Rollout
from math_tta.rl_training import (
    _padded_old_logprob_rows,
    _print_progress,
    assign_pseudo_label_rewards,
    group_relative_advantages,
    run_grpo_training,
)


class GrpoTests(unittest.TestCase):
    def test_progress_bar_reports_completion(self) -> None:
        output = StringIO()
        with patch("math_tta.rl_training.sys.stderr", output):
            _print_progress("training", 3, 3)

        self.assertIn("[########################] 3/3 (100%)", output.getvalue())

    def test_group_advantages_normalize_mixed_rewards_and_zero_constant_groups(
        self,
    ) -> None:
        advantages = group_relative_advantages(
            ["q1", "q1", "q2", "q2"],
            [0.0, 1.0, 0.0, 0.0],
        )

        self.assertEqual(advantages, [-1.0, 1.0, 0.0, 0.0])

    def test_old_logprobs_align_to_completion_positions(self) -> None:
        rollout = Rollout(
            candidate=Candidate(
                id="q1",
                question="question",
                completion="answer",
                completion_token_ids=(10, 11),
                prompt_token_ids=(1, 2, 3),
            ),
            reward=1.0,
            old_policy_logprobs=(-0.5, -0.75),
            advantage=1.0,
        )

        rows = _padded_old_logprob_rows([rollout], width=5)

        self.assertEqual(rows, [[0.0, 0.0, -0.5, -0.75, 0.0]])

    def test_pseudo_label_rewards_use_normalized_plurality_and_record_consensus(
        self,
    ) -> None:
        candidates = [
            Candidate("q1", "question", r"\boxed{1/2}"),
            Candidate("q1", "question", r"\boxed{0.5}"),
            Candidate("q1", "question", r"\boxed{2}"),
            Candidate("q1", "question", "No final answer."),
        ]

        rewards, groups = assign_pseudo_label_rewards(candidates)

        self.assertEqual(rewards, [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(groups[0]["pseudo_answer"], "1/2")
        self.assertEqual(groups[0]["consensus"], 0.5)
        self.assertTrue(groups[0]["accepted"])

    def test_pseudo_label_ties_and_low_consensus_abstain(self) -> None:
        tied = [
            Candidate("q1", "question", r"\boxed{1}"),
            Candidate("q1", "question", r"\boxed{2}"),
            Candidate("q1", "question", r"\boxed{2.0}"),
            Candidate("q1", "question", r"\boxed{1.0}"),
        ]
        rewards, groups = assign_pseudo_label_rewards(tied)
        self.assertEqual(rewards, [0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(groups[0]["pseudo_answer"])
        self.assertFalse(groups[0]["accepted"])

        candidates = [
            Candidate("q2", "another question", r"\boxed{1}"),
            Candidate("q2", "another question", r"\boxed{1}"),
            Candidate("q2", "another question", r"\boxed{2}"),
            Candidate("q2", "another question", r"\boxed{3}"),
        ]
        rewards, groups = assign_pseudo_label_rewards(
            candidates,
            min_consensus=0.75,
        )
        self.assertEqual(rewards, [0.0, 0.0, 0.0, 0.0])
        self.assertFalse(groups[0]["accepted"])

    def test_pseudo_label_mode_does_not_require_reference_answers(self) -> None:
        with (
            patch("math_tta.rl_training.set_seed"),
            patch(
                "math_tta.rl_training.load_trainable_model",
                side_effect=RuntimeError("model loader reached"),
            ) as load_model,
        ):
            with self.assertRaisesRegex(RuntimeError, "model loader reached"):
                run_grpo_training(
                    [Problem("train-00001", "question")],
                    model_id="unused",
                    output_dir="unused",
                    reward_mode="pseudo-label",
                )
        load_model.assert_called_once()

    def test_verifier_mode_still_requires_reference_answers(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference answer"):
            run_grpo_training(
                [Problem("train-00001", "question")],
                model_id="unused",
                output_dir="unused",
            )

    def test_rejects_clip_epsilon_outside_open_unit_interval_before_model_load(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "clip_epsilon"):
            run_grpo_training(
                [Problem("train-00001", "question", "42")],
                model_id="unused",
                output_dir="unused",
                clip_epsilon=1.0,
            )

    def test_requires_multiple_samples_per_problem(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            run_grpo_training(
                [Problem("train-00001", "question", "42")],
                model_id="unused",
                output_dir="unused",
                samples=1,
            )


if __name__ == "__main__":
    unittest.main()
