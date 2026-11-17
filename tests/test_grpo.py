import unittest

from math_tta.data import Candidate, Problem, Rollout
from math_tta.rl_training import (
    _padded_old_logprob_rows,
    group_relative_advantages,
    run_grpo_training,
)


class GrpoTests(unittest.TestCase):
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
