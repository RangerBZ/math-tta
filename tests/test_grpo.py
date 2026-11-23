from math import log
import unittest
from unittest.mock import Mock, patch

from math_tta.data import Candidate, Problem, Rollout
from math_tta.rl_training import (
    _assign_tta1_rewards,
    _generate_problem_shard,
    _partition_problems,
    _padded_old_logprob_rows,
    _assign_tta2_rewards,
    assign_pseudo_label_rewards,
    group_relative_advantages,
    run_grpo_training,
    _token_self_certainty,
    _tta1_group_decision,
)


class GrpoTests(unittest.TestCase):
    def test_problem_shards_are_balanced_and_keep_original_indices(self) -> None:
        problems = [Problem(f"q{index}", f"question {index}") for index in range(5)]

        shards = _partition_problems(problems, worker_count=2)

        self.assertEqual(
            [[index for index, _ in shard] for shard in shards],
            [[0, 2, 4], [1, 3]],
        )

    def test_generate_problem_shard_returns_all_samples_per_original_problem(
        self,
    ) -> None:
        problems = [
            (3, Problem("q3", "question 3")),
            (7, Problem("q7", "question 7")),
        ]
        candidates = [
            Candidate("q3", "question 3", "answer 1"),
            Candidate("q3", "question 3", "answer 2"),
            Candidate("q7", "question 7", "answer 1"),
            Candidate("q7", "question 7", "answer 2"),
        ]
        progress = Mock()

        with patch(
            "math_tta.rl_training.generate_candidate_batch",
            return_value=iter(candidates),
        ) as generate:
            result = _generate_problem_shard(
                problems,
                model=object(),
                tokenizer=object(),
                samples=2,
                max_new_tokens=16,
                generation_batch_size=1,
                progress_bar=progress,
            )

        self.assertEqual(
            [(index, [candidate.id for candidate in group]) for index, group in result],
            [(3, ["q3", "q3"]), (7, ["q7", "q7"])],
        )
        self.assertEqual(progress.update.call_count, 2)
        generate.assert_called_once()

    def test_requires_tqdm_before_loading_model(self) -> None:
        with (
            patch("math_tta.rl_training.set_seed"),
            patch(
                "math_tta.rl_training.load_tqdm",
                side_effect=RuntimeError("tqdm is required"),
            ),
            patch("math_tta.rl_training.load_trainable_model") as load_model,
        ):
            with self.assertRaisesRegex(RuntimeError, "tqdm is required"):
                run_grpo_training(
                    [Problem("train-00001", "question")],
                    model_id="unused",
                    output_dir="unused",
                    reward_mode="pseudo-label",
                )
        load_model.assert_not_called()

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

    def test_tta1_extends_below_consensus_threshold_and_accepts_agreement(self) -> None:
        candidates = [
            Candidate("q1", "question", r"\boxed{1}"),
            Candidate("q1", "question", r"\boxed{1.0}"),
            Candidate("q1", "question", r"\boxed{2}"),
            Candidate("q1", "question", r"\boxed{3}"),
        ]
        decision = _tta1_group_decision(
            candidates,
            [0.7, 0.7, 0.6, 0.5],
            threshold=0.8,
        )

        self.assertEqual(decision["count_winner"], "1")
        self.assertEqual(decision["consensus"], 0.5)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["decision_reason"], "below_consensus_threshold")

        unanimous = [
            Candidate("q2", "another question", r"\boxed{7}")
            for _ in range(4)
        ]
        decision = _tta1_group_decision(
            unanimous,
            [0.5, 0.6, 0.4, 0.7],
            threshold=0.8,
        )
        self.assertEqual(decision["count_winner"], "7")
        self.assertEqual(decision["confidence_winner"], "7")
        self.assertTrue(decision["accepted"])

    def test_tta1_adds_sampling_when_confidence_weighted_winner_conflicts(
        self,
    ) -> None:
        candidates = [
            Candidate("q1", "question", r"\boxed{1}")
            for _ in range(5)
        ] + [Candidate("q1", "question", r"\boxed{2}")]

        decision = _tta1_group_decision(
            candidates,
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            threshold=0.8,
        )

        self.assertEqual(decision["consensus"], 5 / 6)
        self.assertEqual(decision["count_winner"], "1")
        self.assertEqual(decision["confidence_winner"], "2")
        self.assertTrue(decision["confidence_conflict"])
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["decision_reason"], "confidence_conflict")

    def test_tta1_does_not_reward_conflicting_winners(self) -> None:
        candidates = [
            Candidate("q1", "question", r"\boxed{1}")
            for _ in range(5)
        ] + [Candidate("q1", "question", r"\boxed{2}")]
        rewards, groups = _assign_tta1_rewards(
            candidates,
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            threshold=0.8,
        )

        self.assertEqual(rewards, [0.0] * 6)
        self.assertFalse(groups[0]["accepted"])
        self.assertEqual(groups[0]["sample_count"], 6)

        agreed_candidates = [
            Candidate("q2", "another question", r"\boxed{1}")
            for _ in range(5)
        ] + [Candidate("q2", "another question", r"\boxed{2}")]
        rewards, groups = _assign_tta1_rewards(
            agreed_candidates,
            [0.3, 0.3, 0.3, 0.3, 0.3, 0.1],
            threshold=0.8,
        )
        self.assertEqual(rewards, [1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        self.assertTrue(groups[0]["accepted"])

    def test_tta2_weights_accepted_pseudo_label_rewards_by_confidence(self) -> None:
        candidates = [
            Candidate("q1", "question", r"\boxed{1}")
            for _ in range(4)
        ]
        rewards, groups = _assign_tta2_rewards(
            candidates,
            [0.2, 0.4, 0.6, 0.8],
            threshold=0.8,
        )

        self.assertEqual(rewards, [0.2, 0.4, 0.6, 0.8])
        self.assertEqual(
            groups[0]["reward_scheme"],
            "pseudo_label_match_times_self_certainty",
        )
        self.assertNotEqual(
            group_relative_advantages(["q1"] * 4, rewards),
            [0.0] * 4,
        )

    def test_tta2_gives_zero_reward_to_rejected_groups_and_dissenters(self) -> None:
        rejected = [
            Candidate("q1", "question", r"\boxed{1}"),
            Candidate("q1", "question", r"\boxed{1}"),
            Candidate("q1", "question", r"\boxed{2}"),
            Candidate("q1", "question", r"\boxed{3}"),
        ]
        rewards, groups = _assign_tta2_rewards(
            rejected,
            [0.7, 0.7, 0.6, 0.5],
            threshold=0.8,
        )
        self.assertEqual(rewards, [0.0] * 4)
        self.assertFalse(groups[0]["accepted"])

        accepted = [
            Candidate("q2", "another question", r"\boxed{1}"),
            Candidate("q2", "another question", r"\boxed{1.0}"),
            Candidate("q2", "another question", r"\boxed{1}"),
            Candidate("q2", "another question", r"\boxed{2}"),
            Candidate("q2", "another question", r"\boxed{1}"),
            Candidate("q2", "another question", r"\boxed{1}"),
        ]
        rewards, groups = _assign_tta2_rewards(
            accepted,
            [0.3, 0.5, 0.4, 0.2, 0.6, 0.8],
            threshold=0.8,
        )
        self.assertEqual(rewards, [0.3, 0.5, 0.4, 0.0, 0.6, 0.8])
        self.assertTrue(groups[0]["accepted"])

    def test_tta1_confidence_uses_normalized_mean_token_entropy(self) -> None:
        self.assertEqual(_token_self_certainty(0.0, 100), 1.0)
        self.assertAlmostEqual(
            _token_self_certainty(0.5 * log(100), 100),
            0.5,
        )
        self.assertEqual(_token_self_certainty(log(100), 100), 0.0)

    def test_pseudo_label_rejects_below_minimum_consensus(self) -> None:
        candidates = [
            Candidate("q1", "question", r"\boxed{1}"),
            Candidate("q1", "question", r"\boxed{1}"),
            Candidate("q1", "question", r"\boxed{2}"),
            Candidate("q1", "question", r"\boxed{3}"),
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
            patch("math_tta.rl_training.load_tqdm", return_value=object()),
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

    def test_tta1_requires_pseudo_label_rewards_before_loading_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires pseudo-label"):
            run_grpo_training(
                [Problem("train-00001", "question", "42")],
                model_id="unused",
                output_dir="unused",
                sampling_policy="tta-1",
                reward_mode="verifier",
            )

    def test_tta1_requires_max_samples_to_cover_initial_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least samples"):
            run_grpo_training(
                [Problem("train-00001", "question")],
                model_id="unused",
                output_dir="unused",
                samples=4,
                sampling_policy="tta-1",
                max_samples=3,
                reward_mode="pseudo-label",
            )

    def test_tta2_requires_pseudo_label_rewards_before_loading_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires pseudo-label"):
            run_grpo_training(
                [Problem("train-00001", "question", "42")],
                model_id="unused",
                output_dir="unused",
                sampling_policy="tta-2",
                reward_mode="verifier",
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
