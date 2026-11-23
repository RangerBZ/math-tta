import unittest

from math_tta.audit_rollouts import audit_rollouts
from math_tta.data import Candidate, Problem
from math_tta.rl_training import group_relative_advantages


class RolloutAuditTests(unittest.TestCase):
    def test_fixed_prefix_reports_correctness_and_reward_changes(self) -> None:
        problem = Problem("q1", "question", "rationale #### 1")
        candidates = [
            Candidate("q1", "question", r"\boxed{1}")
            for _ in range(4)
        ] + [
            Candidate("q1", "question", r"\boxed{2}"),
            Candidate("q1", "question", r"\boxed{1}"),
        ]
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
        advantages = group_relative_advantages(["q1"] * 6, rewards)

        details, summaries = audit_rollouts(
            [problem],
            candidates,
            policy="fixed",
            confidence_weights=[None] * 6,
            stored_rewards=rewards,
            stored_advantages=advantages,
            initial_samples=4,
        )

        prefixes = details[0]["prefixes"]
        self.assertEqual([state["sample_count"] for state in prefixes], [4, 6])
        self.assertTrue(prefixes[0]["pseudo_label_correct_vs_reference"])
        self.assertEqual(prefixes[0]["advantage_zero_count"], 4)
        self.assertEqual(prefixes[1]["label_transition"], "unchanged")
        self.assertEqual(prefixes[1]["new_rollout_correct_count"], 1)
        self.assertEqual(prefixes[1]["previous_reward_flips"], 0)
        self.assertTrue(prefixes[1]["mixed_rewards"])
        self.assertEqual(summaries[-1]["reward_zero_count"], 1)
        self.assertEqual(summaries[-1]["reward_one_count"], 5)
        self.assertEqual(summaries[-1]["cumulative"]["accepted_groups"], 1)
        self.assertEqual(summaries[-1]["stored_reward_mismatches"], 0)
        self.assertEqual(summaries[-1]["stored_advantage_mismatches"], 0)

    def test_tta1_prefix_audit_reports_new_label_and_advantage_signal(self) -> None:
        problem = Problem("q1", "question", "rationale #### 1")
        candidates = [
            Candidate("q1", "question", r"\boxed{1}"),
            Candidate("q1", "question", r"\boxed{1.0}"),
            Candidate("q1", "question", r"\boxed{1}"),
            Candidate("q1", "question", r"\boxed{2}"),
            Candidate("q1", "question", r"\boxed{1}"),
            Candidate("q1", "question", r"\boxed{1}"),
        ]

        details, summaries = audit_rollouts(
            [problem],
            candidates,
            policy="tta-1",
            confidence_weights=[0.3, 0.3, 0.3, 0.1, 0.3, 0.3],
            stored_rewards=[None] * 6,
            stored_advantages=[None] * 6,
            initial_samples=4,
        )

        prefixes = details[0]["prefixes"]
        self.assertFalse(prefixes[0]["accepted"])
        self.assertEqual(prefixes[0]["consensus"], 0.75)
        self.assertEqual(prefixes[1]["label_transition"], "newly_accepted")
        self.assertTrue(prefixes[1]["pseudo_label_correct_vs_reference"])
        self.assertEqual(prefixes[1]["previous_reward_flips"], 3)
        self.assertTrue(prefixes[1]["mixed_rewards"])
        self.assertEqual(summaries[-1]["nonzero_advantage_rollouts"], 6)
        self.assertEqual(summaries[-1]["cumulative"]["accepted_groups"], 1)

    def test_audit_rejects_noncontiguous_rollout_groups(self) -> None:
        problems = [
            Problem("q1", "question 1", "#### 1"),
            Problem("q2", "question 2", "#### 2"),
        ]
        candidates = [
            Candidate("q1", "question 1", r"\boxed{1}"),
            Candidate("q2", "question 2", r"\boxed{2}"),
            Candidate("q1", "question 1", r"\boxed{1}"),
        ]

        with self.assertRaisesRegex(ValueError, "contiguous"):
            audit_rollouts(
                problems,
                candidates,
                policy="fixed",
                confidence_weights=[None] * 3,
                stored_rewards=[None] * 3,
                stored_advantages=[None] * 3,
                initial_samples=1,
            )


if __name__ == "__main__":
    unittest.main()
