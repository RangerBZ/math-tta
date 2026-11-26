from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from math_tta.audit_rollouts import (
    _bayesian_calibration_diagnostics,
    _confidence_correctness_diagnostics,
    _tta4_group_reliability_diagnostics,
    audit_rollouts,
    main,
)
from math_tta.data import Candidate, Problem, write_jsonl
from math_tta.rl_training import _assign_tta4_rewards, group_relative_advantages


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

    def test_tta2_audit_simulates_confidence_shaped_rewards(self) -> None:
        problem = Problem("q1", "question", "rationale #### 1")
        candidates = [
            Candidate("q1", "question", r"\boxed{1}")
            for _ in range(4)
        ]
        details, summaries = audit_rollouts(
            [problem],
            candidates,
            policy="tta-2",
            confidence_weights=[0.2, 0.4, 0.6, 0.8],
            stored_rewards=[None] * 4,
            stored_advantages=[None] * 4,
            initial_samples=4,
        )

        prefix = details[0]["prefixes"][0]
        self.assertEqual(prefix["rewards"], [0.2, 0.4, 0.6, 0.8])
        self.assertEqual(prefix["advantage_zero_count"], 0)
        self.assertEqual(summaries[0]["reward_positive_count"], 4)
        self.assertEqual(summaries[0]["mixed_reward_groups"], 1)
        confidence_report = _confidence_correctness_diagnostics(details)
        self.assertEqual(confidence_report["sample_count"], 4)
        self.assertEqual(confidence_report["mean_confidence_correct"], 0.5)
        self.assertEqual(
            sum(bucket["count"] for bucket in confidence_report["bins"]),
            4,
        )

    def test_tta3_audit_replays_bayesian_rewards_and_reports_calibration(self) -> None:
        problems = [
            Problem("q1", "question 1", "rationale #### 1"),
            Problem("q2", "question 2", "rationale #### 2"),
        ]
        candidates = [
            Candidate("q1", "question 1", r"\boxed{1}")
            for _ in range(4)
        ] + [
            Candidate("q2", "question 2", r"\boxed{2}")
            for _ in range(4)
        ]
        confidences = [0.2, 0.4, 0.6, 0.8] * 2
        details, summaries = audit_rollouts(
            problems,
            candidates,
            policy="tta-3",
            confidence_weights=confidences,
            stored_rewards=[None] * 8,
            stored_advantages=[None] * 8,
            initial_samples=4,
        )

        self.assertEqual(len(details), 2)
        self.assertTrue(
            all(
                rollout["bayesian_correctness_probability"] is not None
                for detail in details
                for rollout in detail["rollouts"]
            )
        )
        self.assertEqual(
            _bayesian_calibration_diagnostics(details)["sample_count"],
            8,
        )
        self.assertEqual(summaries[0]["accepted_groups"], 2)

        stored_rewards = [
            reward
            for detail in details
            for reward in detail["prefixes"][-1]["rewards"]
        ]
        stored_advantages = [
            advantage
            for detail in details
            for advantage in detail["prefixes"][-1]["advantages"]
        ]
        _, replay_summaries = audit_rollouts(
            problems,
            candidates,
            policy="tta-3",
            confidence_weights=confidences,
            stored_rewards=stored_rewards,
            stored_advantages=stored_advantages,
            initial_samples=4,
        )
        self.assertEqual(replay_summaries[0]["stored_reward_mismatches"], 0)
        self.assertEqual(replay_summaries[0]["stored_advantage_mismatches"], 0)

    def test_tta4_audit_replays_rank_rewards_and_reports_reliability(self) -> None:
        problems = [
            Problem("q1", "question 1", "rationale #### 1"),
            Problem("q2", "question 2", "rationale #### 2"),
        ]
        candidates = [
            Candidate("q1", "question 1", r"\boxed{1}")
            for _ in range(4)
        ] + [
            Candidate("q2", "question 2", r"\boxed{2}")
            for _ in range(4)
        ]
        confidences = [0.2, 0.4, 0.6, 0.8] * 2

        details, summaries = audit_rollouts(
            problems,
            candidates,
            policy="tta-4",
            confidence_weights=confidences,
            stored_rewards=[None] * 8,
            stored_advantages=[None] * 8,
            initial_samples=4,
        )

        prefix = details[0]["prefixes"][-1]
        expected = [1.0, 1.0 + 1 / 3, 1.0 + 2 / 3, 2.0]
        for actual, value in zip(prefix["rewards"], expected, strict=True):
            self.assertAlmostEqual(actual, value)

        training_rewards, _ = _assign_tta4_rewards(
            candidates[:4],
            confidences[:4],
            {"q1": 0.9},
            threshold=0.8,
        )
        self.assertEqual(prefix["rewards"], training_rewards)

        self.assertIsNotNone(details[0]["group_reliability_probability"])
        report = _tta4_group_reliability_diagnostics(details)
        self.assertTrue(report["available"])
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["observed_accuracy"], 1.0)
        self.assertEqual(summaries[0]["mixed_reward_groups"], 2)
        self.assertEqual(summaries[0]["nonzero_advantage_rollouts"], 8)

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

    def test_audit_cli_simulates_tta2_rewards_from_tta1_rollouts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            problems_path = root / "problems.jsonl"
            fixed_path = root / "fixed6.jsonl"
            tta1_path = root / "tta1.jsonl"
            tta4_path = root / "tta4.jsonl"
            output_dir = root / "audit"

            write_jsonl(
                problems_path,
                [{"id": "q1", "question": "question", "answer": "#### 1"}],
            )
            completions = [r"\boxed{1}"] * 6
            write_jsonl(
                fixed_path,
                (
                    {
                        "id": "q1",
                        "question": "question",
                        "completion": completion,
                        "reward": 1.0,
                        "advantage": 0.0,
                    }
                    for completion in completions
                ),
            )
            write_jsonl(
                tta1_path,
                (
                    {
                        "id": "q1",
                        "question": "question",
                        "completion": r"\boxed{1}",
                        "token_self_certainty": confidence,
                        "reward": 1.0,
                        "advantage": 0.0,
                    }
                    for confidence in (0.2, 0.4, 0.6, 0.8)
                ),
            )

            write_jsonl(
                tta4_path,
                (
                    {
                        "id": "q1",
                        "question": "question",
                        "completion": r"\boxed{1}",
                        "token_self_certainty": confidence,
                    }
                    for confidence in (0.2, 0.4, 0.6, 0.8)
                ),
            )

            with redirect_stdout(StringIO()):
                result = main(
                    [
                        "--problems",
                        str(problems_path),
                        "--fixed6-rollouts",
                        str(fixed_path),
                        "--tta1-rollouts",
                        str(tta1_path),
                        "--tta4-rollouts",
                        str(tta4_path),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads((output_dir / "summary.json").read_text())
            self.assertEqual(
                summary["runs"]["tta2"][0]["nonzero_advantage_rollouts"],
                4,
            )
            self.assertEqual(
                summary["confidence_correctness"]["tta2"]["sample_count"],
                4,
            )
            self.assertEqual(
                summary["runs"]["tta4"][0]["mixed_reward_groups"],
                1,
            )
            self.assertEqual(
                summary["runs"]["tta4"][0]["stored_reward_mismatches"],
                0,
            )
            self.assertFalse(
                summary["tta4_group_reliability"]["available"],
            )


if __name__ == "__main__":
    unittest.main()
