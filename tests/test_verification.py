import unittest

from math_tta.data import Candidate, Problem
from math_tta.verification import verify_candidates, verification_metrics


class VerificationTests(unittest.TestCase):
    def test_uses_reference_answer_not_sample_majority(self) -> None:
        problem = Problem("p1", "question", "Reasoning\n#### 7")
        candidates = [
            Candidate("p1", "question", r"\boxed{5}"),
            Candidate("p1", "question", r"\boxed{5}"),
            Candidate("p1", "question", r"\boxed{7}"),
        ]

        verified = verify_candidates([problem], candidates)

        self.assertEqual([item.is_correct for item in verified], [False, False, True])
        self.assertEqual(verified[0].gold_answer, "Reasoning\n#### 7")
        self.assertEqual(verification_metrics(verified)["problem_pass_at_k"], 1.0)

    def test_unparseable_and_empty_outputs_are_failed_attempts(self) -> None:
        problem = Problem("p1", "question", "7")
        candidates = [
            Candidate("p1", "question", ""),
            Candidate("p1", "question", "I cannot determine the answer."),
        ]

        verified = verify_candidates([problem], candidates)
        metrics = verification_metrics(verified)

        self.assertEqual(metrics["candidates"], 2)
        self.assertEqual(metrics["parseable_candidates"], 0)
        self.assertEqual(metrics["candidate_accuracy"], 0.0)
        self.assertEqual(metrics["problem_pass_at_k"], 0.0)

    def test_reports_raw_sampled_token_log_probability_summary(self) -> None:
        verified = verify_candidates(
            [Problem("p1", "question", "7")],
            [
                Candidate(
                    "p1",
                    "question",
                    r"\boxed{7}",
                    token_logprobs=(-0.5, -1.5),
                )
            ],
        )

        metrics = verification_metrics(verified)

        self.assertEqual(metrics["candidates_with_token_logprobs"], 1)
        self.assertEqual(metrics["mean_sampled_token_logprob"], -1.0)

    def test_rejects_missing_or_mismatched_training_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "no reference answer"):
            verify_candidates(
                [Problem("p1", "question")],
                [Candidate("p1", "question", r"\boxed{7}")],
            )
        with self.assertRaisesRegex(ValueError, "differs from problem data"):
            verify_candidates(
                [Problem("p1", "question", "7")],
                [Candidate("p1", "different", r"\boxed{7}")],
            )


if __name__ == "__main__":
    unittest.main()
