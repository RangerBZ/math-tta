import unittest

from math_tta.data import Candidate, Problem
from math_tta.metrics import evaluate


class MetricTests(unittest.TestCase):
    def test_reports_accuracy_and_selection_coverage(self) -> None:
        problems = [
            Problem("p1", "question one", "2"),
            Problem("p2", "question two", "9"),
        ]
        candidates = [
            Candidate("p1", "question one", r"\boxed{2}"),
            Candidate("p1", "question one", r"\boxed{2}"),
            Candidate("p1", "question one", r"\boxed{3}"),
            Candidate("p2", "question two", r"\boxed{8}"),
            Candidate("p2", "question two", r"\boxed{8}"),
        ]

        result = evaluate(problems, candidates, min_value=0.5)

        self.assertEqual(result["candidate_pass_at_k"], 0.5)
        self.assertEqual(result["selection_coverage"], 1.0)
        self.assertEqual(result["selected_accuracy"], 0.5)
        self.assertEqual(result["selected_count"], 2)


if __name__ == "__main__":
    unittest.main()
