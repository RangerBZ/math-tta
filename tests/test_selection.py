import unittest

from math_tta.data import Candidate
from math_tta.selection import select_trajectories


class SelectionTests(unittest.TestCase):
    def test_selects_unique_majority_and_reports_value(self) -> None:
        candidates = [
            Candidate("p1", "question", r"Work \boxed{2}"),
            Candidate("p1", "question", r"Work \boxed{2}"),
            Candidate("p1", "question", r"Work \boxed{3}"),
            Candidate("p1", "question", "No final answer"),
        ]

        selected = select_trajectories(candidates, min_value=0.5)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].predicted_answer, "2")
        self.assertEqual(selected[0].votes, 2)
        self.assertEqual(selected[0].samples, 4)
        self.assertEqual(selected[0].value, 0.5)

    def test_drops_tied_votes_and_values_below_threshold(self) -> None:
        candidates = [
            Candidate("tie", "question", r"\boxed{2}"),
            Candidate("tie", "question", r"\boxed{3}"),
            Candidate("low", "question", r"\boxed{2}"),
            Candidate("low", "question", r"\boxed{3}"),
            Candidate("low", "question", r"\boxed{3}"),
            Candidate("low", "question", r"\boxed{4}"),
        ]

        self.assertEqual(select_trajectories(candidates, min_value=0.6), [])

    def test_rejects_inconsistent_question_text(self) -> None:
        candidates = [
            Candidate("p1", "question one", r"\boxed{2}"),
            Candidate("p1", "question two", r"\boxed{2}"),
        ]
        with self.assertRaisesRegex(ValueError, "different questions"):
            select_trajectories(candidates)


if __name__ == "__main__":
    unittest.main()
