"""Agreement-based trajectory-value estimates and pseudo-label selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from math_tta.answers import extract_answer, normalize_answer
from math_tta.data import Candidate


@dataclass(frozen=True)
class Selection:
    id: str
    question: str
    completion: str
    predicted_answer: str
    value: float
    votes: int
    samples: int

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def select_trajectories(
    candidates: Iterable[Candidate],
    min_value: float = 0.5,
) -> list[Selection]:
    if not 0.0 <= min_value <= 1.0:
        raise ValueError("min_value must be between 0 and 1")

    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.id].append(candidate)

    selected: list[Selection] = []
    for problem_id, group in groups.items():
        question = group[0].question
        if any(candidate.question != question for candidate in group):
            raise ValueError(f"candidates for {problem_id!r} contain different questions")

        parsed = [
            (candidate, extract_answer(candidate.completion))
            for candidate in group
        ]
        votes = Counter(
            normalize_answer(answer)
            for _, answer in parsed
            if answer is not None
        )
        if not votes:
            continue

        ranked_answers = votes.most_common()
        winning_answer, winning_votes = ranked_answers[0]
        if len(ranked_answers) > 1 and ranked_answers[1][1] == winning_votes:
            continue

        value = winning_votes / len(group)
        if value < min_value:
            continue

        representative = next(
            candidate
            for candidate, answer in parsed
            if answer is not None and normalize_answer(answer) == winning_answer
        )
        selected.append(
            Selection(
                id=problem_id,
                question=question,
                completion=representative.completion,
                predicted_answer=winning_answer,
                value=value,
                votes=winning_votes,
                samples=len(group),
            )
        )
    return selected
