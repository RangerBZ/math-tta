"""Outcome verification for generated mathematical reasoning traces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from math_tta.answers import extract_answer, normalize_answer
from math_tta.data import Candidate, Problem


@dataclass(frozen=True)
class VerifiedCandidate:
    id: str
    question: str
    completion: str
    gold_answer: str
    predicted_answer: str | None
    is_correct: bool
    token_logprobs: tuple[float, ...] | None = None

    @property
    def is_parseable(self) -> bool:
        return self.predicted_answer is not None

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["is_parseable"] = self.is_parseable
        return record


def normalize_reference_answer(answer: str) -> str:
    extracted = extract_answer(answer)
    return normalize_answer(extracted if extracted is not None else answer)


def verify_candidates(
    problems: Iterable[Problem],
    candidates: Iterable[Candidate],
) -> list[VerifiedCandidate]:
    problem_list = list(problems)
    problem_by_id = {problem.id: problem for problem in problem_list}
    if not problem_by_id:
        raise ValueError("verification requires at least one problem")
    if len(problem_by_id) != len(problem_list):
        raise ValueError("problem data contains duplicate ids")

    verified: list[VerifiedCandidate] = []
    for candidate in candidates:
        problem = problem_by_id.get(candidate.id)
        if problem is None:
            raise ValueError(f"candidate id {candidate.id!r} is missing from problem data")
        if candidate.question != problem.question:
            raise ValueError(f"candidate {candidate.id!r} question differs from problem data")
        if problem.answer is None:
            raise ValueError(f"problem {problem.id!r} has no reference answer")
        if not problem.answer.strip():
            raise ValueError(f"problem {problem.id!r} has an empty reference answer")

        predicted_answer = extract_answer(candidate.completion)
        is_correct = (
            predicted_answer is not None
            and normalize_answer(predicted_answer)
            == normalize_reference_answer(problem.answer)
        )
        verified.append(
            VerifiedCandidate(
                id=candidate.id,
                question=problem.question,
                completion=candidate.completion,
                gold_answer=problem.answer,
                predicted_answer=predicted_answer,
                is_correct=is_correct,
                token_logprobs=candidate.token_logprobs,
            )
        )
    return verified


def verification_metrics(
    verified_candidates: Iterable[VerifiedCandidate],
) -> dict[str, float | int | None]:
    verified = list(verified_candidates)
    if not verified:
        raise ValueError("verification metrics require at least one candidate")

    by_problem: dict[str, list[VerifiedCandidate]] = defaultdict(list)
    for candidate in verified:
        by_problem[candidate.id].append(candidate)

    correct = sum(candidate.is_correct for candidate in verified)
    parseable = sum(candidate.is_parseable for candidate in verified)
    sampled_logprobs = [
        value
        for candidate in verified
        if candidate.token_logprobs is not None
        for value in candidate.token_logprobs
    ]
    problems_with_correct = sum(
        any(candidate.is_correct for candidate in group)
        for group in by_problem.values()
    )
    return {
        "problems": len(by_problem),
        "candidates": len(verified),
        "parseable_candidates": parseable,
        "correct_candidates": correct,
        "candidate_accuracy": correct / len(verified),
        "parse_rate": parseable / len(verified),
        "problem_pass_at_k": problems_with_correct / len(by_problem),
        "accepted_traces": correct,
        "candidates_with_token_logprobs": sum(
            candidate.token_logprobs is not None for candidate in verified
        ),
        "mean_sampled_token_logprob": (
            sum(sampled_logprobs) / len(sampled_logprobs)
            if sampled_logprobs
            else None
        ),
    }
