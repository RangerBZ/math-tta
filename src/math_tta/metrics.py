"""Evaluation metrics for candidate generation and pseudo-label selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from math_tta.data import Candidate, Problem
from math_tta.selection import select_trajectories
from math_tta.verification import (
    normalize_reference_answer,
    verify_candidates,
    verification_metrics,
)


def evaluate(
    problems: Iterable[Problem],
    candidates: Iterable[Candidate],
    min_value: float = 0.5,
) -> dict[str, float | int]:
    problem_list = list(problems)
    candidate_list = list(candidates)
    labeled = [problem for problem in problem_list if problem.answer is not None]
    if not labeled:
        raise ValueError("evaluation requires at least one problem with an 'answer'")
    problem_by_id = {problem.id: problem for problem in problem_list}
    candidates_by_id: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidate_list:
        if candidate.id not in problem_by_id:
            raise ValueError(f"candidate id {candidate.id!r} is missing from problem data")
        candidates_by_id[candidate.id].append(candidate)
    evaluated = [
        problem
        for problem in labeled
        if candidates_by_id.get(problem.id)
    ]
    if not evaluated:
        raise ValueError("no labeled problems have generated candidates")

    evaluated_ids = {problem.id for problem in evaluated}
    relevant_candidates = [
        candidate
        for candidate in candidate_list
        if candidate.id in evaluated_ids
    ]
    verified = verify_candidates(evaluated, relevant_candidates)
    verified_summary = verification_metrics(verified)
    gold_by_id = {
        problem.id: normalize_reference_answer(problem.answer or "")
        for problem in labeled
    }

    selections = select_trajectories(candidate_list, min_value=min_value)
    selected_by_id = {
        selection.id: selection
        for selection in selections if selection.id in gold_by_id
    }
    selected_correct = sum(
        normalize_reference_answer(selection.predicted_answer)
        == gold_by_id[problem_id]
        for problem_id, selection in selected_by_id.items()
    )

    return {
        "labeled_problems": len(labeled),
        "problems_with_candidates": len(evaluated),
        "candidate_pass_at_k": verified_summary["problem_pass_at_k"],
        "candidate_accuracy": verified_summary["candidate_accuracy"],
        "parse_rate": verified_summary["parse_rate"],
        "selection_coverage": len(selected_by_id) / len(evaluated),
        "selected_accuracy": (
            selected_correct / len(selected_by_id) if selected_by_id else 0.0
        ),
        "selected_count": len(selected_by_id),
        "min_value": min_value,
    }
