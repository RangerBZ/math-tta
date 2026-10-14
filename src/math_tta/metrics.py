"""Evaluation metrics for candidate generation and pseudo-label selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from math_tta.answers import extract_answer, normalize_answer
from math_tta.data import Candidate, Problem
from math_tta.selection import select_trajectories


def evaluate(
    problems: Iterable[Problem],
    candidates: Iterable[Candidate],
    min_value: float = 0.5,
) -> dict[str, float | int]:
    problem_list = list(problems)
    candidate_list = list(candidates)
    problem_by_id = {problem.id: problem for problem in problem_list}
    candidates_by_id: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidate_list:
        if candidate.id not in problem_by_id:
            raise ValueError(f"candidate id {candidate.id!r} is missing from problem data")
        candidates_by_id[candidate.id].append(candidate)

    labeled = {
        problem.id: normalize_answer(problem.answer)
        for problem in problem_list
        if problem.answer is not None
    }
    if not labeled:
        raise ValueError("evaluation requires at least one problem with an 'answer'")

    evaluated_ids = [problem_id for problem_id in labeled if problem_id in candidates_by_id]
    if not evaluated_ids:
        raise ValueError("no labeled problems have generated candidates")

    any_correct = 0
    for problem_id in evaluated_ids:
        gold = labeled[problem_id]
        any_correct += int(
            any(
                (answer := extract_answer(candidate.completion)) is not None
                and normalize_answer(answer) == gold
                for candidate in candidates_by_id[problem_id]
            )
        )

    selections = select_trajectories(candidate_list, min_value=min_value)
    selected_by_id = {
        selection.id: selection
        for selection in selections
        if selection.id in labeled
    }
    selected_correct = sum(
        normalize_answer(selection.predicted_answer) == labeled[problem_id]
        for problem_id, selection in selected_by_id.items()
    )

    return {
        "labeled_problems": len(labeled),
        "problems_with_candidates": len(evaluated_ids),
        "candidate_pass_at_k": any_correct / len(evaluated_ids),
        "selection_coverage": len(selected_by_id) / len(evaluated_ids),
        "selected_accuracy": (
            selected_correct / len(selected_by_id) if selected_by_id else 0.0
        ),
        "selected_count": len(selected_by_id),
        "min_value": min_value,
    }
