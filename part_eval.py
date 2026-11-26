"""Paired evaluation of two candidate files on the same labelled problems.

Compares answered accuracy (selected and correct over all problems), selection
accuracy among answered problems, and coverage, then reports an exact McNemar
test and a paired bootstrap interval for the answered-accuracy difference.

Usage:
    python part_eval.py --problems data/gsm8k_test.jsonl \
        --baseline outputs/full-fixed6/test-candidates.jsonl \
        --treatment outputs/full-tta2/test-candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from math_tta.data import load_candidates, load_problems
from math_tta.selection import select_trajectories
from math_tta.verification import normalize_reference_answer


def _policy_metrics(
    problems: Sequence[Any],
    candidates_path: str | Path,
) -> dict[str, Any]:
    gold = {
        problem.id: normalize_reference_answer(problem.answer or "")
        for problem in problems
    }
    selected = {
        selection.id: selection.predicted_answer
        for selection in select_trajectories(load_candidates(candidates_path))
        if selection.id in gold
    }
    answered = {
        problem_id: int(selected.get(problem_id) == answer)
        for problem_id, answer in gold.items()
    }
    answered_count = len(selected)
    correct_count = sum(answered.values())
    selected_correct = sum(
        int(selected[problem_id] == answer)
        for problem_id, answer in gold.items()
        if problem_id in selected
    )
    return {
        "gold": gold,
        "answered": answered,
        "answered_count": answered_count,
        "correct_count": correct_count,
        "answered_accuracy": correct_count / len(gold),
        "selected_accuracy": (
            selected_correct / answered_count if answered_count else 0.0
        ),
        "coverage": answered_count / len(gold),
    }


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "answered_count": result["answered_count"],
        "correct_count": result["correct_count"],
        "answered_accuracy": result["answered_accuracy"],
        "selected_accuracy": result["selected_accuracy"],
        "coverage": result["coverage"],
    }


def _mcnemar(
    baseline: dict[str, int],
    treatment: dict[str, int],
) -> tuple[float, int, int]:
    baseline_only = sum(
        baseline[problem_id] == 1 and treatment[problem_id] == 0
        for problem_id in baseline
    )
    treatment_only = sum(
        baseline[problem_id] == 0 and treatment[problem_id] == 1
        for problem_id in baseline
    )
    discordant = baseline_only + treatment_only
    if discordant == 0:
        return 1.0, baseline_only, treatment_only
    tail = sum(
        math.comb(discordant, count)
        for count in range(min(baseline_only, treatment_only) + 1)
    ) / 2**discordant
    return min(1.0, 2.0 * tail), baseline_only, treatment_only


def _bootstrap_interval(
    baseline: dict[str, int],
    treatment: dict[str, int],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    generator = random.Random(seed)
    problem_ids = list(baseline)
    deltas: list[float] = []
    for _ in range(resamples):
        sample = [generator.choice(problem_ids) for _ in problem_ids]
        deltas.append(
            100.0
            * sum(treatment[problem_id] - baseline[problem_id] for problem_id in sample)
            / len(sample)
        )
    deltas.sort()
    return deltas[int(0.025 * resamples)], deltas[int(0.975 * resamples) - 1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python part_eval.py",
        description=(
            "Paired comparison of answered accuracy for two candidate files "
            "on the same labelled problems."
        ),
    )
    parser.add_argument("--problems", required=True, help="labelled problem JSONL")
    parser.add_argument(
        "--baseline",
        required=True,
        help="candidate JSONL for the baseline policy",
    )
    parser.add_argument(
        "--treatment",
        required=True,
        help="candidate JSONL for the treatment policy",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="optional JSON report path")
    arguments = parser.parse_args(argv)

    problems = load_problems(arguments.problems)
    baseline = _policy_metrics(problems, arguments.baseline)
    treatment = _policy_metrics(problems, arguments.treatment)
    if set(baseline["gold"]) != set(treatment["gold"]):
        raise ValueError("candidate files must cover the same problem ids")

    p_value, baseline_only, treatment_only = _mcnemar(
        baseline["answered"],
        treatment["answered"],
    )
    interval_low, interval_high = _bootstrap_interval(
        baseline["answered"],
        treatment["answered"],
        resamples=arguments.resamples,
        seed=arguments.seed,
    )
    report = {
        "problems": len(baseline["gold"]),
        "baseline": _summary(baseline),
        "treatment": _summary(treatment),
        "delta_answered_points": 100.0
        * (treatment["answered_accuracy"] - baseline["answered_accuracy"]),
        "delta_selected_points": 100.0
        * (treatment["selected_accuracy"] - baseline["selected_accuracy"]),
        "delta_coverage_points": 100.0
        * (treatment["coverage"] - baseline["coverage"]),
        "mcnemar": {
            "p_value": p_value,
            "baseline_only": baseline_only,
            "treatment_only": treatment_only,
        },
        "bootstrap_95_ci_answered_points": [interval_low, interval_high],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if arguments.output:
        Path(arguments.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
