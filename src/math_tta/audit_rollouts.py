"""Audit rollout answer correctness, pseudo-label stability, and GRPO rewards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Literal, Sequence

from math_tta.answers import extract_answer, normalize_answer
from math_tta.data import (
    Candidate,
    Problem,
    load_candidates,
    load_problems,
    write_jsonl,
)
from math_tta.rl_training import (
    _bayesian_calibrated_confidence,
    _tta1_group_decision,
    assign_pseudo_label_rewards,
    group_relative_advantages,
)
from math_tta.verification import normalize_reference_answer

SamplingPolicy = Literal["fixed", "tta-1", "tta-2", "tta-3"]


def _read_rollout_metadata(
    path: str | Path,
    expected_count: int,
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            metadata.append(row)
    if len(metadata) != expected_count:
        raise ValueError(
            f"{path}: found {len(metadata)} records, expected {expected_count}"
        )
    return metadata


def _optional_number(
    row: dict[str, Any],
    field: str,
    *,
    path: str | Path,
    required: bool = False,
) -> float | None:
    value = row.get(field)
    if value is None:
        if required:
            raise ValueError(f"{path}: TTA-1 rollout is missing {field!r}")
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"{path}: {field!r} must be a finite number")
    return float(value)


def load_rollout_artifact(
    path: str | Path,
    *,
    policy: SamplingPolicy,
) -> tuple[
    list[Candidate],
    list[float | None],
    list[float | None],
    list[float | None],
]:
    candidates = load_candidates(path)
    metadata = _read_rollout_metadata(path, len(candidates))
    confidence_weights: list[float | None] = []
    stored_rewards: list[float | None] = []
    stored_advantages: list[float | None] = []
    for candidate, row in zip(candidates, metadata, strict=True):
        if row.get("id") != candidate.id:
            raise ValueError(f"{path}: rollout metadata order is inconsistent")
        confidence = _optional_number(
            row,
            "token_self_certainty",
            path=path,
            required=policy in {"tta-1", "tta-2", "tta-3"},
        )
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{path}: token_self_certainty must be in [0, 1]")
        confidence_weights.append(confidence)
        stored_rewards.append(_optional_number(row, "reward", path=path))
        stored_advantages.append(_optional_number(row, "advantage", path=path))
    return candidates, confidence_weights, stored_rewards, stored_advantages


def _answer_vote(
    candidates: Sequence[Candidate],
) -> tuple[str | None, float]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        answer = extract_answer(candidate.completion)
        if answer is not None:
            counts[normalize_answer(answer)] += 1
    highest_count = max(counts.values(), default=0)
    winners = [
        answer for answer, count in counts.items() if count == highest_count
    ]
    winner = winners[0] if len(winners) == 1 else None
    return winner, highest_count / len(candidates)


def _label_transition(
    previous_label: str | None,
    current_label: str | None,
    *,
    is_initial: bool,
) -> str:
    if is_initial:
        return "initial"
    if previous_label is None and current_label is None:
        return "still_abstaining"
    if previous_label is None:
        return "newly_accepted"
    if current_label is None:
        return "became_abstention"
    if previous_label == current_label:
        return "unchanged"
    return "changed"


def _prefix_state(
    candidates: Sequence[Candidate],
    confidence_weights: Sequence[float | None],
    *,
    policy: SamplingPolicy,
    fixed_min_consensus: float,
    tta1_threshold: float,
    bayesian_confidence_weights: Sequence[float] | None = None,
) -> tuple[dict[str, Any], list[float], list[float]]:
    count_winner, consensus = _answer_vote(candidates)
    confidence_winner: str | None = None
    if policy == "fixed":
        rewards, group_records = assign_pseudo_label_rewards(
            candidates,
            min_consensus=fixed_min_consensus,
        )
        group_record = group_records[0]
        pseudo_answer_value = group_record["pseudo_answer"]
        if pseudo_answer_value is not None and not isinstance(
            pseudo_answer_value,
            str,
        ):
            raise RuntimeError("fixed pseudo-label returned an invalid answer")
        pseudo_answer = pseudo_answer_value
        accepted = bool(group_record["accepted"])
        if count_winner is None:
            decision_reason = "no_unique_count_winner"
        elif consensus < fixed_min_consensus:
            decision_reason = "below_consensus_threshold"
        else:
            decision_reason = "accepted"
    else:
        complete_confidences: list[float] = []
        for weight in confidence_weights:
            if weight is None:
                raise ValueError("TTA-1 requires confidence for every rollout")
            complete_confidences.append(weight)
        decision = _tta1_group_decision(
            candidates,
            complete_confidences,
            threshold=tta1_threshold,
        )
        decision_count_winner = decision["count_winner"]
        decision_confidence_winner = decision["confidence_winner"]
        if decision_count_winner is not None and not isinstance(
            decision_count_winner,
            str,
        ):
            raise RuntimeError("TTA-1 returned an invalid count winner")
        if decision_confidence_winner is not None and not isinstance(
            decision_confidence_winner,
            str,
        ):
            raise RuntimeError("TTA-1 returned an invalid confidence winner")
        decision_consensus = decision["consensus"]
        if isinstance(decision_consensus, bool) or not isinstance(
            decision_consensus,
            (int, float),
        ):
            raise RuntimeError("TTA-1 returned an invalid consensus")
        count_winner = decision_count_winner
        confidence_winner = decision_confidence_winner
        consensus = float(decision_consensus)
        accepted = bool(decision["accepted"])
        decision_reason = str(decision["decision_reason"])
        pseudo_answer = count_winner if accepted else None
        rewards = []
        for index, candidate in enumerate(candidates):
            answer = extract_answer(candidate.completion)
            matches_pseudo_label = (
                accepted
                and answer is not None
                and normalize_answer(answer) == count_winner
            )
            reward = float(matches_pseudo_label)
            if policy == "tta-2":
                reward *= complete_confidences[index]
            elif policy == "tta-3":
                if bayesian_confidence_weights is None:
                    raise ValueError("TTA-3 requires Bayesian confidence estimates")
                reward *= bayesian_confidence_weights[index]
            rewards.append(reward)

    advantages = group_relative_advantages(
        [candidate.id for candidate in candidates],
        rewards,
    )
    return (
        {
            "sample_count": len(candidates),
            "count_winner": count_winner,
            "confidence_weighted_winner": confidence_winner,
            "consensus": consensus,
            "pseudo_answer": pseudo_answer,
            "accepted": accepted,
            "decision_reason": decision_reason,
            "rewards": rewards,
            "advantages": advantages,
            "reward_zero_count": sum(reward == 0.0 for reward in rewards),
            "reward_one_count": sum(reward == 1.0 for reward in rewards),
            "reward_positive_count": sum(reward > 0.0 for reward in rewards),
            "mean_reward": sum(rewards) / len(rewards),
            "mixed_rewards": min(rewards) != max(rewards),
            "advantage_zero_count": sum(value == 0.0 for value in advantages),
            "advantage_negative_count": sum(value < 0.0 for value in advantages),
            "advantage_positive_count": sum(value > 0.0 for value in advantages),
        },
        rewards,
        advantages,
    )


def _check_aligned_inputs(
    problems: Sequence[Problem],
    candidates: Sequence[Candidate],
    confidence_weights: Sequence[float | None],
    stored_rewards: Sequence[float | None],
    stored_advantages: Sequence[float | None],
) -> dict[str, Problem]:
    problem_by_id = {problem.id: problem for problem in problems}
    if len(problem_by_id) != len(problems):
        raise ValueError("problem input contains duplicate ids")
    if not candidates:
        raise ValueError("rollout input contains no candidates")
    if any(
        len(values) != len(candidates)
        for values in (confidence_weights, stored_rewards, stored_advantages)
    ):
        raise ValueError("rollout metadata lengths do not match candidate count")

    for candidate in candidates:
        problem = problem_by_id.get(candidate.id)
        if problem is None:
            raise ValueError(
                f"rollout id {candidate.id!r} is absent from problem data"
            )
        if problem.answer is None:
            raise ValueError(f"problem {candidate.id!r} has no reference answer")
        if not problem.answer.strip():
            raise ValueError(f"problem {candidate.id!r} has an empty reference answer")
        if candidate.question != problem.question:
            raise ValueError(f"rollout {candidate.id!r} question differs from input")
    return problem_by_id


def _group_candidate_indices(candidates: Sequence[Candidate]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    closed_ids: set[str] = set()
    previous_id: str | None = None
    for index, candidate in enumerate(candidates):
        if candidate.id != previous_id:
            if candidate.id in closed_ids:
                raise ValueError(
                    "rollouts for each problem must be contiguous and in sample order"
                )
            if previous_id is not None:
                closed_ids.add(previous_id)
            previous_id = candidate.id
        groups.setdefault(candidate.id, []).append(index)
    return groups


def audit_rollouts(
    problems: Sequence[Problem],
    candidates: Sequence[Candidate],
    *,
    policy: SamplingPolicy,
    confidence_weights: Sequence[float | None],
    stored_rewards: Sequence[float | None],
    stored_advantages: Sequence[float | None],
    initial_samples: int = 4,
    fixed_min_consensus: float = 0.5,
    tta1_threshold: float = 0.8,
    sample_increment: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if policy not in {"fixed", "tta-1", "tta-2", "tta-3"}:
        raise ValueError("policy must be 'fixed', 'tta-1', 'tta-2', or 'tta-3'")
    if initial_samples < 1 or sample_increment < 1:
        raise ValueError("initial_samples and sample_increment must be positive")
    if not 0.0 < fixed_min_consensus <= 1.0:
        raise ValueError("fixed_min_consensus must be in (0, 1]")
    if not 0.0 < tta1_threshold <= 1.0:
        raise ValueError("tta1_threshold must be in (0, 1]")

    problem_by_id = _check_aligned_inputs(
        problems,
        candidates,
        confidence_weights,
        stored_rewards,
        stored_advantages,
    )
    indices_by_id = _group_candidate_indices(candidates)
    bayesian_confidences: list[float] | None = None
    if policy == "tta-3":
        complete_confidences = [
            float(confidence)
            for confidence in confidence_weights
            if confidence is not None
        ]
        if len(complete_confidences) != len(candidates):
            raise ValueError("TTA-3 requires confidence for every rollout")
        correctness_by_index = [
            normalize_answer(extract_answer(candidate.completion) or "")
            == normalize_reference_answer(problem_by_id[candidate.id].answer or "")
            for candidate in candidates
        ]
        bayesian_confidences = _bayesian_calibrated_confidence(
            [candidate.id for candidate in candidates],
            complete_confidences,
            correctness_by_index,
        )
    maximum_samples = max(map(len, indices_by_id.values()))
    if maximum_samples < initial_samples:
        raise ValueError("rollouts do not reach the configured initial sample count")
    if any(len(indices) < initial_samples for indices in indices_by_id.values()):
        raise ValueError("every problem must have the configured initial sample count")

    checkpoints = list(
        range(initial_samples, maximum_samples + 1, sample_increment)
    )
    if checkpoints[-1] != maximum_samples:
        checkpoints.append(maximum_samples)

    details: list[dict[str, Any]] = []
    states_by_checkpoint: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for problem_id, indices in indices_by_id.items():
        problem = problem_by_id[problem_id]
        group_candidates = [candidates[index] for index in indices]
        group_confidences = [confidence_weights[index] for index in indices]
        gold_answer = normalize_reference_answer(problem.answer or "")
        rollout_answers = [
            extract_answer(candidate.completion) for candidate in group_candidates
        ]
        normalized_answers = [
            normalize_answer(answer) if answer is not None else None
            for answer in rollout_answers
        ]
        rollout_records = [
            {
                "sample_index": sample_index,
                "predicted_answer": rollout_answers[sample_index - 1],
                "normalized_answer": normalized_answers[sample_index - 1],
                "correct_vs_reference": (
                    normalized_answers[sample_index - 1] == gold_answer
                ),
                "token_self_certainty": group_confidences[sample_index - 1],
                "bayesian_correctness_probability": (
                    bayesian_confidences[indices[sample_index - 1]]
                    if bayesian_confidences is not None
                    else None
                ),
                "stored_reward": stored_rewards[indices[sample_index - 1]],
                "stored_advantage": stored_advantages[indices[sample_index - 1]],
            }
            for sample_index in range(1, len(group_candidates) + 1)
        ]

        prefix_states: list[dict[str, Any]] = []
        previous_state: dict[str, Any] | None = None
        previous_count = 0
        for sample_count in checkpoints:
            if sample_count > len(group_candidates):
                continue
            prefix_candidates = group_candidates[:sample_count]
            state, rewards, advantages = _prefix_state(
                prefix_candidates,
                group_confidences[:sample_count],
                policy=policy,
                fixed_min_consensus=fixed_min_consensus,
                tta1_threshold=tta1_threshold,
                bayesian_confidence_weights=(
                    [
                        bayesian_confidences[index]
                        for index in indices[:sample_count]
                    ]
                    if bayesian_confidences is not None
                    else None
                ),
            )
            pseudo_answer = state["pseudo_answer"]
            state["pseudo_label_correct_vs_reference"] = (
                pseudo_answer == gold_answer if state["accepted"] else None
            )
            state["gold_answer"] = gold_answer
            state["rollout_correct_count"] = sum(
                normalized_answers[index] == gold_answer
                for index in range(sample_count)
            )
            state["label_transition"] = _label_transition(
                previous_state["pseudo_answer"] if previous_state else None,
                pseudo_answer,
                is_initial=previous_state is None,
            )
            state["count_winner_changed"] = (
                previous_state is not None
                and previous_state["count_winner"] != state["count_winner"]
            )
            state["previous_reward_flips"] = (
                sum(
                    old_reward != new_reward
                    for old_reward, new_reward in zip(
                        previous_state["rewards"],
                        rewards[:previous_count],
                        strict=True,
                    )
                )
                if previous_state is not None
                else 0
            )
            state["new_rollouts"] = rollout_records[previous_count:sample_count]
            state["new_rollout_count"] = sample_count - previous_count
            state["new_rollout_correct_count"] = sum(
                item["correct_vs_reference"]
                for item in state["new_rollouts"]
            )
            state["new_rollout_label_match_count"] = (
                sum(
                    item["normalized_answer"] == pseudo_answer
                    for item in state["new_rollouts"]
                )
                if state["accepted"]
                else 0
            )
            state["is_final_prefix"] = sample_count == len(group_candidates)
            if state["is_final_prefix"]:
                reward_mismatches = 0
                advantage_mismatches = 0
                for offset, (recomputed_reward, recomputed_advantage) in enumerate(
                    zip(rewards, advantages, strict=True)
                ):
                    source_index = indices[offset]
                    recorded_reward = stored_rewards[source_index]
                    recorded_advantage = stored_advantages[source_index]
                    if recorded_reward is not None and not isclose(
                        recomputed_reward,
                        recorded_reward,
                        rel_tol=1e-6,
                        abs_tol=1e-8,
                    ):
                        reward_mismatches += 1
                    if recorded_advantage is not None and not isclose(
                        recomputed_advantage,
                        recorded_advantage,
                        rel_tol=1e-6,
                        abs_tol=1e-8,
                    ):
                        advantage_mismatches += 1
                state["stored_reward_mismatches"] = reward_mismatches
                state["stored_advantage_mismatches"] = advantage_mismatches
            else:
                state["stored_reward_mismatches"] = None
                state["stored_advantage_mismatches"] = None
            prefix_states.append(state)
            states_by_checkpoint[sample_count].append(state)
            previous_state = state
            previous_count = sample_count

        details.append(
            {
                "id": problem_id,
                "question": problem.question,
                "gold_final_answer": gold_answer,
                "rollouts": rollout_records,
                "prefixes": prefix_states,
            }
        )

    summaries: list[dict[str, Any]] = []
    for checkpoint, states in sorted(states_by_checkpoint.items()):
        cumulative_states = []
        for detail in details:
            eligible_states = [
                state
                for state in detail["prefixes"]
                if state["sample_count"] <= checkpoint
            ]
            if not eligible_states:
                raise RuntimeError(
                    f"problem {detail['id']!r} has no state at prefix {checkpoint}"
                )
            cumulative_states.append(eligible_states[-1])
        summaries.append(
            _summarize_checkpoint(
                policy,
                checkpoint,
                states,
                cumulative_states,
            )
        )
    return details, summaries


def _summarize_checkpoint(
    policy: SamplingPolicy,
    sample_count: int,
    states: Sequence[dict[str, Any]],
    cumulative_states: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [state for state in states if state["accepted"]]
    accepted_correct = sum(
        state["pseudo_label_correct_vs_reference"] is True
        for state in accepted
    )
    rollout_count = sum(state["sample_count"] for state in states)
    correct_rollouts = sum(
        state["rollout_correct_count"]
        for state in states
    )
    added_count = sum(state["new_rollout_count"] for state in states)
    added_correct = sum(state["new_rollout_correct_count"] for state in states)
    rewards = [
        reward
        for state in states
        for reward in state["rewards"]
    ]
    final_states = [state for state in states if state["is_final_prefix"]]
    changed_transitions = {"changed", "became_abstention", "newly_accepted"}
    cumulative_accepted = [
        state for state in cumulative_states if state["accepted"]
    ]
    cumulative_rewards = [
        reward
        for state in cumulative_states
        for reward in state["rewards"]
    ]
    cumulative_rollout_count = sum(
        state["sample_count"] for state in cumulative_states
    )
    cumulative_correct_rollouts = sum(
        state["rollout_correct_count"] for state in cumulative_states
    )
    return {
        "policy": policy,
        "sample_count": sample_count,
        "prefix_cohort": "groups with at least this many rollouts",
        "groups_with_prefix": len(states),
        "accepted_groups": len(accepted),
        "pseudo_label_coverage": len(accepted) / len(states) if states else 0.0,
        "correct_accepted_pseudo_labels": accepted_correct,
        "accepted_pseudo_label_accuracy": (
            accepted_correct / len(accepted) if accepted else None
        ),
        "rollout_count": rollout_count,
        "correct_rollouts": correct_rollouts,
        "rollout_accuracy": correct_rollouts / rollout_count if rollout_count else 0.0,
        "new_rollout_count": added_count,
        "new_rollout_correct_count": added_correct,
        "new_rollout_accuracy": added_correct / added_count if added_count else None,
        "reward_zero_count": sum(reward == 0.0 for reward in rewards),
        "reward_one_count": sum(reward == 1.0 for reward in rewards),
        "reward_positive_count": sum(reward > 0.0 for reward in rewards),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "mixed_reward_groups": sum(state["mixed_rewards"] for state in states),
        "zero_advantage_groups": sum(
            state["advantage_zero_count"] == state["sample_count"]
            for state in states
        ),
        "zero_advantage_rollouts": sum(
            state["advantage_zero_count"] for state in states
        ),
        "nonzero_advantage_rollouts": sum(
            state["advantage_negative_count"] + state["advantage_positive_count"]
            for state in states
        ),
        "groups_with_label_change": sum(
            state["label_transition"] in changed_transitions
            for state in states
        ),
        "groups_with_pseudo_label_stable": sum(
            state["label_transition"] == "unchanged" for state in states
        ),
        "groups_with_count_winner_change": sum(
            state["count_winner_changed"] for state in states
        ),
        "previous_reward_flips": sum(
            state["previous_reward_flips"] for state in states
        ),
        "final_groups_checked_against_stored_rewards": len(final_states),
        "stored_reward_mismatches": sum(
            state["stored_reward_mismatches"] for state in final_states
        ),
        "stored_advantage_mismatches": sum(
            state["stored_advantage_mismatches"] for state in final_states
        ),
        "cumulative": {
            "groups": len(cumulative_states),
            "samples_used": cumulative_rollout_count,
            "accepted_groups": len(cumulative_accepted),
            "pseudo_label_coverage": (
                len(cumulative_accepted) / len(cumulative_states)
                if cumulative_states
                else 0.0
            ),
            "accepted_pseudo_label_accuracy": (
                sum(
                    state["pseudo_label_correct_vs_reference"] is True
                    for state in cumulative_accepted
                )
                / len(cumulative_accepted)
                if cumulative_accepted
                else None
            ),
            "correct_rollouts": cumulative_correct_rollouts,
            "rollout_accuracy": (
                cumulative_correct_rollouts / cumulative_rollout_count
                if cumulative_rollout_count
                else 0.0
            ),
            "reward_zero_count": sum(
                reward == 0.0 for reward in cumulative_rewards
            ),
            "reward_one_count": sum(
                reward == 1.0 for reward in cumulative_rewards
            ),
            "reward_positive_count": sum(
                reward > 0.0 for reward in cumulative_rewards
            ),
            "mean_reward": (
                sum(cumulative_rewards) / len(cumulative_rewards)
                if cumulative_rewards
                else 0.0
            ),
            "mixed_reward_groups": sum(
                state["mixed_rewards"] for state in cumulative_states
            ),
            "zero_advantage_groups": sum(
                state["advantage_zero_count"] == state["sample_count"]
                for state in cumulative_states
            ),
            "zero_advantage_rollouts": sum(
                state["advantage_zero_count"] for state in cumulative_states
            ),
            "nonzero_advantage_rollouts": sum(
                state["advantage_negative_count"]
                + state["advantage_positive_count"]
                for state in cumulative_states
            ),
        },
    }


def _confidence_correctness_diagnostics(
    details: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    samples = [
        rollout
        for detail in details
        for rollout in detail["rollouts"]
        if rollout["token_self_certainty"] is not None
    ]
    if not samples:
        return {"available": False, "bins": []}

    bins = []
    for bin_index in range(5):
        lower = bin_index / 5
        upper = (bin_index + 1) / 5
        members = [
            rollout
            for rollout in samples
            if lower <= rollout["token_self_certainty"]
            and (
                rollout["token_self_certainty"] <= upper
                if bin_index == 4
                else rollout["token_self_certainty"] < upper
            )
        ]
        correct_count = sum(
            rollout["correct_vs_reference"] for rollout in members
        )
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "correct_count": correct_count,
                "accuracy": correct_count / len(members) if members else None,
                "mean_confidence": (
                    sum(item["token_self_certainty"] for item in members)
                    / len(members)
                    if members
                    else None
                ),
            }
        )

    correct_samples = [
        item for item in samples if item["correct_vs_reference"]
    ]
    incorrect_samples = [
        item for item in samples if not item["correct_vs_reference"]
    ]
    return {
        "available": True,
        "sample_count": len(samples),
        "mean_confidence_correct": (
            sum(item["token_self_certainty"] for item in correct_samples)
            / len(correct_samples)
            if correct_samples
            else None
        ),
        "mean_confidence_incorrect": (
            sum(item["token_self_certainty"] for item in incorrect_samples)
            / len(incorrect_samples)
            if incorrect_samples
            else None
        ),
        "bins": bins,
    }


def _bayesian_calibration_diagnostics(
    details: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    samples = [
        rollout
        for detail in details
        for rollout in detail["rollouts"]
        if rollout["bayesian_correctness_probability"] is not None
    ]
    if not samples:
        return {"available": False}
    return {
        "available": True,
        "sample_count": len(samples),
        "mean_predicted_correctness": sum(
            item["bayesian_correctness_probability"] for item in samples
        )
        / len(samples),
        "observed_accuracy": sum(
            item["correct_vs_reference"] for item in samples
        )
        / len(samples),
        "brier_score": sum(
            (
                item["bayesian_correctness_probability"]
                - float(item["correct_vs_reference"])
            )
            ** 2
            for item in samples
        )
        / len(samples),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m math_tta.audit_rollouts",
        description=(
            "Compare fixed-6 and adaptive rollout correctness, pseudo-label "
            "stability, and recomputed GRPO reward/advantage distributions."
        ),
    )
    parser.add_argument("--problems", required=True, help="labeled training JSONL")
    parser.add_argument(
        "--fixed6-rollouts",
        required=True,
        help="fixed-sampling rollout JSONL",
    )
    parser.add_argument(
        "--tta1-rollouts",
        required=True,
        help="raw TTA-1 rollout JSONL",
    )
    parser.add_argument(
        "--tta2-rollouts",
        help=(
            "optional raw TTA-2 rollout JSONL; omit to simulate TTA-2 rewards "
            "on the TTA-1 rollouts"
        ),
    )
    parser.add_argument(
        "--tta3-rollouts",
        help=(
            "optional raw TTA-3 rollout JSONL; omit to simulate TTA-3 rewards "
            "on TTA-2 rollouts, or TTA-1 rollouts when TTA-2 is omitted"
        ),
    )
    parser.add_argument("--initial-samples", type=int, default=4)
    parser.add_argument("--sample-increment", type=int, default=2)
    parser.add_argument("--fixed-min-consensus", type=float, default=0.5)
    parser.add_argument("--tta1-threshold", type=float, default=0.8)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        problems = load_problems(arguments.problems)
        if any(problem.id.startswith("test-") for problem in problems):
            raise ValueError(
                "rollout auditing must use training data, not GSM8K test data"
            )
        runs: list[tuple[str, SamplingPolicy, str]] = [
            ("fixed6", "fixed", arguments.fixed6_rollouts),
            ("tta1", "tta-1", arguments.tta1_rollouts),
        ]
        run_details: dict[str, list[dict[str, Any]]] = {}
        run_summaries: dict[str, list[dict[str, Any]]] = {}
        run_id_sets: dict[str, set[str]] = {}
        for run_name, policy, path in runs:
            candidates, confidences, stored_rewards, stored_advantages = (
                load_rollout_artifact(path, policy=policy)
            )
            details, summaries = audit_rollouts(
                problems,
                candidates,
                policy=policy,
                confidence_weights=confidences,
                stored_rewards=stored_rewards,
                stored_advantages=stored_advantages,
                initial_samples=arguments.initial_samples,
                fixed_min_consensus=arguments.fixed_min_consensus,
                tta1_threshold=arguments.tta1_threshold,
                sample_increment=arguments.sample_increment,
            )
            run_details[run_name] = details
            run_summaries[run_name] = summaries
            run_id_sets[run_name] = {str(record["id"]) for record in details}
        tta2_path = arguments.tta2_rollouts or arguments.tta1_rollouts
        tta2_candidates, tta2_confidences, tta2_rewards, tta2_advantages = (
            load_rollout_artifact(tta2_path, policy="tta-2")
        )
        if arguments.tta2_rollouts is None:
            tta2_rewards = [None] * len(tta2_candidates)
            tta2_advantages = [None] * len(tta2_candidates)
        tta2_details, tta2_summaries = audit_rollouts(
            problems,
            tta2_candidates,
            policy="tta-2",
            confidence_weights=tta2_confidences,
            stored_rewards=tta2_rewards,
            stored_advantages=tta2_advantages,
            initial_samples=arguments.initial_samples,
            fixed_min_consensus=arguments.fixed_min_consensus,
            tta1_threshold=arguments.tta1_threshold,
            sample_increment=arguments.sample_increment,
        )
        run_details["tta2"] = tta2_details
        run_summaries["tta2"] = tta2_summaries
        run_id_sets["tta2"] = {str(record["id"]) for record in tta2_details}
        tta3_path = (
            arguments.tta3_rollouts
            or arguments.tta2_rollouts
            or arguments.tta1_rollouts
        )
        tta3_candidates, tta3_confidences, tta3_rewards, tta3_advantages = (
            load_rollout_artifact(tta3_path, policy="tta-3")
        )
        if arguments.tta3_rollouts is None:
            tta3_rewards = [None] * len(tta3_candidates)
            tta3_advantages = [None] * len(tta3_candidates)
        tta3_details, tta3_summaries = audit_rollouts(
            problems,
            tta3_candidates,
            policy="tta-3",
            confidence_weights=tta3_confidences,
            stored_rewards=tta3_rewards,
            stored_advantages=tta3_advantages,
            initial_samples=arguments.initial_samples,
            fixed_min_consensus=arguments.fixed_min_consensus,
            tta1_threshold=arguments.tta1_threshold,
            sample_increment=arguments.sample_increment,
        )
        run_details["tta3"] = tta3_details
        run_summaries["tta3"] = tta3_summaries
        run_id_sets["tta3"] = {str(record["id"]) for record in tta3_details}
        if run_id_sets["fixed6"] != run_id_sets["tta1"]:
            raise ValueError(
                "fixed-6 and TTA-1 rollout files must contain the same problem IDs"
            )
        if run_id_sets["tta1"] != run_id_sets["tta2"]:
            raise ValueError(
                "TTA-1 and TTA-2 rollout files must contain the same problem IDs"
            )
        if run_id_sets["tta2"] != run_id_sets["tta3"]:
            raise ValueError(
                "TTA-2 and TTA-3 rollout files must contain the same problem IDs"
            )

        output_dir = Path(arguments.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for run_name, details in run_details.items():
            write_jsonl(output_dir / f"{run_name}-question-audit.jsonl", details)
        summary = {
            "problems": len(run_id_sets["fixed6"]),
            "settings": {
                "initial_samples": arguments.initial_samples,
                "sample_increment": arguments.sample_increment,
                "fixed_min_consensus": arguments.fixed_min_consensus,
                "tta1_threshold": arguments.tta1_threshold,
            },
            "runs": run_summaries,
            "confidence_correctness": {
                run_name: _confidence_correctness_diagnostics(details)
                for run_name, details in run_details.items()
            },
            "bayesian_reward_calibration": _bayesian_calibration_diagnostics(
                run_details["tta3"]
            ),
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as target:
            json.dump(summary, target, ensure_ascii=False, indent=2)
            target.write("\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Per-question audit written to {output_dir}")
        return 0
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
