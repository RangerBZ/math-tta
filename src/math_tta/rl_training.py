"""Outcome-reward GRPO training on on-policy language-model rollouts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from collections import Counter, defaultdict
from contextlib import nullcontext
from itertools import islice
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Literal, Sequence

from math_tta.answers import extract_answer, normalize_answer
from math_tta.data import Candidate, Problem, Rollout, write_jsonl
from math_tta.generation import generate_candidate_batch, set_seed
from math_tta.model_training import (
    TuneMode,
    load_trainable_model,
    save_training_checkpoint,
)
from math_tta.progress import load_tqdm
from math_tta.verification import verify_candidates, verification_metrics

RewardMode = Literal["verifier", "pseudo-label"]


def _partition_problems(
    problems: Sequence[Problem],
    worker_count: int,
) -> list[list[tuple[int, Problem]]]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    shards: list[list[tuple[int, Problem]]] = [
        [] for _ in range(min(worker_count, len(problems)))
    ]
    for index, problem in enumerate(problems):
        shards[index % len(shards)].append((index, problem))
    return shards


def _generate_problem_shard(
    indexed_problems: Sequence[tuple[int, Problem]],
    model: Any,
    tokenizer: Any,
    *,
    samples: int,
    max_new_tokens: int,
    generation_batch_size: int,
    progress_bar: Any,
) -> list[tuple[int, list[Candidate]]]:
    problems = [problem for _, problem in indexed_problems]
    generated = iter(
        generate_candidate_batch(
            problems,
            model,
            tokenizer,
            samples=samples,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            batch_size=generation_batch_size,
        )
    )
    indexed_groups: list[tuple[int, list[Candidate]]] = []
    for index, problem in indexed_problems:
        candidates = list(islice(generated, samples))
        if len(candidates) != samples:
            raise RuntimeError(
                f"generation returned fewer than {samples} samples "
                f"for problem {problem.id!r}"
            )
        indexed_groups.append((index, candidates))
        progress_bar.update(1)
    return indexed_groups


def group_relative_advantages(
    group_ids: Sequence[str],
    rewards: Sequence[float],
) -> list[float]:
    if len(group_ids) != len(rewards):
        raise ValueError("group_ids and rewards must have the same length")
    if any(not isfinite(reward) for reward in rewards):
        raise ValueError("rewards must be finite")

    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        indices_by_group[group_id].append(index)

    advantages = [0.0] * len(rewards)
    for indices in indices_by_group.values():
        group_rewards = [rewards[index] for index in indices]
        mean = sum(group_rewards) / len(group_rewards)
        variance = sum((reward - mean) ** 2 for reward in group_rewards) / len(
            group_rewards
        )
        standard_deviation = sqrt(variance)
        if standard_deviation == 0:
            continue
        for index in indices:
            advantages[index] = (rewards[index] - mean) / standard_deviation
    return advantages


def assign_pseudo_label_rewards(
    candidates: Sequence[Candidate],
    *,
    min_consensus: float = 0.5,
) -> tuple[list[float], list[dict[str, object]]]:
    if not candidates:
        raise ValueError("pseudo-label rewards require at least one candidate")
    if not 0.0 < min_consensus <= 1.0:
        raise ValueError("min_consensus must be in the interval (0, 1]")

    candidates_by_problem: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        candidates_by_problem[candidate.id].append(index)

    rewards = [0.0] * len(candidates)
    group_records: list[dict[str, object]] = []
    for problem_id, indices in candidates_by_problem.items():
        questions = {candidates[index].question for index in indices}
        if len(questions) != 1:
            raise ValueError(
                f"candidates for problem {problem_id!r} have inconsistent questions"
            )

        normalized_answers: dict[int, str | None] = {}
        answer_counts: Counter[str] = Counter()
        for index in indices:
            extracted = extract_answer(candidates[index].completion)
            normalized = normalize_answer(extracted) if extracted is not None else None
            normalized_answers[index] = normalized
            if normalized is not None:
                answer_counts[normalized] += 1

        highest_count = max(answer_counts.values(), default=0)
        winners = [
            answer
            for answer, count in answer_counts.items()
            if count == highest_count
        ]
        consensus = highest_count / len(indices)
        has_pseudo_label = (
            len(winners) == 1
            and consensus >= min_consensus
        )
        pseudo_answer = winners[0] if has_pseudo_label else None

        if pseudo_answer is not None:
            for index in indices:
                rewards[index] = float(
                    normalized_answers[index] == pseudo_answer
                )

        group_records.append(
            {
                "id": problem_id,
                "question": candidates[indices[0]].question,
                "pseudo_answer": pseudo_answer,
                "consensus": consensus,
                "sample_count": len(indices),
                "parseable_samples": sum(
                    normalized_answers[index] is not None for index in indices
                ),
                "accepted": has_pseudo_label,
            }
        )
    return rewards, group_records


def _pseudo_label_metrics(
    candidates: Sequence[Candidate],
    rewards: Sequence[float],
    group_records: Sequence[dict[str, object]],
) -> dict[str, float | int]:
    accepted_groups = [record for record in group_records if record["accepted"]]
    parseable_samples = sum(
        extract_answer(candidate.completion) is not None
        for candidate in candidates
    )
    return {
        "problems": len(group_records),
        "candidates": len(candidates),
        "parseable_candidates": parseable_samples,
        "parse_rate": parseable_samples / len(candidates),
        "pseudo_labeled_groups": len(accepted_groups),
        "abstained_groups": len(group_records) - len(accepted_groups),
        "pseudo_label_coverage": len(accepted_groups) / len(group_records),
        "mean_accepted_consensus": (
            sum(float(record["consensus"]) for record in accepted_groups)
            / len(accepted_groups)
            if accepted_groups
            else 0.0
        ),
        "pseudo_label_reward_rate": sum(rewards) / len(rewards),
    }


def _collate_candidates(
    candidates: Sequence[Candidate],
    *,
    pad_token_id: int,
    device: Any,
    torch: Any,
    max_length: int,
) -> dict[str, Any]:
    sequences: list[list[int]] = []
    response_masks: list[list[float]] = []
    for candidate in candidates:
        if candidate.prompt_token_ids is None:
            raise ValueError(
                f"rollout {candidate.id!r} is missing prompt_token_ids; "
                "regenerate it with this version"
            )
        if candidate.completion_token_ids is None:
            raise ValueError(
                f"rollout {candidate.id!r} is missing completion_token_ids; "
                "regenerate it with this version"
            )
        if not candidate.prompt_token_ids or not candidate.completion_token_ids:
            raise ValueError(f"rollout {candidate.id!r} has an empty token sequence")

        prefix = list(candidate.prompt_token_ids)
        completion = list(candidate.completion_token_ids)
        sequence = prefix + completion
        if len(sequence) > max_length:
            raise ValueError(
                f"rollout {candidate.id!r} has {len(sequence)} tokens, exceeding "
                f"--max-length={max_length}"
            )
        sequences.append(sequence)
        response_masks.append(
            [0.0] * (len(prefix) - 1) + [1.0] * len(completion)
        )

    width = max(len(sequence) - 1 for sequence in sequences)
    input_rows = [sequence[:-1] for sequence in sequences]
    target_rows = [sequence[1:] for sequence in sequences]
    input_ids = [
        row + [pad_token_id] * (width - len(row))
        for row in input_rows
    ]
    target_ids = [
        row + [pad_token_id] * (width - len(row))
        for row in target_rows
    ]
    attention_masks = [
        [1] * len(row) + [0] * (width - len(row))
        for row in input_rows
    ]
    padded_response_masks = [
        row + [0.0] * (width - len(row))
        for row in response_masks
    ]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            attention_masks,
            dtype=torch.long,
            device=device,
        ),
        "target_ids": torch.tensor(target_ids, dtype=torch.long, device=device),
        "response_mask": torch.tensor(
            padded_response_masks,
            dtype=torch.float32,
            device=device,
        ),
    }


def _selected_token_logprobs(
    model: Any,
    batch: dict[str, Any],
    torch: Any,
) -> Any:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    log_probs = torch.nn.functional.log_softmax(output.logits.float(), dim=-1)
    return log_probs.gather(
        -1,
        batch["target_ids"].unsqueeze(-1),
    ).squeeze(-1)


def _score_old_policy(
    candidates: Sequence[Candidate],
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    *,
    label: str,
    batch_size: int,
    max_length: int,
    progress: Any,
) -> list[tuple[float, ...]]:
    scores: list[tuple[float, ...]] = []
    model.eval()
    with torch.inference_mode():
        batches = range(0, len(candidates), batch_size)
        for start in progress(
            batches,
            desc=label,
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
        ):
            batch_candidates = candidates[start : start + batch_size]
            batch = _collate_candidates(
                batch_candidates,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
                torch=torch,
                max_length=max_length,
            )
            token_logprobs = _selected_token_logprobs(model, batch, torch)
            for row, mask in zip(
                token_logprobs,
                batch["response_mask"],
                strict=True,
            ):
                scores.append(
                    tuple(
                        float(value)
                        for value in row[mask.bool()].cpu().tolist()
                    )
                )
    return scores


def _collate_rollouts(
    rollouts: Sequence[Rollout],
    *,
    pad_token_id: int,
    device: Any,
    torch: Any,
    max_length: int,
) -> dict[str, Any]:
    batch = _collate_candidates(
        [rollout.candidate for rollout in rollouts],
        pad_token_id=pad_token_id,
        device=device,
        torch=torch,
        max_length=max_length,
    )
    width = batch["response_mask"].shape[1]
    old_logprob_rows = _padded_old_logprob_rows(rollouts, width)
    batch["old_logprobs"] = torch.tensor(
        old_logprob_rows,
        dtype=torch.float32,
        device=device,
    )
    batch["advantages"] = torch.tensor(
        [rollout.advantage for rollout in rollouts],
        dtype=torch.float32,
        device=device,
    )
    return batch


def _padded_old_logprob_rows(
    rollouts: Sequence[Rollout],
    width: int,
) -> list[list[float]]:
    old_logprob_rows: list[list[float]] = []
    for rollout in rollouts:
        prompt_token_ids = rollout.candidate.prompt_token_ids
        completion_token_ids = rollout.candidate.completion_token_ids
        if prompt_token_ids is None or completion_token_ids is None:
            raise ValueError(
                f"rollout {rollout.candidate.id!r} is missing prompt or "
                "completion token ids"
            )
        if len(rollout.old_policy_logprobs) != len(completion_token_ids):
            raise ValueError(
                f"rollout {rollout.candidate.id!r} has mismatched old-policy "
                "log probabilities and completion tokens"
            )
        row = [0.0] * (len(prompt_token_ids) - 1)
        row.extend(rollout.old_policy_logprobs)
        if len(row) > width:
            raise ValueError(
                f"rollout {rollout.candidate.id!r} exceeds the padded token width"
            )
        row.extend([0.0] * (width - len(row)))
        old_logprob_rows.append(row)
    return old_logprob_rows


def _train_grpo_rollouts(
    rollouts: Sequence[Rollout],
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    torch: Any,
    device: Any,
    *,
    epochs: int,
    batch_size: int,
    clip_epsilon: float,
    max_grad_norm: float,
    max_length: int,
    seed: int,
    progress: Any,
) -> dict[str, float | int]:
    trainable_rollouts = [rollout for rollout in rollouts if rollout.advantage != 0]
    if not trainable_rollouts:
        return {
            "grpo_loss": 0.0,
            "optimizer_steps": 0,
            "trained_rollouts": 0,
            "clip_fraction": 0.0,
        }

    model.config.use_cache = False
    model.eval()
    generator = torch.Generator()
    generator.manual_seed(seed)
    losses: list[float] = []
    clipped_fractions: list[float] = []
    optimizer_steps = 0
    for epoch_index in range(1, epochs + 1):
        order = torch.randperm(
            len(trainable_rollouts),
            generator=generator,
        ).tolist()
        batches = range(0, len(order), batch_size)
        for start in progress(
            batches,
            desc=f"GRPO epoch {epoch_index}/{epochs}",
            unit="step",
            dynamic_ncols=True,
            mininterval=1.0,
        ):
            batch_rollouts = [
                trainable_rollouts[index]
                for index in order[start : start + batch_size]
            ]
            batch = _collate_rollouts(
                batch_rollouts,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
                torch=torch,
                max_length=max_length,
            )
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else nullcontext()
            )
            with autocast:
                new_logprobs = _selected_token_logprobs(model, batch, torch)
                log_ratio = new_logprobs - batch["old_logprobs"]
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - clip_epsilon,
                    1.0 + clip_epsilon,
                )
                advantages = batch["advantages"].unsqueeze(1)
                surrogate = torch.minimum(
                    ratio * advantages,
                    clipped_ratio * advantages,
                )
                response_mask = batch["response_mask"]
                token_counts = response_mask.sum(dim=1).clamp_min(1.0)
                loss = -(
                    (surrogate * response_mask).sum(dim=1) / token_counts
                ).mean()

            loss.backward()
            trainable_parameters = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach().float().cpu()))
            clipped_fractions.append(
                float(
                    (
                        (
                            (ratio < 1.0 - clip_epsilon)
                            | (ratio > 1.0 + clip_epsilon)
                        )
                        * response_mask.bool()
                    ).sum().float().cpu()
                    / response_mask.sum().clamp_min(1.0).cpu()
                )
            )

    model.config.use_cache = True
    return {
        "grpo_loss": sum(losses) / len(losses),
        "optimizer_steps": optimizer_steps,
        "trained_rollouts": len(trainable_rollouts),
        "clip_fraction": sum(clipped_fractions) / len(clipped_fractions),
    }


def _rollout_record(
    rollout: Rollout,
    *,
    reward_mode: RewardMode,
    pseudo_label_record: dict[str, object] | None = None,
) -> dict[str, object]:
    candidate = rollout.candidate
    record: dict[str, object] = {
        "id": candidate.id,
        "question": candidate.question,
        "completion": candidate.completion,
        "prompt_token_ids": list(candidate.prompt_token_ids or ()),
        "completion_token_ids": list(candidate.completion_token_ids or ()),
        "reward": rollout.reward,
        "advantage": rollout.advantage,
        "old_policy_logprobs": list(rollout.old_policy_logprobs),
        "reward_source": reward_mode,
    }
    if pseudo_label_record is not None:
        record.update(
            {
                "pseudo_answer": pseudo_label_record["pseudo_answer"],
                "pseudo_label_consensus": pseudo_label_record["consensus"],
                "pseudo_label_accepted": pseudo_label_record["accepted"],
            }
        )
    return record


def run_grpo_training(
    problems: Sequence[Problem],
    *,
    model_id: str,
    output_dir: str | Path,
    samples: int = 4,
    generation_batch_size: int = 1,
    sampling_devices: int = 1,
    max_new_tokens: int = 512,
    rounds: int = 1,
    epochs: int = 1,
    batch_size: int = 1,
    max_length: int = 2048,
    learning_rate: float = 1e-6,
    clip_epsilon: float = 0.2,
    max_grad_norm: float = 1.0,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    tune_mode: TuneMode = "lora",
    reward_mode: RewardMode = "verifier",
    min_consensus: float = 0.5,
    seed: int = 42,
) -> list[dict[str, object]]:
    if not problems:
        raise ValueError("GRPO training requires at least one problem")
    if len({problem.id for problem in problems}) != len(problems):
        raise ValueError("training problem data contains duplicate ids")
    if any(problem.id.startswith("test-") for problem in problems):
        raise ValueError(
            "training input contains GSM8K test-split ids; use the training split only"
        )
    if reward_mode not in {"verifier", "pseudo-label"}:
        raise ValueError("reward_mode must be 'verifier' or 'pseudo-label'")
    if reward_mode == "verifier" and any(
        problem.answer is None for problem in problems
    ):
        raise ValueError("every training problem must include a reference answer")
    if not 0.0 < min_consensus <= 1.0:
        raise ValueError("min_consensus must be in the interval (0, 1]")
    if samples < 2:
        raise ValueError("GRPO requires at least two samples per problem")
    if min(
        generation_batch_size,
        sampling_devices,
        rounds,
        epochs,
        batch_size,
        max_length,
    ) < 1:
        raise ValueError(
            "batch sizes, sampling_devices, rounds, epochs, and max_length "
            "must be positive"
        )
    if learning_rate <= 0 or lora_rank < 1 or lora_alpha < 1:
        raise ValueError("learning_rate, lora_rank, and lora_alpha must be positive")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in the interval [0, 1)")
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in the interval (0, 1)")
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if tune_mode not in {"lora", "full"}:
        raise ValueError("tune_mode must be 'lora' or 'full'")

    set_seed(seed)
    progress = load_tqdm()
    torch, tokenizer, model, device = load_trainable_model(
        model_id,
        tune_mode=tune_mode,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    if sampling_devices > 1:
        if device.type != "cuda":
            raise ValueError("--sampling-devices greater than 1 requires CUDA")
        available_devices = torch.cuda.device_count()
        if sampling_devices > available_devices:
            raise ValueError(
                f"--sampling-devices={sampling_devices} requested, but only "
                f"{available_devices} CUDA devices are visible"
            )

    primary_device = next(model.parameters()).device
    primary_index = (
        primary_device.index
        if primary_device.index is not None
        else torch.cuda.current_device()
        if primary_device.type == "cuda"
        else None
    )
    active_sampling_devices = min(sampling_devices, len(problems))
    visible_indices = (
        list(range(torch.cuda.device_count()))
        if device.type == "cuda"
        else []
    )
    sampling_device_indices = (
        [primary_index]
        + [
            index
            for index in visible_indices
            if index != primary_index
        ][: active_sampling_devices - 1]
        if device.type == "cuda"
        else [None]
    )
    sampling_models = [model]
    for device_index in sampling_device_indices[1:]:
        sampling_device = torch.device(f"cuda:{device_index}")
        print(
            f"Creating rollout model replica on {sampling_device}...",
            flush=True,
        )
        replica = deepcopy(model).to(sampling_device)
        replica.requires_grad_(False)
        replica.eval()
        sampling_models.append(replica)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    for round_index in range(1, rounds + 1):
        round_name = f"round-{round_index:02d}"
        set_seed(seed + round_index)
        model.eval()
        for replica in sampling_models[1:]:
            replica.load_state_dict(model.state_dict())
            replica.eval()
        print(
            f"{round_name}: sampling {samples} rollouts for "
            f"{len(problems)} problems across {len(sampling_models)} "
            "sampling replica(s)...",
            flush=True,
        )
        problem_shards = _partition_problems(
            problems,
            len(sampling_models),
        )
        grouped_candidates: list[list[Candidate] | None] = [
            None
        ] * len(problems)
        with progress(
            total=len(problems),
            desc=f"{round_name} sampling",
            unit="problem",
            dynamic_ncols=True,
            mininterval=1.0,
        ) as sampling_progress:
            if len(problem_shards) == 1:
                indexed_groups = _generate_problem_shard(
                    problem_shards[0],
                    sampling_models[0],
                    tokenizer,
                    samples=samples,
                    max_new_tokens=max_new_tokens,
                    generation_batch_size=generation_batch_size,
                    progress_bar=sampling_progress,
                )
                worker_results = [indexed_groups]
            else:
                with ThreadPoolExecutor(
                    max_workers=len(problem_shards)
                ) as executor:
                    futures = [
                        executor.submit(
                            _generate_problem_shard,
                            shard,
                            sampling_model,
                            tokenizer,
                            samples=samples,
                            max_new_tokens=max_new_tokens,
                            generation_batch_size=generation_batch_size,
                            progress_bar=sampling_progress,
                        )
                        for shard, sampling_model in zip(
                            problem_shards,
                            sampling_models,
                            strict=True,
                        )
                    ]
                    worker_results = [
                        future.result()
                        for future in futures
                    ]

        for worker_result in worker_results:
            for problem_index, candidate_group in worker_result:
                grouped_candidates[problem_index] = candidate_group
        if any(group is None for group in grouped_candidates):
            raise RuntimeError("parallel sampling did not return every problem")
        candidates = [
            candidate
            for group in grouped_candidates
            if group is not None
            for candidate in group
        ]

        verified = (
            verify_candidates(problems, candidates)
            if reward_mode == "verifier"
            else None
        )
        print(f"{round_name}: scoring old-policy token probabilities...", flush=True)
        old_logprobs = _score_old_policy(
            candidates,
            model,
            tokenizer,
            torch,
            device,
            label=f"{round_name} scoring",
            batch_size=batch_size,
            max_length=max_length,
            progress=progress,
        )
        pseudo_label_groups: list[dict[str, object]] = []
        if verified is not None:
            rewards = [1.0 if item.is_correct else 0.0 for item in verified]
            metrics: dict[str, object] = {
                **verification_metrics(verified),
            }
        else:
            rewards, pseudo_label_groups = assign_pseudo_label_rewards(
                candidates,
                min_consensus=min_consensus,
            )
            metrics = {
                **_pseudo_label_metrics(candidates, rewards, pseudo_label_groups),
            }
        advantages = group_relative_advantages(
            [candidate.id for candidate in candidates],
            rewards,
        )
        rollouts = [
            Rollout(
                candidate=candidate,
                reward=reward,
                old_policy_logprobs=old_policy_logprobs,
                advantage=advantage,
            )
            for candidate, reward, old_policy_logprobs, advantage in zip(
                candidates,
                rewards,
                old_logprobs,
                advantages,
                strict=True,
            )
        ]

        grouped_rewards: dict[str, list[float]] = defaultdict(list)
        for candidate, reward in zip(candidates, rewards, strict=True):
            grouped_rewards[candidate.id].append(reward)
        mixed_groups = sum(
            min(group_rewards) != max(group_rewards)
            for group_rewards in grouped_rewards.values()
        )
        metrics.update(
            {
                "objective": "grpo",
                "round": round_index,
                "mixed_reward_groups": mixed_groups,
                "reward_mode": reward_mode,
                "zero_advantage_rollouts": sum(
                    rollout.advantage == 0 for rollout in rollouts
                ),
                "clip_epsilon": clip_epsilon,
                "tune_mode": tune_mode,
                "min_consensus": min_consensus,
            }
        )

        pseudo_label_by_id = {
            str(record["id"]): record for record in pseudo_label_groups
        }
        write_jsonl(
            output_root / f"rollouts-{round_name}.jsonl",
            (
                _rollout_record(
                    rollout,
                    reward_mode=reward_mode,
                    pseudo_label_record=pseudo_label_by_id.get(
                        rollout.candidate.id
                    ),
                )
                for rollout in rollouts
            ),
        )
        if verified is not None:
            write_jsonl(
                output_root / f"verified-{round_name}.jsonl",
                (item.to_record() for item in verified),
            )
        else:
            write_jsonl(
                output_root / f"pseudo-labels-{round_name}.jsonl",
                pseudo_label_groups,
            )
        if mixed_groups:
            metrics.update(
                _train_grpo_rollouts(
                    rollouts,
                    model,
                    tokenizer,
                    optimizer,
                    torch,
                    device,
                    epochs=epochs,
                    batch_size=batch_size,
                    clip_epsilon=clip_epsilon,
                    max_grad_norm=max_grad_norm,
                    max_length=max_length,
                    seed=seed + round_index,
                    progress=progress,
                )
            )
        else:
            print(
                f"{round_name}: no mixed-reward groups; GRPO has no relative "
                "advantage signal, so skipping the update.",
                flush=True,
            )
            metrics.update(
                {
                    "grpo_loss": 0.0,
                    "optimizer_steps": 0,
                    "trained_rollouts": 0,
                    "clip_fraction": 0.0,
                }
            )

        metrics["tune_mode"] = tune_mode
        metrics_path = output_root / f"metrics-{round_name}.json"
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
            metrics_file.write("\n")

        checkpoint_kind = "adapter" if tune_mode == "lora" else "model"
        checkpoint_dir = output_root / round_name / checkpoint_kind
        save_training_checkpoint(model, tokenizer, checkpoint_dir)
        print(f"{round_name}: saved checkpoint to {checkpoint_dir}", flush=True)
        summaries.append(metrics)

    final_checkpoint = output_root / (
        "final-adapter" if tune_mode == "lora" else "final-model"
    )
    save_training_checkpoint(model, tokenizer, final_checkpoint)
    with (output_root / "training-summary.json").open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            {
                "objective": "grpo",
                "model": model_id,
                "tune_mode": tune_mode,
                "reward_mode": reward_mode,
                "min_consensus": min_consensus,
                "samples_per_problem": samples,
                "generation_batch_size": generation_batch_size,
                "sampling_devices": len(sampling_models),
                "max_new_tokens": max_new_tokens,
                "epochs_per_round": epochs,
                "batch_size": batch_size,
                "max_length": max_length,
                "learning_rate": learning_rate,
                "clip_epsilon": clip_epsilon,
                "max_grad_norm": max_grad_norm,
                "rounds": len(summaries),
                "final_checkpoint": str(final_checkpoint),
                "round_metrics": summaries,
            },
            summary_file,
            ensure_ascii=False,
            indent=2,
        )
        summary_file.write("\n")
    return summaries
