"""Outcome-reward GRPO training on on-policy language-model rollouts."""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import nullcontext
from itertools import islice
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Sequence

from math_tta.data import Candidate, Problem, Rollout, write_jsonl
from math_tta.generation import generate_candidate_batch, set_seed
from math_tta.model_training import (
    TuneMode,
    load_trainable_model,
    save_training_checkpoint,
)
from math_tta.verification import verify_candidates, verification_metrics


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
    batch_size: int,
    max_length: int,
) -> list[tuple[float, ...]]:
    scores: list[tuple[float, ...]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
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
    progress_interval = max(1, (len(trainable_rollouts) + 19) // 20)

    for epoch_index in range(1, epochs + 1):
        order = torch.randperm(
            len(trainable_rollouts),
            generator=generator,
        ).tolist()
        for start in range(0, len(order), batch_size):
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

            if (
                optimizer_steps == 1
                or optimizer_steps % progress_interval == 0
                or start + batch_size >= len(order)
            ):
                print(
                    f"GRPO epoch {epoch_index}/{epochs}: "
                    f"rollouts {min(start + batch_size, len(order))}/{len(order)}",
                    flush=True,
                )

    model.config.use_cache = True
    return {
        "grpo_loss": sum(losses) / len(losses),
        "optimizer_steps": optimizer_steps,
        "trained_rollouts": len(trainable_rollouts),
        "clip_fraction": sum(clipped_fractions) / len(clipped_fractions),
    }


def _rollout_record(rollout: Rollout) -> dict[str, object]:
    candidate = rollout.candidate
    return {
        "id": candidate.id,
        "question": candidate.question,
        "completion": candidate.completion,
        "prompt_token_ids": list(candidate.prompt_token_ids or ()),
        "completion_token_ids": list(candidate.completion_token_ids or ()),
        "reward": rollout.reward,
        "advantage": rollout.advantage,
        "old_policy_logprobs": list(rollout.old_policy_logprobs),
    }


def run_grpo_training(
    problems: Sequence[Problem],
    *,
    model_id: str,
    output_dir: str | Path,
    samples: int = 4,
    generation_batch_size: int = 1,
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
    if any(problem.answer is None for problem in problems):
        raise ValueError("every training problem must include a reference answer")
    if samples < 2:
        raise ValueError("GRPO requires at least two samples per problem")
    if min(generation_batch_size, rounds, epochs, batch_size, max_length) < 1:
        raise ValueError("batch sizes, rounds, epochs, and max_length must be positive")
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
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []

    for round_index in range(1, rounds + 1):
        round_name = f"round-{round_index:02d}"
        set_seed(seed + round_index)
        model.eval()
        print(
            f"{round_name}: sampling {samples} rollouts for "
            f"{len(problems)} problems...",
            flush=True,
        )
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
        candidates: list[Candidate] = []
        progress_interval = max(1, (len(problems) + 19) // 20)
        for problem_index, problem in enumerate(problems, start=1):
            group = list(islice(generated, samples))
            if len(group) != samples:
                raise RuntimeError(
                    f"generation returned fewer than {samples} samples "
                    f"for problem {problem.id!r}"
                )
            candidates.extend(group)
            if (
                problem_index == 1
                or problem_index % progress_interval == 0
                or problem_index == len(problems)
            ):
                print(
                    f"{round_name}: sampled {problem_index}/{len(problems)} problems",
                    flush=True,
                )

        verified = verify_candidates(problems, candidates)
        print(f"{round_name}: scoring old-policy token probabilities...", flush=True)
        old_logprobs = _score_old_policy(
            candidates,
            model,
            tokenizer,
            torch,
            device,
            batch_size=batch_size,
            max_length=max_length,
        )
        rewards = [1.0 if item.is_correct else 0.0 for item in verified]
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

        metrics: dict[str, object] = {
            **verification_metrics(verified),
        }
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
                "zero_advantage_rollouts": sum(
                    rollout.advantage == 0 for rollout in rollouts
                ),
                "clip_epsilon": clip_epsilon,
                "tune_mode": tune_mode,
            }
        )

        write_jsonl(
            output_root / f"rollouts-{round_name}.jsonl",
            (_rollout_record(rollout) for rollout in rollouts),
        )
        write_jsonl(
            output_root / f"verified-{round_name}.jsonl",
            (item.to_record() for item in verified),
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
                "samples_per_problem": samples,
                "generation_batch_size": generation_batch_size,
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
