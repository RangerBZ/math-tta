"""Outcome-verified iterative self-training with parameter-efficient SFT."""

from __future__ import annotations

import json
from contextlib import nullcontext
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Any

from math_tta.data import Problem
from math_tta.generation import generate_candidate_batch, set_seed
from math_tta.prompts import tokenize_assistant_target
from math_tta.verification import (
    VerifiedCandidate,
    verification_metrics,
    verify_candidates,
)


def _write_record(output: Any, record: dict[str, object]) -> None:
    output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, record: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _training_collator(tokenizer: Any, torch: Any, max_length: int) -> Any:
    def collate(examples: list[VerifiedCandidate]) -> dict[str, Any]:
        tokenized = [
            item
            for example in examples
            if (
                item := tokenize_assistant_target(
                    tokenizer,
                    example.question,
                    example.completion,
                    max_length,
                )
            )
            is not None
        ]
        if not tokenized:
            raise RuntimeError("training batch contains no tokenizable accepted traces")

        width = max(len(input_ids) for input_ids, _ in tokenized)
        input_ids = []
        attention_masks = []
        labels = []
        for sequence, sequence_labels in tokenized:
            padding = width - len(sequence)
            input_ids.append(sequence + [tokenizer.pad_token_id] * padding)
            attention_masks.append([1] * len(sequence) + [0] * padding)
            labels.append(sequence_labels + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def _train_verified_examples(
    examples: Sequence[VerifiedCandidate],
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    torch: Any,
    device: Any,
    *,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    seed: int,
) -> dict[str, float | int]:
    if not examples:
        raise ValueError("fine-tuning requires at least one verifier-passing trace")

    trainable_examples = [
        example
        for example in examples
        if tokenize_assistant_target(
            tokenizer,
            example.question,
            example.completion,
            max_length,
        )
        is not None
    ]
    skipped_examples = len(examples) - len(trainable_examples)
    if not trainable_examples:
        oversized_lengths: list[int] = []
        no_target_count = 0
        for example in examples:
            unbounded = tokenize_assistant_target(
                tokenizer,
                example.question,
                example.completion,
                max_length=2**63 - 1,
            )
            if unbounded is None:
                no_target_count += 1
            else:
                oversized_lengths.append(len(unbounded[0]))

        if oversized_lengths:
            length_range = (
                f"tokenized lengths {min(oversized_lengths)}"
                f"-{max(oversized_lengths)}"
            )
        else:
            length_range = "no tokenized assistant targets"
        detail = (
            f"{len(oversized_lengths)} exceed --max-length={max_length}; "
            f"{no_target_count} contain no target tokens; {length_range}."
        )
        raise ValueError(
            "No accepted traces can be used for SFT: "
            f"{detail} Check the tokenizer's chat template and completion "
            "tokenization; increase --max-length only if the measured lengths "
            "show traces are being truncated by the configured limit."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        trainable_examples,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_training_collator(tokenizer, torch, max_length),
    )
    model.config.use_cache = False
    model.train()
    losses: list[float] = []
    optimizer_steps = 0
    trained_examples = 0
    batches_per_epoch = len(loader)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            trained_examples += len(batch["input_ids"])
            accumulation_group = batch_index // gradient_accumulation_steps
            group_start = accumulation_group * gradient_accumulation_steps
            group_size = min(
                gradient_accumulation_steps,
                batches_per_epoch - group_start,
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else nullcontext()
            )
            with autocast:
                output = model(**model_inputs)
                loss = output.loss
            losses.append(float(loss.detach().float().cpu()))
            (loss / group_size).backward()

            end_of_group = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == batches_per_epoch
            )
            if end_of_group:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

    model.config.use_cache = True
    if not losses:
        raise ValueError(
            "No accepted traces fit within --max-length; increase the limit "
            "or reduce prompt/completion length."
        )
    return {
        "accepted_traces": len(examples),
        "trained_examples": trained_examples,
        "skipped_examples": skipped_examples,
        "trained_batches": len(losses),
        "optimizer_steps": optimizer_steps,
        "mean_loss": sum(losses) / len(losses),
    }


def run_self_training(
    problems: Sequence[Problem],
    *,
    model_id: str,
    output_dir: str | Path,
    samples: int = 4,
    generation_batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    rounds: int = 1,
    epochs: int = 1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    max_length: int = 2048,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    seed: int = 42,
    record_token_logprobs: bool = False,
) -> list[dict[str, float | int | None]]:
    if not problems:
        raise ValueError("training requires at least one problem")
    if len({problem.id for problem in problems}) != len(problems):
        raise ValueError("training problem data contains duplicate ids")
    if any(problem.id.startswith("test-") for problem in problems):
        raise ValueError(
            "training input contains GSM8K test-split ids; use the training split only"
        )
    if any(problem.answer is None for problem in problems):
        raise ValueError("every training problem must include a reference answer")
    if samples < 1 or generation_batch_size < 1 or rounds < 1 or epochs < 1:
        raise ValueError("samples, batch sizes, rounds, and epochs must be positive")
    if batch_size < 1 or gradient_accumulation_steps < 1 or max_length < 1:
        raise ValueError("training batch sizes and max_length must be positive")
    if learning_rate <= 0 or lora_rank < 1 or lora_alpha < 1:
        raise ValueError("learning_rate, lora_rank, and lora_alpha must be positive")
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in the interval [0, 1)")

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are missing. Install the matching PyTorch "
            "build and run `python -m pip install -e \".[train]\"`."
        ) from error

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define a pad token or an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
    )
    model.to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, float | int | None]] = []

    for round_index in range(1, rounds + 1):
        round_name = f"round-{round_index:02d}"
        candidates_path = output_root / f"candidates-{round_name}.jsonl"
        verified_path = output_root / f"verified-{round_name}.jsonl"
        accepted_path = output_root / f"accepted-{round_name}.jsonl"
        metrics_path = output_root / f"metrics-{round_name}.json"
        accepted: list[VerifiedCandidate] = []
        scored_for_metrics: list[VerifiedCandidate] = []

        model.config.use_cache = True
        model.eval()
        generated = iter(
            generate_candidate_batch(
                problems,
                model,
                tokenizer,
                samples=samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                batch_size=generation_batch_size,
                record_token_logprobs=record_token_logprobs,
            )
        )
        with (
            candidates_path.open("w", encoding="utf-8") as raw_output,
            verified_path.open("w", encoding="utf-8") as verified_output,
            accepted_path.open("w", encoding="utf-8") as accepted_output,
        ):
            for problem in problems:
                problem_candidates = list(islice(generated, samples))
                if len(problem_candidates) != samples:
                    raise RuntimeError(
                        f"generation returned fewer than {samples} samples "
                        f"for problem {problem.id!r}"
                    )
                verified = verify_candidates([problem], problem_candidates)
                scored_for_metrics.extend(verified)
                for sample_index, (candidate, scored) in enumerate(
                    zip(problem_candidates, verified)
                ):
                    _write_record(
                        raw_output,
                        {
                            "id": candidate.id,
                            "question": candidate.question,
                            "completion": candidate.completion,
                            "sample_index": sample_index,
                            "round": round_index,
                            "model": model_id,
                            "seed": seed,
                            "samples_per_problem": samples,
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "top_p": top_p,
                            "token_logprobs": (
                                list(candidate.token_logprobs)
                                if candidate.token_logprobs is not None
                                else None
                            ),
                        },
                    )
                    _write_record(verified_output, scored.to_record())
                    if scored.is_correct:
                        accepted.append(scored)
                        _write_record(accepted_output, scored.to_record())

        metrics = verification_metrics(scored_for_metrics)
        if not accepted:
            _write_json(metrics_path, metrics)
            raise RuntimeError(
                f"{round_name} produced no verifier-passing traces; no "
                "fine-tuning update was performed."
            )

        training_metrics = _train_verified_examples(
            accepted,
            model,
            tokenizer,
            optimizer,
            torch,
            device,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            seed=seed + round_index,
        )
        metrics.update(training_metrics)
        _write_json(metrics_path, metrics)
        checkpoint_dir = output_root / round_name / "adapter"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        summaries.append(metrics)

    final_adapter = output_root / "final-adapter"
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    _write_json(
        output_root / "training-summary.json",
        {
            "model": model_id,
            "seed": seed,
            "samples_per_problem": samples,
            "generation_batch_size": generation_batch_size,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "epochs_per_round": epochs,
            "training_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_length": max_length,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "rounds": len(summaries),
            "final_adapter": str(final_adapter),
            "round_metrics": summaries,
        },
    )
    return summaries
