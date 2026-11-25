"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence

from math_tta.data import (
    load_candidates,
    load_problems,
    prepare_gsm8k,
    write_jsonl,
)
from math_tta.generation import generate_candidates
from math_tta.metrics import evaluate
from math_tta.progress import load_tqdm
from math_tta.rl_training import run_grpo_training
from math_tta.selection import select_trajectories
from math_tta.training import run_self_training
from math_tta.verification import verify_candidates, verification_metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="math-tta")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="sample candidate solutions")
    generate.add_argument("--input", required=True, help="input problem JSONL")
    generate.add_argument("--output", required=True, help="output candidate JSONL")
    generate.add_argument("--model", required=True, help="Hugging Face model ID or path")
    generate.add_argument(
        "--adapter",
        help="optional PEFT adapter path, loaded on top of --model",
    )
    generate.add_argument("--samples", type=int, default=8)
    generate.add_argument("--max-new-tokens", type=int, default=512)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--batch-size", type=int, default=1)
    generate.add_argument(
        "--record-token-logprobs",
        action="store_true",
        help="record raw-model log probabilities for sampled tokens",
    )

    prepare = commands.add_parser(
        "prepare-gsm8k",
        help="download a GSM8K split and write problem JSONL",
    )
    prepare.add_argument("--split", choices=("train", "test"), default="train")
    prepare.add_argument("--output", required=True, help="output problem JSONL")

    score = commands.add_parser(
        "score",
        help="verify candidates against reference answers and report outcomes",
    )
    score.add_argument("--problems", required=True, help="labeled problem JSONL")
    score.add_argument("--candidates", required=True, help="candidate JSONL")
    score.add_argument("--output", required=True, help="verified candidate JSONL")

    select = commands.add_parser("select", help="select agreement-based pseudo-labels")
    select.add_argument("--input", required=True, help="candidate JSONL")
    select.add_argument("--output", required=True, help="selected trajectory JSONL")
    select.add_argument("--min-value", type=float, default=0.5)

    evaluate_command = commands.add_parser("evaluate", help="evaluate candidate selection")
    evaluate_command.add_argument("--problems", required=True, help="labeled problem JSONL")
    evaluate_command.add_argument("--candidates", required=True, help="candidate JSONL")
    evaluate_command.add_argument("--min-value", type=float, default=0.5)

    train = commands.add_parser(
        "train",
        help="sample, verify, and fine-tune on correct generated traces",
    )
    train.add_argument(
        "--input",
        required=True,
        help="labeled training problem JSONL (do not use the test split)",
    )
    train.add_argument("--output-dir", required=True, help="artifacts and adapters")
    train.add_argument("--model", default="Qwen/Qwen3-0.6B")
    train.add_argument("--samples", type=int, default=4)
    train.add_argument("--generation-batch-size", type=int, default=1)
    train.add_argument("--max-new-tokens", type=int, default=512)
    train.add_argument("--temperature", type=float, default=0.7)
    train.add_argument("--top-p", type=float, default=0.95)
    train.add_argument("--rounds", type=int, default=1)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--max-length", type=int, default=2048)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--lora-dropout", type=float, default=0.05)
    train.add_argument(
        "--tune-mode",
        choices=("lora", "full"),
        default="lora",
        help="update LoRA adapter parameters or all model parameters",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--record-token-logprobs",
        action="store_true",
        help="record raw-model log probabilities for sampled tokens",
    )
    train.add_argument(
        "--limit-problems",
        type=int,
        help="optional pilot limit; problems are taken from the start of the file",
    )

    grpo = commands.add_parser(
        "grpo",
        help="sample grouped rollouts, verify outcomes, and update with clipped GRPO",
    )
    grpo.add_argument(
        "--input",
        required=True,
        help="labeled training problem JSONL (do not use the test split)",
    )
    grpo.add_argument("--output-dir", required=True, help="rollouts and checkpoints")
    grpo.add_argument("--model", default="Qwen/Qwen3-0.6B")
    grpo.add_argument("--samples", type=int, default=4)
    grpo.add_argument("--generation-batch-size", type=int, default=1)
    grpo.add_argument(
        "--sampling-devices",
        type=int,
        default=1,
        help="number of visible CUDA GPUs to use for parallel rollout sampling",
    )
    grpo.add_argument(
        "--sampling-policy",
        choices=("fixed", "tta-1", "tta-2", "tta-3", "tta-4"),
        default="fixed",
        help=(
            "fixed rollouts, TTA-1 adaptive sampling, TTA-2 confidence rewards, "
            "TTA-3 Bayesian-calibrated rewards, or TTA-4 cross-fitted "
            "group-reliability weighting"
        ),
    )
    grpo.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="adaptive-policy rollout cap per problem",
    )
    grpo.add_argument(
        "--tta1-threshold",
        type=float,
        default=0.8,
        help="adaptive-policy minimum count-consensus fraction",
    )
    grpo.add_argument("--max-new-tokens", type=int, default=512)
    grpo.add_argument("--rounds", type=int, default=1)
    grpo.add_argument("--epochs", type=int, default=1)
    grpo.add_argument("--batch-size", type=int, default=1)
    grpo.add_argument("--max-length", type=int, default=2048)
    grpo.add_argument("--learning-rate", type=float, default=1e-6)
    grpo.add_argument("--clip-epsilon", type=float, default=0.2)
    grpo.add_argument("--max-grad-norm", type=float, default=1.0)
    grpo.add_argument("--lora-rank", type=int, default=8)
    grpo.add_argument("--lora-alpha", type=int, default=16)
    grpo.add_argument("--lora-dropout", type=float, default=0.05)
    grpo.add_argument(
        "--reward-mode",
        choices=("verifier", "pseudo-label"),
        default="verifier",
        help=(
            "use labeled-answer verification or within-group majority "
            "pseudo-label rewards"
        ),
    )
    grpo.add_argument(
        "--min-consensus",
        type=float,
        default=0.5,
        help="minimum unique plurality fraction required for pseudo-label rewards",
    )
    grpo.add_argument(
        "--tune-mode",
        choices=("lora", "full"),
        default="lora",
        help="update LoRA adapter parameters or all model parameters",
    )
    grpo.add_argument("--seed", type=int, default=42)
    grpo.add_argument(
        "--limit-problems",
        type=int,
        help="optional pilot limit; problems are taken from the start of the file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "generate":
            problems = load_problems(arguments.input)
            candidates = generate_candidates(
                problems,
                model_id=arguments.model,
                samples=arguments.samples,
                max_new_tokens=arguments.max_new_tokens,
                temperature=arguments.temperature,
                top_p=arguments.top_p,
                seed=arguments.seed,
                batch_size=arguments.batch_size,
                record_token_logprobs=arguments.record_token_logprobs,
                adapter_path=arguments.adapter,
            )

            with load_tqdm()(
                total=len(problems),
                desc="Generating",
                unit="problem",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as generation_progress:

                def candidate_records() -> Iterator[dict[str, object]]:
                    sample_indices: dict[str, int] = {}
                    for candidate in candidates:
                        sample_index = sample_indices.get(candidate.id, 0)
                        sample_indices[candidate.id] = sample_index + 1
                        yield {
                            "id": candidate.id,
                            "question": candidate.question,
                            "completion": candidate.completion,
                            "completion_token_ids": (
                                list(candidate.completion_token_ids)
                                if candidate.completion_token_ids is not None
                                else None
                            ),
                            "prompt_token_ids": (
                                list(candidate.prompt_token_ids)
                                if candidate.prompt_token_ids is not None
                                else None
                            ),
                            "sample_index": sample_index,
                            "model": arguments.model,
                            "adapter": arguments.adapter,
                            "seed": arguments.seed,
                            "samples_per_problem": arguments.samples,
                            "max_new_tokens": arguments.max_new_tokens,
                            "temperature": arguments.temperature,
                            "top_p": arguments.top_p,
                            "batch_size": arguments.batch_size,
                            "token_logprobs": (
                                list(candidate.token_logprobs)
                                if candidate.token_logprobs is not None
                                else None
                            ),
                        }
                        if sample_index + 1 == arguments.samples:
                            generation_progress.update(1)

                write_jsonl(
                    arguments.output,
                    candidate_records(),
                )
            return 0

        if arguments.command == "prepare-gsm8k":
            count = prepare_gsm8k(arguments.split, arguments.output)
            print(f"Wrote {count} {arguments.split} problems -> {arguments.output}")
            return 0

        if arguments.command == "score":
            verified = verify_candidates(
                load_problems(arguments.problems),
                load_candidates(arguments.candidates),
            )
            write_jsonl(arguments.output, (item.to_record() for item in verified))
            print(json.dumps(verification_metrics(verified), indent=2, sort_keys=True))
            return 0

        if arguments.command == "select":
            selections = select_trajectories(
                load_candidates(arguments.input),
                min_value=arguments.min_value,
            )
            write_jsonl(
                arguments.output,
                (selection.to_record() for selection in selections),
            )
            print(f"Selected {len(selections)} trajectories -> {arguments.output}")
            return 0

        if arguments.command == "evaluate":
            metrics = evaluate(
                load_problems(arguments.problems),
                load_candidates(arguments.candidates),
                min_value=arguments.min_value,
            )
            print(json.dumps(metrics, indent=2, sort_keys=True))
            return 0

        if arguments.command == "train":
            problems = load_problems(arguments.input)
            if arguments.limit_problems is not None:
                if arguments.limit_problems < 1:
                    raise ValueError("--limit-problems must be at least 1")
                problems = problems[: arguments.limit_problems]
            summaries = run_self_training(
                problems,
                model_id=arguments.model,
                output_dir=arguments.output_dir,
                samples=arguments.samples,
                generation_batch_size=arguments.generation_batch_size,
                max_new_tokens=arguments.max_new_tokens,
                temperature=arguments.temperature,
                top_p=arguments.top_p,
                rounds=arguments.rounds,
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                gradient_accumulation_steps=arguments.gradient_accumulation_steps,
                max_length=arguments.max_length,
                learning_rate=arguments.learning_rate,
                lora_rank=arguments.lora_rank,
                lora_alpha=arguments.lora_alpha,
                lora_dropout=arguments.lora_dropout,
                tune_mode=arguments.tune_mode,
                seed=arguments.seed,
                record_token_logprobs=arguments.record_token_logprobs,
            )
            print(json.dumps(summaries, indent=2, sort_keys=True))
            return 0

        if arguments.command == "grpo":
            problems = load_problems(arguments.input)
            if arguments.limit_problems is not None:
                if arguments.limit_problems < 1:
                    raise ValueError("--limit-problems must be at least 1")
                problems = problems[: arguments.limit_problems]
            summaries = run_grpo_training(
                problems,
                model_id=arguments.model,
                output_dir=arguments.output_dir,
                samples=arguments.samples,
                generation_batch_size=arguments.generation_batch_size,
                sampling_devices=arguments.sampling_devices,
                sampling_policy=arguments.sampling_policy,
                max_samples=arguments.max_samples,
                tta1_threshold=arguments.tta1_threshold,
                max_new_tokens=arguments.max_new_tokens,
                rounds=arguments.rounds,
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                max_length=arguments.max_length,
                learning_rate=arguments.learning_rate,
                clip_epsilon=arguments.clip_epsilon,
                max_grad_norm=arguments.max_grad_norm,
                lora_rank=arguments.lora_rank,
                lora_alpha=arguments.lora_alpha,
                lora_dropout=arguments.lora_dropout,
                tune_mode=arguments.tune_mode,
                reward_mode=arguments.reward_mode,
                min_consensus=arguments.min_consensus,
                seed=arguments.seed,
            )
            print(json.dumps(summaries, indent=2, sort_keys=True))
            return 0

        raise AssertionError(f"unhandled command: {arguments.command}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
