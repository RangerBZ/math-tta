"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from math_tta.data import load_candidates, load_problems, write_jsonl
from math_tta.generation import generate_candidates
from math_tta.metrics import evaluate
from math_tta.selection import select_trajectories


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="math-tta")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="sample candidate solutions")
    generate.add_argument("--input", required=True, help="input problem JSONL")
    generate.add_argument("--output", required=True, help="output candidate JSONL")
    generate.add_argument("--model", required=True, help="Hugging Face model ID or path")
    generate.add_argument("--samples", type=int, default=8)
    generate.add_argument("--max-new-tokens", type=int, default=512)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--seed", type=int, default=42)

    select = commands.add_parser("select", help="select agreement-based pseudo-labels")
    select.add_argument("--input", required=True, help="candidate JSONL")
    select.add_argument("--output", required=True, help="selected trajectory JSONL")
    select.add_argument("--min-value", type=float, default=0.5)

    evaluate_command = commands.add_parser("evaluate", help="evaluate candidate selection")
    evaluate_command.add_argument("--problems", required=True, help="labeled problem JSONL")
    evaluate_command.add_argument("--candidates", required=True, help="candidate JSONL")
    evaluate_command.add_argument("--min-value", type=float, default=0.5)
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
            )
            write_jsonl(
                arguments.output,
                (
                    {
                        "id": candidate.id,
                        "question": candidate.question,
                        "completion": candidate.completion,
                    }
                    for candidate in candidates
                ),
            )
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

        raise AssertionError(f"unhandled command: {arguments.command}")
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
