"""JSONL data loading and writing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    answer: str | None = None


@dataclass(frozen=True)
class Candidate:
    id: str
    question: str
    completion: str
    token_logprobs: tuple[float, ...] | None = None
    completion_token_ids: tuple[int, ...] | None = None
    prompt_token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Rollout:
    candidate: Candidate
    reward: float
    old_policy_logprobs: tuple[float, ...]
    advantage: float
    group_weight: float = 1.0


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, record


def load_problems(path: str | Path) -> list[Problem]:
    source = Path(path)
    problems: list[Problem] = []
    seen_ids: set[str] = set()
    for line_number, record in _read_jsonl(source):
        problem_id = record.get("id")
        question = record.get("question")
        answer = record.get("answer")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError(f"{source}:{line_number}: 'id' must be a non-empty string")
        if problem_id in seen_ids:
            raise ValueError(f"{source}:{line_number}: duplicate problem id {problem_id!r}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{source}:{line_number}: 'question' must be a non-empty string")
        if answer is not None and not isinstance(answer, (str, int, float)):
            raise ValueError(f"{source}:{line_number}: 'answer' must be a string or number")
        problems.append(
            Problem(
                id=problem_id,
                question=question.strip(),
                answer=str(answer) if answer is not None else None,
            )
        )
        seen_ids.add(problem_id)
    return problems


def load_candidates(path: str | Path) -> list[Candidate]:
    source = Path(path)
    candidates: list[Candidate] = []
    for line_number, record in _read_jsonl(source):
        problem_id = record.get("id")
        question = record.get("question")
        completion = record.get("completion")
        token_logprobs = record.get("token_logprobs")
        completion_token_ids = record.get("completion_token_ids")
        prompt_token_ids = record.get("prompt_token_ids")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError(f"{source}:{line_number}: 'id' must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{source}:{line_number}: 'question' must be a non-empty string")
        if not isinstance(completion, str):
            raise ValueError(f"{source}:{line_number}: 'completion' must be a string")
        if token_logprobs is not None:
            if not isinstance(token_logprobs, list) or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in token_logprobs
            ):
                raise ValueError(
                    f"{source}:{line_number}: 'token_logprobs' must be a list "
                    "of finite numbers"
                )
        if completion_token_ids is not None:
            if not isinstance(completion_token_ids, list) or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in completion_token_ids
            ):
                raise ValueError(
                    f"{source}:{line_number}: 'completion_token_ids' must be "
                    "a list of integers"
                )
        if prompt_token_ids is not None:
            if not isinstance(prompt_token_ids, list) or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in prompt_token_ids
            ):
                raise ValueError(
                    f"{source}:{line_number}: 'prompt_token_ids' must be "
                    "a list of integers"
                )
        candidates.append(
            Candidate(
                id=problem_id,
                question=question.strip(),
                completion=completion.strip(),
                token_logprobs=(
                    tuple(float(value) for value in token_logprobs)
                    if token_logprobs is not None
                    else None
                ),
                completion_token_ids=(
                    tuple(completion_token_ids)
                    if completion_token_ids is not None
                    else None
                ),
                prompt_token_ids=(
                    tuple(prompt_token_ids)
                    if prompt_token_ids is not None
                    else None
                ),
            )
        )
    return candidates


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_gsm8k(split: str, output_path: str | Path) -> int:
    if split not in {"train", "test"}:
        raise ValueError("GSM8K split must be 'train' or 'test'")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "GSM8K import dependencies are missing. Install "
            '`python -m pip install -e ".[train]"`.'
        ) from error

    rows = load_dataset("openai/gsm8k", "main", split=split)
    records = (
        {
            "id": f"{split}-{index:05d}",
            "question": row["question"],
            "answer": row["answer"],
        }
        for index, row in enumerate(rows)
    )
    count = 0

    def counted_records() -> Iterable[dict[str, Any]]:
        nonlocal count
        for record in records:
            count += 1
            yield record

    write_jsonl(output_path, counted_records())
    return count


def prepare_svamp(input_path: str | Path, output_path: str | Path) -> int:
    """Convert the SVAMP JSON file into labelled problem JSONL."""
    rows = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("SVAMP input must be a JSON list of objects")
    records = []
    for index, row in enumerate(rows):
        try:
            question = f"{row['Body']} {row['Question']}".strip()
            answer = str(row["Answer"])
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"SVAMP row {index} must contain Body, Question, and Answer"
            ) from error
        records.append(
            {
                "id": f"svamp-{index:05d}",
                "question": question,
                "answer": answer,
            }
        )
    write_jsonl(output_path, records)
    return len(records)
