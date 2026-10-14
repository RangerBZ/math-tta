"""JSONL data loading and writing."""

from __future__ import annotations

import json
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
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError(f"{source}:{line_number}: 'id' must be a non-empty string")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{source}:{line_number}: 'question' must be a non-empty string")
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"{source}:{line_number}: 'completion' must be a non-empty string")
        candidates.append(
            Candidate(
                id=problem_id,
                question=question.strip(),
                completion=completion.strip(),
            )
        )
    return candidates


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
