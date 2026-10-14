"""Extract and normalize final answers from generated reasoning."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


_ANSWER_MARKER = re.compile(r"(?:####|final\s+answer\s*:?)", re.IGNORECASE)


def _boxed_contents(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index].strip()
    return None


def extract_answer(completion: str) -> str | None:
    boxed = _boxed_contents(completion)
    if boxed:
        return boxed

    markers = list(_ANSWER_MARKER.finditer(completion))
    if not markers:
        return None
    answer_text = completion[markers[-1].end() :].strip()
    if not answer_text:
        return None
    answer = answer_text.splitlines()[0].strip()
    answer = answer.strip(" \t`$*_")
    if answer.endswith(".") and not re.search(r"\d\.\d$", answer):
        answer = answer[:-1].rstrip()
    return answer or None


def normalize_answer(answer: str) -> str:
    normalized = answer.strip().replace(",", "").replace("$", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.casefold()

    fraction_match = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", normalized)
    if fraction_match:
        try:
            return str(Fraction(int(fraction_match.group(1)), int(fraction_match.group(2))))
        except ZeroDivisionError:
            return normalized

    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return normalized
    if value.is_finite():
        fraction = Fraction(value)
        return str(fraction.numerator) if fraction.denominator == 1 else str(fraction)
    return normalized
