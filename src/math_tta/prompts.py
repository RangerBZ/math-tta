"""Shared prompting and assistant-target tokenization helpers."""

from __future__ import annotations

from typing import Any


def user_message(question: str) -> str:
    return (
        "Solve the following math problem. Show your reasoning and put the "
        f"final answer in \\boxed{{}}.\n\n{question}"
    )


def format_generation_prompt(tokenizer: Any, question: str) -> str:
    message = user_message(question)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return message + "\n"


def tokenize_assistant_target(
    tokenizer: Any,
    question: str,
    completion: str,
    max_length: int,
    completion_token_ids: tuple[int, ...] | list[int] | None = None,
) -> tuple[list[int], list[int]] | None:
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    prompt = format_generation_prompt(tokenizer, question)
    prompt_ids = list(
        tokenizer(
            prompt,
            add_special_tokens=not has_chat_template,
        )["input_ids"]
    )
    if completion_token_ids is not None:
        target_ids = list(completion_token_ids)
    else:
        target_ids = list(
            tokenizer(completion, add_special_tokens=False)["input_ids"]
        )
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and (
            not target_ids or target_ids[-1] != eos_token_id
        ):
            target_ids.append(eos_token_id)

    input_ids = prompt_ids + target_ids
    prompt_length = len(prompt_ids)
    if len(input_ids) > max_length:
        return None
    if prompt_length >= len(input_ids):
        return None
    labels = [-100] * min(prompt_length, len(input_ids))
    labels.extend(input_ids[len(labels) :])
    if not any(label != -100 for label in labels):
        return None
    return input_ids, labels
