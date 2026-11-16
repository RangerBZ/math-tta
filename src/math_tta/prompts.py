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
) -> tuple[list[int], list[int]] | None:
    messages = [
        {"role": "user", "content": user_message(question)},
        {"role": "assistant", "content": completion},
    ]
    if getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    else:
        prompt_text = user_message(question) + "\n"
        target_text = completion
        if tokenizer.eos_token:
            target_text += tokenizer.eos_token
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        input_ids = list(prompt_ids) + list(target_ids)

    prompt_ids = list(prompt_ids)
    input_ids = list(input_ids)
    prompt_length = len(prompt_ids)
    if input_ids[:prompt_length] != prompt_ids:
        raise ValueError(
            "chat template training sequence does not begin with its generation prompt"
        )
    if len(input_ids) > max_length:
        return None
    if prompt_length >= len(input_ids):
        return None
    labels = [-100] * min(prompt_length, len(input_ids))
    labels.extend(input_ids[len(labels) :])
    if not any(label != -100 for label in labels):
        return None
    return input_ids, labels
