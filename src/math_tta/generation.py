"""Optional Hugging Face Transformers candidate generation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from math_tta.data import Candidate, Problem
from math_tta.prompts import format_generation_prompt


def _validate_generation_options(
    samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    top_k: int | None = None,
) -> None:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1]")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")


def set_seed(seed: int) -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Generation dependencies are missing. Install PyTorch and run "
            '`python -m pip install -e ".[model]"`.'
        ) from error

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_generation_model(
    model_id: str,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Generation dependencies are missing. Install PyTorch and run "
            '`python -m pip install -e ".[model]"`.'
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define a pad token or an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
    )
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError(
                "Loading a LoRA adapter requires PEFT. Install "
                '`python -m pip install -e ".[train]"`.'
            ) from error
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def _raw_token_logprobs(
    model: Any,
    sequence: Any,
    prompt_width: int,
    prompt_attention_mask: Any,
    tokenizer: Any,
    torch: Any,
) -> tuple[float, ...]:
    generated_ids = sequence[prompt_width:]
    eos_index: int | None = None
    if tokenizer.eos_token_id is not None:
        eos_positions = (generated_ids == tokenizer.eos_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(eos_positions):
            eos_index = int(eos_positions[0])
    if eos_index is not None:
        generated_length = eos_index + 1
    else:
        generated_length = generated_ids.numel()
    if generated_length == 0:
        return ()

    input_ids = sequence[: prompt_width + generated_length].unsqueeze(0)
    generated_attention = torch.ones(
        (1, generated_length),
        dtype=prompt_attention_mask.dtype,
        device=prompt_attention_mask.device,
    )
    attention_mask = torch.cat(
        (prompt_attention_mask.unsqueeze(0), generated_attention),
        dim=1,
    )
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    causal_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(causal_model, "model", None)
    language_head = getattr(causal_model, "lm_head", None)
    if backbone is None or language_head is None:
        raise RuntimeError(
            "Token log-probability recording requires a causal LM exposing "
            "'model' and 'lm_head' modules (as Qwen3 does)."
        )

    with torch.inference_mode():
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        token_logprobs: list[float] = []
        first_position = prompt_width - 1
        last_position = input_ids.shape[1] - 1
        for start in range(first_position, last_position, 16):
            end = min(start + 16, last_position)
            logits = language_head(hidden[:, start:end, :]).float()
            targets = input_ids[:, start + 1 : end + 1]
            selected = torch.log_softmax(logits, dim=-1).gather(
                -1,
                targets.unsqueeze(-1),
            )
            token_logprobs.extend(
                float(value)
                for value in selected.squeeze(0).squeeze(-1).cpu().tolist()
            )
    return tuple(token_logprobs)


def generate_candidate_batch(
    problems: Iterable[Problem],
    model: Any,
    tokenizer: Any,
    samples: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 1,
    record_token_logprobs: bool = False,
    top_k: int | None = None,
) -> Iterator[Candidate]:
    _validate_generation_options(
        samples,
        max_new_tokens,
        temperature,
        top_p,
        batch_size,
        top_k,
    )
    problem_list = list(problems)
    for start in range(0, len(problem_list), batch_size):
        batch = problem_list[start : start + batch_size]
        prompts = [
            format_generation_prompt(tokenizer, problem.question)
            for problem in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=not bool(getattr(tokenizer, "chat_template", None)),
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]

        import torch

        with torch.inference_mode():
            generation_options: dict[str, Any] = {
                "do_sample": True,
                "num_return_sequences": samples,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if top_k is not None:
                generation_options["top_k"] = top_k
            output = model.generate(
                **encoded,
                **generation_options,
            )

        for problem_index, problem in enumerate(batch):
            first_sequence = problem_index * samples
            prompt_token_ids = tuple(
                int(token_id)
                for token_id in encoded["input_ids"][problem_index][
                    encoded["attention_mask"][problem_index].bool()
                ].tolist()
            )
            for sequence in output[first_sequence : first_sequence + samples]:
                generated_ids = sequence[prompt_width:]
                if tokenizer.eos_token_id is not None:
                    eos_positions = (generated_ids == tokenizer.eos_token_id).nonzero(
                        as_tuple=True
                    )[0]
                    if len(eos_positions):
                        eos_index = int(eos_positions[0])
                        generated_ids = generated_ids[: eos_index + 1]
                completion = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()
                token_logprobs = (
                    _raw_token_logprobs(
                        model,
                        sequence,
                        prompt_width,
                        encoded["attention_mask"][problem_index],
                        tokenizer,
                        torch,
                    )
                    if record_token_logprobs
                    else None
                )
                yield Candidate(
                    id=problem.id,
                    question=problem.question,
                    completion=completion,
                    token_logprobs=token_logprobs,
                    completion_token_ids=tuple(
                        int(token_id) for token_id in generated_ids.tolist()
                    ),
                    prompt_token_ids=prompt_token_ids,
                )


def generate_candidates(
    problems: Iterable[Problem],
    model_id: str,
    samples: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 42,
    batch_size: int = 1,
    record_token_logprobs: bool = False,
    adapter_path: str | None = None,
    top_k: int | None = None,
) -> Iterator[Candidate]:
    _validate_generation_options(
        samples,
        max_new_tokens,
        temperature,
        top_p,
        batch_size,
        top_k,
    )
    set_seed(seed)
    tokenizer, model = load_generation_model(model_id, adapter_path)
    yield from generate_candidate_batch(
        problems,
        model,
        tokenizer,
        samples=samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        record_token_logprobs=record_token_logprobs,
        top_k=top_k,
    )
