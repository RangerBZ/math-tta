"""Optional Hugging Face Transformers candidate generation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from math_tta.data import Candidate, Problem


def generate_candidates(
    problems: Iterable[Problem],
    model_id: str,
    samples: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 42,
) -> Iterator[Candidate]:
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1]")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Generation dependencies are missing. Install PyTorch and run "
            '`python -m pip install -e ".[model]"`.'
        ) from error

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define a pad token or an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    for problem in problems:
        user_message = (
            "Solve the following math problem. Show your reasoning and put the "
            f"final answer in \\boxed{{}}.\n\n{problem.question}"
        )
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = user_message
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=True,
                num_return_sequences=samples,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for sequence in output:
            completion = tokenizer.decode(
                sequence[prompt_length:],
                skip_special_tokens=True,
            ).strip()
            if completion:
                yield Candidate(
                    id=problem.id,
                    question=problem.question,
                    completion=completion,
                )
