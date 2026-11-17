"""Shared model setup and checkpoint saving for training objectives."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

TuneMode = Literal["lora", "full"]


def load_trainable_model(
    model_id: str,
    *,
    tune_mode: TuneMode,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[Any, Any, Any, Any]:
    if tune_mode not in {"lora", "full"}:
        raise ValueError(f"unsupported tune mode: {tune_mode!r}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are missing. Install the matching PyTorch "
            'build and run `python -m pip install -e ".[train]"`.'
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    print(f"Loading tokenizer from {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define a pad token or an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Loading model weights from {model_id}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=model_dtype,
    )
    print(f"Moving model to {device}...", flush=True)
    model.to(device)

    if tune_mode == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as error:
            raise RuntimeError(
                "LoRA training requires PEFT. Install "
                '`python -m pip install -e ".[train]"`.'
            ) from error
        print("Attaching LoRA adapter...", flush=True)
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
    return torch, tokenizer, model, device


def save_training_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
) -> None:
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
