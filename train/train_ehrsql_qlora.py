"""Train one Stage B EHRSQL Prompt-B QLoRA adapter.

Run on Kaggle after attaching the private SFT-data dataset:
    python train/train_ehrsql_qlora.py --config configs/train_ehrsql_qlora.yaml \
      --data-file /kaggle/input/<sft-dataset>/ehrsql_mimic_iii_prompt_b.jsonl
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Protocol


LOGGER = logging.getLogger(__name__)


class Tokenizer(Protocol):
    """Minimal tokenizer interface used to prepare causal-LM training records."""

    eos_token: str
    pad_token_id: int

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
    ) -> Any:
        """Tokenize text and return an object with ``input_ids``."""


def load_config(path: Path) -> dict[str, Any]:
    """Load the Stage B QLoRA YAML configuration."""
    import yaml

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_record(
    record: dict[str, str], tokenizer: Tokenizer, max_seq_length: int
) -> dict[str, list[int]]:
    """Tokenize one SFT record while masking all prompt tokens from loss."""
    prompt_ids = tokenizer(
        record["prompt"],
        truncation=False,
        max_length=max_seq_length,
        add_special_tokens=False,
    ).input_ids
    full_ids = tokenizer(
        record["prompt"] + record["completion"] + tokenizer.eos_token,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    ).input_ids
    prompt_length = min(len(prompt_ids), len(full_ids))
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * prompt_length + full_ids[prompt_length:],
    }


def collate_rows(rows: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    """Right-pad tokenized records while preserving ignored prompt labels."""
    import torch

    maximum = max(len(row["input_ids"]) for row in rows)
    return {
        "input_ids": torch.tensor(
            [row["input_ids"] + [pad_token_id] * (maximum - len(row["input_ids"])) for row in rows]
        ),
        "attention_mask": torch.tensor(
            [row["attention_mask"] + [0] * (maximum - len(row["attention_mask"])) for row in rows]
        ),
        "labels": torch.tensor(
            [row["labels"] + [-100] * (maximum - len(row["labels"])) for row in rows]
        ),
    }


def git_commit() -> str | None:
    """Return the current short Git revision when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage B EHRSQL QLoRA command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_ehrsql_qlora.yaml"))
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument("--smoke", action="store_true", help="Run two CPU steps using a tiny model.")
    return parser


def main() -> None:
    """Load QLoRA, train on SQL completions, and save adapter metadata."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.data_file is not None:
        config["data_file"] = str(args.data_file)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.smoke:
        config.update(
            {
                "model_name": "sshleifer/tiny-gpt2",
                "load_in_4bit": False,
                "fp16": False,
                "max_seq_length": 256,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "save_steps": 2,
                "logging_steps": 1,
            }
        )

    data_file = Path(config["data_file"])
    if not data_file.is_file():
        raise FileNotFoundError(f"EHRSQL SFT JSONL not found: {data_file}")
    set_seed(int(config["seed"]))

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    dataset = load_dataset("json", data_files=str(data_file), split="train")
    required_columns = {"id", "prompt", "completion"}
    if not required_columns.issubset(dataset.column_names):
        raise ValueError(f"SFT JSONL must contain {sorted(required_columns)}; found {dataset.column_names}.")
    if args.smoke:
        dataset = dataset.select(range(min(20, len(dataset))))
    if not len(dataset):
        raise ValueError("EHRSQL SFT JSONL contains no training records.")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16 if config["fp16"] else torch.float32,
    }
    if config["load_in_4bit"]:
        model_kwargs.update(
            {
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                ),
            }
        )
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], **model_kwargs)
    if config.get("gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    if config["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
        ),
    )

    max_seq_length = int(config["max_seq_length"])
    dataset = dataset.map(
        lambda row: tokenize_record(row, tokenizer, max_seq_length),
        remove_columns=dataset.column_names,
    )
    output_dir = Path(config["output_dir"] + ("_smoke" if args.smoke else ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(config["num_train_epochs"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "lr_scheduler_type": "cosine",
        "fp16": bool(config["fp16"]),
        "save_steps": int(config["save_steps"]),
        "logging_steps": int(config["logging_steps"]),
        "save_total_limit": 2,
        "report_to": "none",
        "max_steps": 2 if args.smoke else -1,
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", False)),
        "optim": config.get("optim", "adamw_torch"),
    }
    if "warmup_ratio" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["warmup_ratio"] = float(config["warmup_ratio"])
    else:
        steps = math.ceil(
            len(dataset)
            / (int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"]))
        ) * float(config["num_train_epochs"])
        training_kwargs["warmup_steps"] = max(1, round(steps * float(config["warmup_ratio"])))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=dataset,
        data_collator=lambda rows: collate_rows(rows, tokenizer.pad_token_id),
    )
    resume = None if args.resume in (None, "latest") else args.resume
    if args.resume == "latest":
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
        resume = str(checkpoints[-1]) if checkpoints else None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    adapter_info = {
        "stage": "B",
        "dataset": "EHRSQL v1.5.0 MIMIC-III",
        "prompt": "B",
        "model": config["model_name"],
        "data_file": str(data_file),
        "n_training_examples": len(dataset),
        "max_seq_length": max_seq_length,
        "settings": config,
        "seed": config["seed"],
        "git_commit": git_commit(),
    }
    (output_dir / "adapter_info.json").write_text(
        json.dumps(adapter_info, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved EHRSQL adapter and metadata to %s", output_dir)


if __name__ == "__main__":
    main()
