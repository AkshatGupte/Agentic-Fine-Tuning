"""Generate SQL completions from a Hugging Face causal language model.

Run a CPU smoke generation (downloads the tiny test model on first use):
    python harness/generate.py --smoke
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Sequence


LOGGER = logging.getLogger(__name__)
TINY_SMOKE_MODEL = "sshleifer/tiny-gpt2"


def load_model_and_tokenizer(model_name: str, load_in_4bit: bool = False) -> tuple[Any, Any]:
    """Load a causal model and left-padded tokenizer, optionally quantized to 4-bit."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError("4-bit loading requires a CUDA-enabled environment.")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model, tokenizer


def _chat_prompts(prompts: Sequence[str], tokenizer: Any) -> list[str]:
    """Wrap each frozen text prompt as a user message when the tokenizer supports chat."""
    if getattr(tokenizer, "chat_template", None):
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    return list(prompts)


def generate_sql(
    prompts: Sequence[str], model: Any, tokenizer: Any, batch_size: int = 1, max_new_tokens: int = 256
) -> list[str]:
    """Generate one greedy completion per prompt with correct left-padding handling."""
    import torch

    completions: list[str] = []
    chat_prompts = _chat_prompts(prompts, tokenizer)
    for start in range(0, len(chat_prompts), batch_size):
        batch = chat_prompts[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {name: value.to(model.device) for name, value in inputs.items()}
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        completions.extend(tokenizer.batch_decode(completion_ids, skip_special_tokens=True))
    return completions


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone generation smoke-test parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Hugging Face model id.")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny model and five prompts on CPU.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


def main() -> None:
    """Run a small standalone generation check."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    if not args.model and not args.smoke:
        raise ValueError("Pass --model or --smoke.")
    model_name = TINY_SMOKE_MODEL if args.smoke else args.model
    model, tokenizer = load_model_and_tokenizer(model_name, args.load_in_4bit)
    prompts = ["Return only this SQL: SELECT 1;"] * (5 if args.smoke else 1)
    for completion in generate_sql(prompts, model, tokenizer, args.batch_size):
        print(completion)


if __name__ == "__main__":
    main()
