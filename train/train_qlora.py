"""Train a QLoRA adapter on frozen Prompt-B Spider SFT data.

Run on Kaggle:
    python train/train_qlora.py --config configs/train_qlora.yaml
"""
from __future__ import annotations
import argparse, inspect, json, math, random, subprocess
from pathlib import Path
import yaml

def seed_all(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    import numpy as np, torch
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main() -> None:
    """Load QLoRA, mask prompt tokens, train, checkpoint, and save adapter metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_qlora.yaml"))
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        cfg.update({"model_name":"sshleifer/tiny-gpt2", "load_in_4bit":False, "fp16":False, "max_seq_length":256, "per_device_train_batch_size":1, "gradient_accumulation_steps":1, "save_steps":2, "logging_steps":1})
    seed_all(cfg["seed"])
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    kwargs = {"torch_dtype": torch.float16 if cfg["fp16"] else torch.float32}
    if cfg["load_in_4bit"]:
        kwargs.update({"device_map":"auto", "quantization_config":BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)})
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **kwargs)
    if cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    if cfg["load_in_4bit"]: model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"], target_modules="all-linear", task_type=TaskType.CAUSAL_LM))
    dataset = load_dataset("json", data_files=str(cfg["data_file"]), split="train")
    if args.smoke: dataset = dataset.select(range(min(20, len(dataset))))
    def tokenize(row: dict) -> dict:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids
        ids = tokenizer(row["prompt"] + row["completion"] + tokenizer.eos_token, truncation=True, max_length=cfg["max_seq_length"], add_special_tokens=False).input_ids
        return {"input_ids":ids, "attention_mask":[1]*len(ids), "labels":[-100]*min(len(prompt_ids),len(ids))+ids[min(len(prompt_ids),len(ids)):]} 
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    def collate(rows: list[dict]) -> dict:
        maximum=max(len(r["input_ids"]) for r in rows)
        return {key:torch.tensor([r[key]+(([tokenizer.pad_token_id] if key=="input_ids" else [0] if key=="attention_mask" else [-100])*(maximum-len(r[key]))) for r in rows]) for key in ("input_ids","attention_mask","labels")}
    output = Path(cfg["output_dir"] + ("_smoke" if args.smoke else "")); output.mkdir(parents=True, exist_ok=True)
    train_kwargs = {"output_dir":str(output), "num_train_epochs":cfg["num_train_epochs"], "per_device_train_batch_size":cfg["per_device_train_batch_size"], "gradient_accumulation_steps":cfg["gradient_accumulation_steps"], "learning_rate":cfg["learning_rate"], "lr_scheduler_type":"cosine", "fp16":cfg["fp16"], "save_steps":cfg["save_steps"], "logging_steps":cfg["logging_steps"], "save_total_limit":2, "report_to":"none", "max_steps":2 if args.smoke else -1, "gradient_checkpointing":cfg.get("gradient_checkpointing", False), "optim":cfg.get("optim", "adamw_torch")}
    if "warmup_ratio" in inspect.signature(TrainingArguments).parameters:
        train_kwargs["warmup_ratio"] = cfg["warmup_ratio"]
    else:
        total_steps = math.ceil(len(dataset) / (cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"])) * cfg["num_train_epochs"]
        train_kwargs["warmup_steps"] = max(1, round(total_steps * cfg["warmup_ratio"]))
    train_args = TrainingArguments(**train_kwargs)
    trainer=Trainer(model=model,args=train_args,train_dataset=dataset,data_collator=collate)
    resume = None if args.resume in (None, "latest") else args.resume
    if args.resume == "latest":
        checkpoints=sorted(output.glob("checkpoint-*"), key=lambda p:int(p.name.split("-")[-1])); resume=str(checkpoints[-1]) if checkpoints else None
    trainer.train(resume_from_checkpoint=resume); trainer.save_model(str(output)); tokenizer.save_pretrained(str(output))
    try: commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except Exception: commit=None
    (output/"adapter_info.json").write_text(json.dumps({"model":cfg["model_name"],"data_file":cfg["data_file"],"settings":cfg,"seed":cfg["seed"],"git_commit":commit},indent=2)+"\n",encoding="utf-8")

if __name__ == "__main__": main()
