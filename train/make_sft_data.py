"""Build frozen-Prompt-B Spider SFT records and report token lengths.

Run on Kaggle:
    python train/make_sft_data.py --data-root /kaggle/input/datasets/akshatgupte/spider-data-2
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.prompts import build_prompt_b
from harness.schema import get_schema

def build_record(example: dict, data_root: Path) -> dict:
    """Build one Prompt-B prompt/completion training record."""
    db_id = example["db_id"]
    schema = get_schema(data_root / "database" / db_id / f"{db_id}.sqlite")
    return {"prompt": build_prompt_b(schema, example["question"]), "completion": example["query"]}

def main() -> None:
    """Create JSONL SFT data, dropping records longer than the configured limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", "data/raw/spider")))
    parser.add_argument("--split-path", type=Path, default=Path("data/splits/train_sft.json"))
    parser.add_argument("--output", type=Path, default=Path("data/sft/train_prompt_b.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    examples = json.loads(args.split_path.read_text(encoding="utf-8"))
    if args.smoke: examples = examples[:20]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kept, lengths = [], []
    for example in examples:
        record = build_record(example, args.data_root)
        length = len(tokenizer(record["prompt"] + record["completion"] + tokenizer.eos_token).input_ids)
        lengths.append(length)
        if length <= args.max_length: kept.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record in kept: file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"kept={len(kept)} dropped={len(examples)-len(kept)} token_min={min(lengths)} token_mean={mean(lengths):.1f} token_max={max(lengths)}")

if __name__ == "__main__": main()
