"""Evaluate a Prompt-B LoRA adapter on the frozen Spider dev split.

Run on Kaggle:
    python train/eval_adapter.py --adapter /kaggle/working/outputs/qwen1.5b_prompt_b_adapter
"""
from __future__ import annotations
import argparse, json, sys
from datetime import date
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from harness.generate import generate_sql, load_model_and_tokenizer
from harness.prompts import build_prompt_b, clean_sql_output
from harness.schema import get_schema
from harness.scorer import score_dataset

def main() -> None:
    """Load base plus adapter, generate greedy SQL, and write a Step 4 result."""
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--adapter",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--split-path",type=Path,default=Path("data/splits/dev_eval.json")); p.add_argument("--results-dir",type=Path,default=Path("results")); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--limit",type=int); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    info=json.loads((a.adapter/"adapter_info.json").read_text()); model,tokenizer=load_model_and_tokenizer(info["model"],load_in_4bit=True)
    from peft import PeftModel
    model=PeftModel.from_pretrained(model,a.adapter); examples=json.loads(a.split_path.read_text()); examples=examples[:5 if a.smoke else a.limit] if (a.smoke or a.limit) else examples
    prompts=[build_prompt_b(get_schema(a.data_root/"database"/e["db_id"]/f"{e['db_id']}.sqlite"),e["question"]) for e in examples]
    predictions=[clean_sql_output(x) for x in generate_sql(prompts,model,tokenizer,a.batch_size)]
    summary=score_dataset(examples,predictions,database_root=a.data_root/"database")
    name="step4_adapter_qwen2.5-coder-1.5b"; result={"run_name":name,"date":date.today().isoformat(),"step":4,"model":info["model"],"adapter":str(a.adapter),"prompt":"B","split":"dev_eval","n_examples":summary["n_examples"],"n_gold_failed":summary["n_gold_failed"],"n_correct":summary["n_correct"],"n_scored":summary["n_scored"],"ex":summary["ex"],"valid_sql_rate":summary["valid_sql_rate"],"error_counts":summary["error_counts"],"settings":{"seed":info["seed"],"max_new_tokens":256,"greedy":True},"git_commit":info.get("git_commit"),"notes":""}
    a.results_dir.mkdir(parents=True,exist_ok=True); (a.results_dir/f"{result['date']}_{name}.json").write_text(json.dumps(result,indent=2)+"\n")
    out=a.results_dir/"predictions"/f"{name}.jsonl"; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w") as f:
        for e,pred,o in zip(examples,predictions,summary["outcomes"]): f.write(json.dumps({"question":e["question"],"gold_sql":e["query"],"predicted_sql":pred,"correct":o["correct"],"error_type":o["error_type"]})+"\n")
    print(f"{name}: EX {summary['ex']:.1%} ({summary['n_correct']}/{summary['n_scored']}), valid SQL {summary['valid_sql_rate']:.1%}")
if __name__=="__main__": main()
