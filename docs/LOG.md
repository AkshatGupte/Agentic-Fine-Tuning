# LOG.md: Decisions and Run Notes

Add a short entry whenever you or the agent:
- change a setting from `docs/PROTOCOL.md`,
- make a manual choice (drop bad data, change the learning rate, pick a prompt),
- find a bug that affected scores,
- finish a Kaggle run.

These notes are important. Later, the training agent will need to make these same decisions automatically, so write down **what was decided and why**.

Keep entries short. Newest entry at the bottom.

---

## Entry format

```
### YYYY-MM-DD | Step N | short title
- What happened:
- Decision / change:
- Why:
- Effect on results (if known):
- Files changed / result files:
```

---

## Entries

### 2026-10-22 | Step 1 | Project setup and local Spider smoke test
- What happened: created the Stage A folders, configuration, split preparation script, schema reader, and smoke loader. The local smoke test prepared and loaded the fixed Spider samples and printed one SQLite schema.
- Decision / change: used frozen seed 42, `train_sft` size 3000, `dev_eval` size 300, and SQL timeout 10 seconds. Local Spider root is `data/raw/spider`; Kaggle root is `/kaggle/input/datasets/akshatgupte/spider-data-2`.
- Why: these are the frozen Stage A settings and the user supplied the data locations.
- Effect on results (if known): no model or scoring results yet.
- Files changed / result files: `requirements.txt`, `configs/base.yaml`, `data/prepare_spider.py`, `harness/schema.py`, `scripts/smoke_load.py`, `data/splits/train_sft.json`, `data/splits/dev_eval.json`.

### 2026-10-22 | Step 1 | Dev model and Spider license confirmed
- What happened: user selected the development model and confirmed the Spider license is acceptable for this project.
- Decision / change: development model is `Qwen/Qwen2.5-Coder-1.5B-Instruct`; Spider is used under CC BY-SA 4.0.
- Why: user confirmation.
- Effect on results (if known): no model or scoring results yet.
- Files changed / result files: `docs/PROTOCOL.md`, `docs/LOG.md`.

### 2026-10-22 | Step 1 | Kaggle setup check passed
- What happened: the user ran split preparation and the schema smoke test successfully on Kaggle with the attached private Spider dataset.
- Decision / change: none.
- Why: validate the Stage A data path and scripts before building the scorer.
- Effect on results (if known): `train_sft` has 3000 examples and `dev_eval` has 300 examples; no model or scoring results yet.
- Files changed / result files: no result file; Kaggle setup-check notebook run completed.

### 2026-10-22 | Step 2 | Local gold-query validation passed
- What happened: ran the complete `dev_eval` gold-query check locally after the executor and scorer unit tests passed.
- Decision / change: none; no gold-query failures or metric issues were found.
- Why: validate the execution-accuracy scorer before any model evaluation.
- Effect on results (if known): EX 100.0% (300/300), valid SQL 100.0%, gold failed 0.
- Files changed / result files: `results/2026-10-22_step2_check_gold.json`, `results/predictions/step2_check_gold.jsonl`.

### 2026-10-22 | Step 3 | Prompt B frozen after Kaggle baseline comparison
- What happened: user ran both full 300-example prompt-only baselines on Kaggle using `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
- Decision / change: froze Prompt B for all subsequent training and evaluation.
- Why: Prompt B achieved EX 59.3% (178/300) and valid SQL 77.7%, versus Prompt A at EX 12.7% (38/300) and valid SQL 15.0%. The choice uses aggregate scores only, not individual dev-set error inspection.
- Effect on results (if known): Prompt B is the baseline to compare with the adapter in Step 4. Prompt A errors: 249 `other`, 6 `no_such_column`, 7 `wrong_result`; Prompt B errors: 23 `other`, 42 `no_such_column`, 2 `no_such_table`, 55 `wrong_result`.
- Files changed / result files: `results/2026-10-22_step3_promptA_qwen2.5-coder-1.5b.json`, `results/2026-10-22_step3_promptB_qwen2.5-coder-1.5b.json`, and their 300-row prediction JSONL files.

### 2026-10-23 | Step 4 | First Prompt-B QLoRA adapter run
- What happened: trained a QLoRA adapter for two epochs on Kaggle, then evaluated it on the frozen 300-example `dev_eval` split.
- Decision / change: used the frozen Prompt B and starting QLoRA settings (rank 16, alpha 32, dropout 0.05, 4-bit NF4, fp16, learning rate 2e-4, effective batch size 16, seed 42). The SFT builder kept 2963 of 3000 examples and dropped 37 above the 2048-token limit.
- Why: Prompt B was frozen after Step 3; over-length examples cannot fit the frozen maximum sequence length.
- Effect on results (if known): adapter EX 59.7% (179/300), valid SQL 85.0%; Prompt-B base EX 59.3% (178/300), valid SQL 77.7%. The EX difference is one example and should not be over-claimed. Adapter errors: 7 `no_such_table`, 38 `no_such_column`, 76 `wrong_result`, 0 `other`.
- Files changed / result files: `results/2026-10-23_step4_adapter_qwen2.5-coder-1.5b.json`, `results/predictions/step4_adapter_qwen2.5-coder-1.5b.jsonl`; adapter saved as a private Kaggle dataset with `adapter_info.json` (commit `c2274e4`). Evaluation emitted a PEFT compatibility warning that `kasa_config` was ignored by PEFT 0.20.0; the core LoRA adapter loaded and evaluated successfully.

### 2026-10-24 | Step 5 | 7B Prompt-B QLoRA run completed
- What happened: ran the frozen Prompt-B baseline and a two-epoch 7B QLoRA adapter run on Kaggle, then evaluated the adapter on the fixed 300-example `dev_eval` split.
- Decision / change: used `Qwen/Qwen2.5-Coder-7B-Instruct`, max sequence length 1024, batch size 1, gradient accumulation 16, gradient checkpointing, and paged 8-bit AdamW. The SFT builder retained 2527 training examples at the shorter length.
- Why: these memory-saving settings allowed the 7B model to fit on the Kaggle GPU without an out-of-memory fallback.
- Effect on results (if known): Prompt-B baseline EX 72.6% (217/299), valid SQL 95.3%; one gold query timed out and was excluded during this Kaggle run. The verified adapter result was EX 72.7% (218/300), valid SQL 94.0%, with 0 `no_such_table`, 16 `no_such_column`, 64 `wrong_result`, and 2 `other` errors. The baseline and adapter denominators differ, so the small EX difference is not a conclusive training gain.
- Files changed / result files: `results/2026-10-24_step5_promptB_qwen2.5-coder-7b.json`, `results/predictions/step5_promptB_qwen2.5-coder-7b.jsonl`, `results/2026-10-24_step5_adapter_qwen2.5-coder-7b.json`, `results/predictions/step5_adapter_qwen2.5-coder-7b.jsonl`; both adapters are preserved together in the private Kaggle dataset `stage-a-sql-adapters`, containing only the required adapter model, configuration, and project metadata files.

---

## Open questions

- [x] Confirm the Spider license is OK for this use (CC BY-SA 4.0).
- [x] Final dev model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (1.5B).
- [x] Frozen prompt: B (chosen from the full Step 3 aggregate results).

---

## Run summary table

Fill this in as runs finish. Use `correct/total` next to the percentage.

| Date | Run name | Model | Prompt | Adapter | EX | Valid SQL | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-22 | step3_promptA_qwen2.5-coder-1.5b | Qwen/Qwen2.5-Coder-1.5B-Instruct | A | — | 12.7% (38/300) | 15.0% | Kaggle run; copy result artifacts locally |
| 2026-10-22 | step3_promptB_qwen2.5-coder-1.5b | Qwen/Qwen2.5-Coder-1.5B-Instruct | B | — | 59.3% (178/300) | 77.7% | Kaggle run; frozen prompt |
| 2026-10-23 | step4_adapter_qwen2.5-coder-1.5b | Qwen/Qwen2.5-Coder-1.5B-Instruct | B | QLoRA adapter | 59.7% (179/300) | 85.0% | 2963 training examples retained at 2048 tokens |
| 2026-10-24 | step5_promptB_qwen2.5-coder-7b | Qwen/Qwen2.5-Coder-7B-Instruct | B | — | 72.6% (217/299) | 95.3% | One gold query timed out and was excluded |
| 2026-10-24 | step5_adapter_qwen2.5-coder-7b | Qwen/Qwen2.5-Coder-7B-Instruct | B | QLoRA adapter | 72.7% (218/300) | 94.0% | 2527 training examples retained at 1024 tokens; 7B fit on Kaggle |
