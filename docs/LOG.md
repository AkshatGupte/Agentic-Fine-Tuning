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

### 2026-10-28 | Stage B | EHRSQL MIMIC-III selected and verified
- What happened: user selected EHRSQL v1.5.0's MIMIC-III subset for Stage B, downloaded the provided preprocessed SQLite database, and verified read-only SQLite access locally and in a private Kaggle dataset.
- Decision / change: preserve EHRSQL's official train/valid/test files without resampling; use MIMIC-III only for the initial Stage B work. Treat `is_impossible` questions as a separate abstention task using the exact output token `ABSTAIN`.
- Why: the dataset is a realistic healthcare-domain text-to-SQL benchmark that fits the existing SQLite execution harness. Official splits prevent train/development/test leakage, and separate abstention metrics prevent unsafe SQL output on unsupported questions from being hidden by execution accuracy.
- Effect on results (if known): no Stage B baseline or training result yet. Split counts are train 9,318 answerable; validation 760 answerable and 362 impossible; test 1,198 answerable and 588 impossible.
- Files changed / result files: `docs/STAGE_B_PROTOCOL.md`, `docs/STAGE_B_TASKS.md`, `docs/LOG.md`; no result file.

### 2026-10-28 | Stage B1 | EHRSQL loader, abstention scorer, and smoke test
- What happened: added a validated EHRSQL MIMIC-III split loader, separate Stage B scorer, exact `ABSTAIN` output handling, unit tests, and a CPU smoke script. The smoke test used gold SQL for three answerable validation records and `ABSTAIN` for one impossible validation record.
- Decision / change: kept Stage A scorer and prompt code unchanged; Stage B behavior lives in EHRSQL-specific modules. Impossible records are never passed to SQLite.
- Why: preserve frozen Stage A behavior while making EHRSQL abstention safety metrics explicit and testable.
- Effect on results (if known): smoke answerable EX 100.0% (3/3), valid SQL 100.0%, abstention accuracy 100.0% (1/1), unsafe answers 0, unnecessary abstentions 0. This is an oracle smoke test, not a model result.
- Files changed / result files: `data/ehrsql.py`, `harness/ehrsql_scorer.py`, `harness/ehrsql_output.py`, `harness/test_ehrsql.py`, `scripts/smoke_ehrsql.py`, `configs/ehrsql.yaml`, `results/2026-10-28_stageb1_ehrsql_mimic_iii_smoke.json`, `results/predictions/stageb1_ehrsql_mimic_iii_smoke.jsonl`.

### 2026-10-28 | Stage B2 | Two unfrozen EHRSQL baseline prompts prepared
- What happened: added two general EHRSQL prompt candidates and a dedicated baseline runner/comparison tool. Candidate A is concise; candidate B additionally tells the model to use only tables and columns in the supplied schema. Both require the exact `ABSTAIN` token for unsupported questions and prohibit medical advice.
- Decision / change: no prompt is frozen yet; the two candidates will be compared only on aggregate metrics from the full official validation split.
- Why: Stage B protocol permits at most two general prompt candidates and requires selection without per-example validation tuning.
- Effect on results (if known): no Stage B model baseline result yet. The runner and all Stage A/Stage B unit tests pass (18 tests).
- Files changed / result files: `harness/ehrsql_prompts.py`, `harness/run_ehrsql_baseline.py`, `harness/compare_ehrsql.py`, `harness/test_ehrsql_prompts.py`, `configs/ehrsql.yaml`; no new model result file.

### 2026-10-28 | Stage B1 | Support flat Kaggle EHRSQL upload layout
- What happened: the first Kaggle smoke run found `mimic_iii.sqlite` at the attached dataset root rather than under `mimic_iii/`.
- Decision / change: the EHRSQL loader now accepts both the official nested layout and a flat Kaggle layout for the SQLite database and official split JSON files.
- Why: Kaggle dataset uploads may flatten the selected files while retaining the same official EHRSQL contents; accepting both paths avoids a storage-layout-only failure.
- Effect on results (if known): no model result changed. All 19 unit tests and the local oracle smoke test passed after the change.
- Files changed / result files: `data/ehrsql.py`, `harness/test_ehrsql.py`, `configs/ehrsql.yaml`, `docs/LOG.md`, `results/2026-10-28_stageb1_ehrsql_mimic_iii_smoke.json`.

### 2026-10-31 | Stage B2 | Fixed EHRSQL gold-query accounting
- What happened: verification of the first Prompt A and Prompt B validation artifacts found prompt-dependent gold-failure counts. The scorer had classified answerable `ABSTAIN` outputs before running their gold SQL, so broken gold queries could be hidden by a model abstention.
- Decision / change: Stage B now executes gold SQL before handling an answerable prediction, and four validation records are excluded identically from every baseline using the saved gold-only preflight IDs in `configs/ehrsql.yaml`.
- Why: gold failures must depend only on the fixed database and gold SQL, never on a model output. The supplied MIMIC-III SQLite database has four gold queries with missing columns: `systolic_bp_lower`, `temperature_lower` (two queries), and `sao2_lower`.
- Effect on results (if known): the valid split has 760 answerable records, 4 fixed gold failures and 756 EX-scored records. The earlier Prompt A/B artifacts remain internally consistent but are invalid for prompt selection and must be retained only as debugging artifacts. The local gold preflight achieved oracle EX 100.0% (756/756); this is not a model result.
- Files changed / result files: `harness/ehrsql_scorer.py`, `harness/check_ehrsql_gold.py`, `harness/run_ehrsql_baseline.py`, `harness/test_ehrsql.py`, `configs/ehrsql.yaml`, `docs/STAGE_B_PROTOCOL.md`, `docs/LOG.md`, `results/2026-10-31_stageb2_ehrsql_mimic_iii_check_gold_valid.json`, `results/predictions/stageb2_ehrsql_mimic_iii_check_gold_valid.jsonl`.

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
