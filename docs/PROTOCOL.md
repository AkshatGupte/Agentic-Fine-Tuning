# PROTOCOL.md: Frozen Settings for Stage A

These settings are **frozen**. Change them only on purpose, and write the change in `docs/LOG.md`.

Items marked **[FILL IN]** are decisions the user still has to make. The agent must ask the user, not guess.

---

## 1. Models

| Role | Model (Hugging Face id) | Notes |
|---|---|---|
| Dev model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Fast. Use for building and testing. |
| Dev model (alt) | `Qwen/Qwen2.5-Coder-3B-Instruct` | Use if 1.5B is too weak. |
| Main model | `Qwen/Qwen2.5-Coder-7B-Instruct` | Step 5 only. Must be under 10B parameters. |

**Final dev model choice:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (1.5B)

Check that the model ids still exist on Hugging Face before using them.

---

## 2. Dataset: Spider

- Source: Spider (Yale), licensed under CC BY-SA 4.0. License use for this project is confirmed. On Kaggle, the data will be stored as a private Kaggle dataset.
- Files used: `train_spider.json`, `dev.json`, and the `database/` folder of SQLite files.
- Each example has: `db_id`, `question`, `query` (the correct SQL).
- The database schema is **read directly from the SQLite file** (not from `tables.json`), so the same code can work on any SQLite database later.

### Splits

| Name | Source | Size | Use |
|---|---|---|---|
| `train_sft` | random sample of Spider **train** | 3000 (change only on purpose) | training in Step 4 |
| `dev_eval` | random sample of Spider **dev** | 300 | scoring in Steps 2 to 5 |

- Both samples are picked **once** with seed 42, then **saved to disk** as JSON files (`data/splits/train_sft.json`, `data/splits/dev_eval.json`) so everyone uses the exact same examples.
- Train and dev databases are different in Spider, so there is no database overlap. Do not mix them.
- `dev_eval` is never used for training or for tuning the prompt by looking at errors one by one.

---

## 3. Metric

**Execution Accuracy (EX)**

For each example:
1. Run the correct SQL on the database. Get result A.
2. Run the predicted SQL on the same database. Get result B.
3. The example is correct if A equals B.

Comparison rule:
- If the correct SQL has `ORDER BY`, compare the rows **in order**.
- Otherwise compare the rows as an **unordered multiset** (sort the rows, keep duplicates).
- Compare values as they come from SQLite. Treat `1` and `1.0` as equal. Round floats to 6 decimals.

Also record, for every run:
- **Valid SQL rate**: share of predictions that run without error.
- **Error types** for failures: `syntax_error`, `no_such_table`, `no_such_column`, `timeout`, `wrong_result`, `empty_prediction`, `other`.

Safety for running SQL:
- Open the database **read-only** (`file:...?mode=ro` with `uri=True`).
- Time limit per query: **10 seconds** (change only on purpose).
- Only run a single statement. Reject anything that is not a `SELECT`/`WITH` query.

If the **correct** SQL itself fails or times out, mark the example `gold_failed`, exclude it from EX, and report how many there were.

---

## 4. Prompt template (frozen after Step 3)

Two prompts are compared in Step 3. After that, **one** format is frozen and used for training and evaluation.

### Prompt A (plain)

```
You are an expert SQLite developer. Given the database schema and a question, write one SQLite query that answers the question. Return only the SQL.

### Schema
{schema}

### Question
{question}

### SQL
```

### Prompt B (with examples)

Same as Prompt A, but with 3 short fixed examples placed before the real question. The examples must:
- use a **small made-up schema**, not Spider data,
- stay **the same** for every question,
- be written once and saved in `harness/prompts.py`.

### Schema format

Use the `CREATE TABLE` statements from SQLite (`sqlite_master`), one per table. Keep the original order. Do not add sample rows in Stage A.

### Output cleaning

Models may wrap SQL in markdown fences or add text. The cleaning function must:
- remove ```` ```sql ```` fences,
- keep the first statement only,
- strip whitespace.

**Chosen frozen prompt after Step 3:** `B`. Prompt B is used for all subsequent training and evaluation. It was selected from the two full `dev_eval` baseline runs; see `docs/LOG.md` for the aggregate comparison.

---

## 5. Generation settings (for scoring)

- Greedy decoding (`do_sample=False`), `max_new_tokens=256`, no sampling.
- Same settings for baseline and adapter runs.

---

## 6. Training settings (starting values for Step 4)

These are starting suggestions. Adjust only with a note in `docs/LOG.md`.

| Setting | Value |
|---|---|
| Method | QLoRA (4-bit NF4 base model + LoRA adapter) |
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | all linear layers of the transformer |
| Learning rate | 2e-4 (cosine schedule, 3% warmup) |
| Epochs | 2 to 3 |
| Max sequence length | 2048 |
| Precision | **fp16** (Kaggle T4/P100 do not support bf16) |
| Batch size | small (1 to 4) with gradient accumulation to reach an effective batch of 16 |
| Seed | 42 |
| Loss on | the **SQL answer tokens only** (not the prompt) |

---

## 6b. Training data format

For each example in `train_sft`, build text as:

```
<frozen prompt filled with schema and question> <correct SQL><end-of-text token>
```

Use the same prompt as the one used at evaluation time. Train only on the SQL part.

---

## 7. Seeds

- Global seed: **42** for sampling, training, and anything random.
- Set Python, NumPy, and PyTorch seeds at the start of every script.

---

## 8. Result file format

Every run writes one JSON file to `results/`, named `{date}_{step}_{short_name}.json`:

```json
{
  "run_name": "step3_promptA_qwen1.5b",
  "date": "YYYY-MM-DD",
  "step": 3,
  "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "adapter": null,
  "prompt": "A",
  "split": "dev_eval",
  "n_examples": 300,
  "n_gold_failed": 0,
  "ex": 0.0,
  "valid_sql_rate": 0.0,
  "error_counts": {"syntax_error": 0, "no_such_table": 0, "no_such_column": 0, "timeout": 0, "wrong_result": 0, "empty_prediction": 0, "other": 0},
  "settings": {"seed": 42, "max_new_tokens": 256, "greedy": true},
  "git_commit": "abc1234",
  "notes": ""
}
```

Also save per-example predictions to `results/predictions/{run_name}.jsonl` (question, gold SQL, predicted SQL, correct yes/no, error type) so errors can be studied later.

Do not commit large files (model weights, datasets). Adapters are saved on Kaggle, not in git.

---

## 9. Comparison rule

To say "training helped", compare on the **same** `dev_eval` set, with the **same** prompt format and generation settings. Report EX for:
- Prompt A, base model
- Prompt B, base model
- Fine-tuned adapter (frozen prompt)

Only 300 examples means a few points of difference can be noise. Report the count of correct answers next to the percentage (for example `201/300`) and do not over-claim small gaps.
