# TASKS.md: Stage A Task List (Steps 1 to 5 only)

**Rules for the agent**
- Do **one step at a time**. After a step's "Done when" is met, **stop and report**. Wait for the user to say "go on".
- Read `AGENTS.md` and `docs/PROTOCOL.md` first.
- Everything must have a **smoke test** that runs on a laptop CPU with a tiny setup.
- Real GPU runs are done by the user on Kaggle (see `docs/KAGGLE.md`). The agent prepares the code and tells the user exactly what to run there.
- Out of scope: any agents, Olist/e-commerce, synthetic data, chat interface. Ask before going outside the list.

Mark tasks with `[x]` when done.

---

## Step 1: Set up (Days 1 to 2)

**Goal:** a clean repo and a notebook that can load Spider.

- [x] **1.1** Create the folders from `AGENTS.md` (section 5) and a `.gitignore` that excludes: data files, model weights, `__pycache__`, `.env`, `results/predictions/` large files if needed.
- [x] **1.2** Create `requirements.txt` with the libraries listed in `AGENTS.md` (pin versions later, after the first successful Kaggle run).
- [x] **1.3** Create `configs/base.yaml` with: seed, paths (with a `DATA_ROOT` setting that can point to local folder or `/kaggle/input/...`), model name, split sizes, SQL timeout.
- [x] **1.4** Write `data/prepare_spider.py`:
  - reads `train_spider.json` and `dev.json` from `DATA_ROOT`,
  - samples `train_sft` (3000) and `dev_eval` (300) with seed 42,
  - saves them to `data/splits/train_sft.json` and `data/splits/dev_eval.json`,
  - prints counts and one example.
- [x] **1.5** Write `harness/schema.py` with a function `get_schema(db_path) -> str` that returns the `CREATE TABLE` statements from `sqlite_master`.
- [x] **1.6** Write `docs/LOG.md` first entry: date, "Step 1 started", settings chosen.
- [x] **1.7** Add a smoke test: a script `scripts/smoke_load.py` that loads the splits, builds the schema for one example, and prints the example.

**Ask the user:** where the Spider files are (local path), and confirm the license is OK for this use.

**Done when:** a Kaggle notebook (or local run) with Spider available prints one example with its schema, using `data/prepare_spider.py` and `harness/schema.py`.

**Then stop and report.**

---

## Step 2: Build the scoring tool (Days 3 to 5)

**Goal:** a trusted EX scorer. Build and test it **before** any model is used.

- [x] **2.1** `harness/executor.py`
  - `run_sql(db_path, sql, timeout_s) -> Result` where `Result` has: `ok`, `rows`, `error_type`, `error_message`.
  - Open SQLite read-only (`uri=True`, `mode=ro`).
  - Enforce the time limit (for example with a progress handler or a separate process).
  - Allow only one `SELECT`/`WITH` statement.
  - Map errors to the types in `PROTOCOL.md`.
- [x] **2.2** `harness/scorer.py`
  - `compare_results(gold_rows, pred_rows, ordered: bool) -> bool` using the rules in `PROTOCOL.md` (unordered multiset unless gold has `ORDER BY`; round floats; `1 == 1.0`).
  - `score_example(db_path, gold_sql, pred_sql) -> dict` with: `correct`, `valid`, `error_type`, `gold_failed`.
  - `score_dataset(examples, predictions) -> summary` with: EX, valid rate, error counts, number correct / total, number of `gold_failed`.
- [x] **2.3** Unit tests in `harness/test_scorer.py` (pytest). Include tests for:
  - gold used as prediction gives correct,
  - same rows in a different order is correct when no `ORDER BY`,
  - different order is **wrong** when gold has `ORDER BY`,
  - a syntax error is invalid,
  - a missing table is `no_such_table`,
  - a long-running query hits the timeout,
  - a non-SELECT statement (for example `DROP TABLE`) is rejected and never run,
  - float rounding (`0.1+0.2` style cases).
- [x] **2.4** `harness/check_gold.py`: runs the **gold SQL as the prediction** for all of `dev_eval` and prints the summary.
- [x] **2.5** Update `docs/LOG.md` with any problem found (for example, gold queries that fail). No gold-query problems were found.

**Done when:** `harness/check_gold.py` reports **100% EX** on `dev_eval` (300 examples), apart from any `gold_failed` cases, which are listed and explained. All unit tests pass.

**Then stop and report.**

---

## Step 3: Prompt-only baselines (Days 5 to 8)

**Goal:** EX numbers for the base model with no training, using two prompts.

- [x] **3.1** `harness/prompts.py`
  - `build_prompt_a(schema, question)` and `build_prompt_b(schema, question)` exactly as in `PROTOCOL.md`.
  - The 3 fixed examples for Prompt B use a made-up schema (not Spider).
  - `clean_sql_output(text)` as in `PROTOCOL.md`.
- [x] **3.2** `harness/generate.py`
  - Loads a model (Hugging Face, with optional 4-bit loading) and generates SQL for a list of prompts.
  - Greedy decoding, `max_new_tokens=256`.
  - Batches prompts to save time. Handles left padding correctly.
  - Has a `--smoke` option: tiny model or 5 examples, runs on CPU.
- [x] **3.3** `harness/run_baseline.py`
  - Arguments: config file, `--prompt A|B`, `--model`, `--limit N`.
  - For each example in `dev_eval`: build prompt, generate, clean, score.
  - Writes the result JSON to `results/` and predictions to `results/predictions/` (formats in `PROTOCOL.md`).
- [x] **3.4** `harness/compare.py`: reads result JSON files and prints a small table (run name, EX with `correct/total`, valid rate).
- [x] **3.5** Tell the user exactly which Kaggle commands to run for: Prompt A and Prompt B on the dev model.
- [ ] **3.6** After the user returns the results, help read the error types and **freeze one prompt** (record the choice and reason in `PROTOCOL.md` section 4 and `docs/LOG.md`). Do **not** tune the prompt by reading `dev_eval` errors one by one.

**Done when:** a results table exists with Prompt A and Prompt B (EX, valid rate, error counts) on `dev_eval`, and one prompt is frozen.

**Then stop and report.**

---

## Step 4: First manual QLoRA run on the small model (Days 8 to 14)

**Goal:** train one adapter and compare it with the prompt-only results.

- [ ] **4.1** `train/make_sft_data.py`
  - Reads `train_sft.json`, builds `{prompt, completion}` pairs using the **frozen prompt** and the schema from SQLite.
  - Saves a JSONL file.
  - Prints token length statistics. Reports how many examples are longer than the max length (default 2048) and drops them, with the count logged.
- [ ] **4.2** `configs/train_qlora.yaml` with the starting values from `PROTOCOL.md` section 6.
- [ ] **4.3** `train/train_qlora.py`
  - 4-bit NF4 base model, LoRA on all linear layers, fp16.
  - Loss on completion tokens only.
  - Saves checkpoints every N steps to a configurable output folder (so it can point to `/kaggle/working`).
  - Can **resume** from the latest checkpoint.
  - Logs training loss to a file.
  - Has `--smoke` mode (tiny model, 20 examples, 2 steps) that runs on CPU.
- [ ] **4.3b** Write the adapter's training settings into a small `adapter_info.json` next to the adapter (model, data file, settings, seed, git commit).
- [ ] **4.4** `train/eval_adapter.py`
  - Loads the base model plus the adapter, runs the same flow as `run_baseline.py` on `dev_eval` with the frozen prompt.
  - Writes the same result JSON format (with `adapter` filled in).
- [ ] **4.5** Update `harness/compare.py` so one table shows: Prompt A, Prompt B, and the adapter.
- [ ] **4.6** Tell the user the exact Kaggle steps: where to put the data, how to start training, how to save the adapter as a Kaggle dataset before the session ends, how to resume.
- [ ] **4.7** After results come back, help check: training loss curve, EX versus baseline, error types that changed. Log any manual decision in `docs/LOG.md`.

**Done when:** a table compares the prompt-only results with the fine-tuned adapter on `dev_eval`, and every setting for the run is written down in a result file and `adapter_info.json`.

**Then stop and report.**

---

## Step 5: Try the 7B model (Days 14 to 21)

**Goal:** see if a 7B QLoRA run fits on a 16 GB Kaggle GPU. Time-box this step.

- [ ] **5.1** `configs/train_qlora_7b.yaml`: same as the dev config but with a 7B model, shorter max length (try 1024 to 1536), batch size 1, higher gradient accumulation, gradient checkpointing on.
- [ ] **5.2** Add optional memory-saving features to `train/train_qlora.py` behind config flags (gradient checkpointing, paged 8-bit optimizer). Only add Unsloth if the standard setup runs out of memory, and ask the user first.
- [ ] **5.3** Tell the user to first run the **7B prompt-only baseline** on Kaggle (Prompt frozen), then try the training run.
- [ ] **5.4** If it runs out of memory, try the steps in order: shorter length, gradient checkpointing, smaller LoRA rank, then Unsloth. Log each attempt in `docs/LOG.md`.
- [ ] **5.5** **Time limit:** if it does not fit after a few tries, stop. Write the decision in `docs/LOG.md`: keep the small model for development and plan the 7B for final runs with other hardware or a trimmed setup.

**Done when:** there is either a 7B adapter result (same table format), or a written decision on how the 7B will be handled.

**Then stop and report.** Stage A is finished here. Do not start the next stage (e-commerce data, agents) until the user asks.

---

## Notes for the agent

- Do not skip the smoke tests. They save Kaggle GPU hours.
- Keep all numbers in `results/` JSON files, never only in chat.
- When the user reports a Kaggle error, ask for the error text and the notebook cell, then fix the code locally and tell the user which files to copy again.
