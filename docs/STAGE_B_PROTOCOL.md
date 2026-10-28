# Stage B Protocol: EHRSQL MIMIC-III

This document fixes the initial Stage B experiment settings. It is separate
from `docs/PROTOCOL.md`, which remains the frozen Stage A (Spider) protocol.

Change a setting here only on purpose, and record the reason in `docs/LOG.md`.

## 1. Objective

Test whether prompt design and QLoRA fine-tuning improve text-to-SQL on a
single realistic domain: electronic health records. Stage B uses EHRSQL's
MIMIC-III subset. It is not a clinical decision-support system and must not be
used for clinical analysis or patient care.

## 2. Dataset and license

- Dataset: EHRSQL v1.5.0, MIMIC-III subset only.
- Source repository: `https://github.com/glee4810/EHRSQL`.
- Dataset/repository license: CC BY 4.0.
- Database: EHRSQL's provided, preprocessed `mimic_iii.sqlite` database.
- Raw data is local-only and ignored by Git. On Kaggle it is stored in a
  private dataset.

The database is for benchmark execution only. Do not treat generated answers
as medical advice or use the database for clinical analysis.

## 3. Frozen splits

Use the official EHRSQL split files unchanged. Do not resample or mix examples
between these files.

| Split | File | Total | Answerable | Impossible | Use |
|---|---|---:|---:|---:|---|
| Train | `mimic_iii/train.json` | 9,318 | 9,318 | 0 | SFT training only |
| Validation | `mimic_iii/valid.json` | 1,122 | 760 | 362 | prompt choice, smoke checks, development evaluation |
| Test | `mimic_iii/test.json` | 1,786 | 1,198 | 588 | final, untouched evaluation |

The supplied `split` field must agree with the source file. Treat a mismatch as
bad input and stop the run.

## 4. Models and generation

| Role | Model |
|---|---|
| Development model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Larger comparison model (later) | `Qwen/Qwen2.5-Coder-7B-Instruct` |

- Use greedy decoding (`do_sample=False`) and `max_new_tokens=256` for
  baseline and adapter comparisons.
- Use global seed 42 for any random operation.
- Model, adapter path, data root, timeout, and all training parameters must be
  configurable rather than hard-coded.

## 5. SQL execution safety

- Open SQLite databases read-only with `mode=ro` and `uri=True`.
- Allow exactly one `SELECT` or `WITH` statement.
- Use a fixed 10-second query timeout unless a recorded decision changes it.
- Build the schema from `sqlite_master`, not `tables.json`, for generation.
- Preserve Stage A execution comparison rules for answerable questions:
  respect gold `ORDER BY`; otherwise compare unordered multisets; round floats
  to six decimal places; and treat `1` and `1.0` as equal.

If a gold query fails or times out, mark it `gold_failed`, exclude it from
execution accuracy, and report it separately.

## 6. Answerable and impossible questions

EHRSQL marks unsupported questions with `is_impossible: true`. These are a
separate reliability task, not SQL-execution examples.

- For answerable examples, the model must return one SQL query.
- For impossible examples, the model must return exactly the token `ABSTAIN`.
- `ABSTAIN` is not passed to SQLite.
- A query generated for an impossible question is an `unsafe_answer`.
- `ABSTAIN` for an answerable question is an `unnecessary_abstention` and is
  invalid SQL for that example.

The prompt format and output cleaner must implement this rule before any full
baseline is run.

## 7. Metrics

Report these metrics for every Stage B run:

| Metric | Definition |
|---|---|
| Answerable EX | Correct executions / answerable examples whose gold SQL ran successfully |
| Valid SQL rate | Executable SQL predictions / answerable examples |
| Impossible abstention accuracy | Correct `ABSTAIN` outputs / impossible examples |
| Unsafe answers | SQL outputs for impossible examples |
| Unnecessary abstentions | `ABSTAIN` outputs for answerable examples |
| Gold failures | Answerable examples whose gold SQL failed or timed out |

Report counts beside percentages. Never combine answerable EX and abstention
accuracy into an unlabelled single number.

## 8. Prompt protocol

- Create at most two general EHRSQL prompt candidates. They may mention the
  `ABSTAIN` output rule, but must not provide medical advice or schema-specific
  hand-written answers.
- Compare prompt candidates only on the full validation split using aggregate
  metrics.
- Freeze one prompt after validation comparison and use it unchanged for SFT
  construction and all adapter/test evaluations.
- Do not inspect individual validation errors to iteratively tune the prompt.

## 9. Training protocol

- Train only on answerable examples from `train.json`.
- Build each SFT item from the frozen prompt, SQLite-derived schema, question,
  gold SQL, and end-of-text token.
- Apply loss to SQL completion tokens only.
- Start with Stage A QLoRA settings: 4-bit NF4, LoRA rank 16, alpha 32,
  dropout 0.05, learning rate `2e-4`, cosine schedule, 3% warmup, fp16, and
  seed 42.
- Log and justify any examples dropped for sequence length.

## 10. Reproducibility and results

- Every run writes a JSON result file under `results/` and one prediction JSONL
  file under `results/predictions/`.
- Include the dataset version, split, answerable/impossible counts, model,
  adapter, prompt name, seed, generation settings, SQL timeout, and Git commit.
- Keep model weights and datasets out of Git. Save adapters as a private Kaggle
  dataset with `adapter_info.json`.
