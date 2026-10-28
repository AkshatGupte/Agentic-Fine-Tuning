# Stage B Tasks: EHRSQL MIMIC-III

Stage B evaluates the existing text-to-SQL pipeline on EHRSQL's MIMIC-III
subset. Complete one step at a time. After each step's **Done when** condition
is met, stop and report before starting the next step.

Stage A files and results remain unchanged. Do not start eICU, another dataset,
agents, synthetic data, a chat UI, or deployment work without an explicit user
request.

## Step B0: Register the dataset and freeze the protocol

Goal: establish a reproducible, safe Stage B experiment definition before
changing code.

- [x] Download EHRSQL v1.5.0 and use the MIMIC-III subset only.
- [x] Verify the provided MIMIC-III SQLite database locally and on Kaggle.
- [x] Record the official train/validation/test split sizes.
- [x] Create `docs/STAGE_B_PROTOCOL.md` and this task list.
- [x] Record the dataset decision in `docs/LOG.md`.

**Done when:** EHRSQL MIMIC-III is accessible locally and in the user's private
Kaggle dataset, and the Stage B protocol records the data, splits, metrics, and
abstention rule.

**Then stop and report.**

## Step B1: Adapt and smoke-test the data and scorer

Goal: load EHRSQL safely and score both SQL generation and abstention behavior.

- [x] Add a configurable EHRSQL data loader for the official split files.
- [x] Validate required fields, source-file split labels, and database paths.
- [x] Extend prompt/output handling to use the exact `ABSTAIN` token.
- [x] Extend scoring and result JSON/JSONL records for impossible examples,
  unsafe answers, and unnecessary abstentions.
- [x] Add unit tests for answerable execution, correct abstention, unsafe SQL
  on an impossible question, and abstention on an answerable question.
- [x] Add a CPU-friendly smoke mode that loads a few examples and executes
  answerable gold SQL read-only.

**Done when:** unit tests and the local/Kaggle smoke test pass, and the smoke
result is saved under `results/`.

**Then stop and report.**

## Step B2: Prompt-only baselines

Goal: select one EHRSQL prompt from aggregate validation results.

- [x] Implement up to two protocol-compliant prompt candidates.
- [ ] Run each base-model candidate on the full validation split.
- [ ] Save per-example predictions and all Stage B metrics.
- [ ] Compare aggregate results only; freeze one prompt and log the reason.
- [ ] Run the frozen-prompt base-model baseline once on the untouched test
  split.

**Done when:** validation results select a frozen prompt, and a corresponding
untuned baseline result exists on test.

**Then stop and report.**

## Step B3: First EHRSQL QLoRA run

Goal: fine-tune the development model on the EHRSQL training split and compare
it fairly with the frozen-prompt baseline.

- [ ] Build SFT data from answerable training examples using the frozen prompt.
- [ ] Report token lengths and dropped examples before training.
- [ ] Run one QLoRA training job with configured, recorded settings.
- [ ] Save the adapter and `adapter_info.json` privately on Kaggle.
- [ ] Evaluate the adapter once on the untouched test split.
- [ ] Compare answerable EX, valid SQL, abstention accuracy, unsafe answers,
  and unnecessary abstentions with the base-model test baseline.

**Done when:** the baseline and adapter results use the same frozen prompt,
test split, generation settings, and metric definitions; all settings and
artifacts are recorded.

**Then stop and report.**

## Step B4: Optional 7B comparison

Goal: repeat the final frozen-prompt comparison with the larger model only if
the user requests it after reviewing the 1.5B results.

- [ ] Run a 7B frozen-prompt base-model test baseline.
- [ ] Decide with the user whether a 7B QLoRA run is justified.
- [ ] If approved, train and evaluate one adapter with documented memory
  settings.

**Done when:** either the 7B comparison is completed or a documented user
decision defers it.

**Then stop and report.**
