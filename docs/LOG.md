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

---

## Open questions

- [x] Confirm the Spider license is OK for this use (CC BY-SA 4.0).
- [x] Final dev model: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (1.5B).
- [ ] Frozen prompt: A or B? (decide at the end of Step 3)

---

## Run summary table

Fill this in as runs finish. Use `correct/total` next to the percentage.

| Date | Run name | Model | Prompt | Adapter | EX | Valid SQL | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
