# AGENTS.md: Project Instructions for the Coding Agent

> If your agent is Claude Code, rename or copy this file to `CLAUDE.md`.
> Other agents (Copilot, Cursor, etc.) can read `AGENTS.md` directly or you can paste it into their rules file.

## 1. What this project is

A final-year project: an agentic framework that fine-tunes a small language model (under 10B parameters) to turn natural language into SQL for a specific organization.

The long-term idea is to compare:
- the same base model **with a prompt only** (schema put in the prompt), against
- the same base model **with a fine-tuned LoRA adapter**.

The main score is **Execution Accuracy (EX)**: a predicted query is correct if running it gives the same result as running the correct query.

## 2. CURRENT SCOPE (very important)

We are doing **only the first part of the plan**, called **Stage A**. It is a pipeline test on the Spider dataset.

Work only on the tasks in `docs/TASKS.md` (Steps 1 to 5).

### In scope
- Project setup and repo folders
- A scoring tool (SQL runner + EX scorer)
- Prompt-only baselines on a small model
- One manual QLoRA training run on a small model
- A try at a 7B model

### Out of scope (do NOT build, do NOT suggest building yet)
- Any "agents" (training agent, evaluation agent, gatekeeper, data agents)
- The e-commerce / Olist dataset and tenant variants
- Reverse labeling or synthetic data generation
- The chat interface or the analyst feedback loop
- Any deployment or serving code

If a task seems to need something out of scope, **stop and ask the user**.

## 3. How to work (rules)

1. **One step at a time.** Do the tasks of one step in `docs/TASKS.md`. When the step's "Done when" is met, **stop and report**. Wait for the user to say "go on" before starting the next step.
2. **Read first.** Before coding, read `docs/PROTOCOL.md` (frozen settings) and `docs/TASKS.md`.
3. **Do not change frozen settings** (prompt template, metric, seeds, data splits) without asking. If you think one should change, explain why and wait.
4. **Keep code simple.** Short scripts, clear names, comments for anything not obvious. No heavy frameworks beyond the ones listed below.
5. **Everything must be configurable.** Model name, paths, seeds, and training settings go in `configs/*.yaml`, not hard-coded.
6. **Make runs reproducible.** Set seeds. Every run writes a JSON result file to `results/` with its settings and scores (see `docs/PROTOCOL.md`).
7. **Test small first.** Every script must have a quick "smoke test" mode (a few examples, tiny settings) that runs on a laptop CPU, because the real runs happen on Kaggle GPUs.
8. **Log decisions.** When you or the user make a manual choice (change a setting, drop bad data, fix a bug that affected scores), add a line to `docs/LOG.md`.
9. **Do not invent results.** Never write fake numbers. If a run was not executed, say so.
10. **Ask before downloading data or models** and before installing big packages. Check dataset licenses when relevant.
11. **No secrets in the repo.** API tokens (Hugging Face, Kaggle) go in environment variables or Kaggle secrets, never in files.

## 4. Environment

- Code is written and smoke-tested **locally in VS Code** (likely no GPU).
- Real training and evaluation runs happen on **Kaggle notebooks**. See `docs/KAGGLE.md` for how code moves there and the Kaggle limits to respect.
- Language: Python 3.10+.
- Main libraries: `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, `accelerate`, `pyyaml`, `pytest`. Optional: `unsloth` (only for the 7B memory problem in Step 5).
- Database: SQLite (Spider uses SQLite files).

## 5. Folder layout

```
slm-sql-project/
  AGENTS.md              # this file
  docs/
    PROTOCOL.md          # frozen settings
    TASKS.md             # step-by-step tasks
    KAGGLE.md            # how to run on Kaggle
    LOG.md               # decisions and run notes
  configs/               # yaml settings for each script
  data/                  # dataset prep scripts (data files are NOT committed)
  harness/               # SQL runner, EX scorer, prompts, baseline runner
  train/                 # SFT data builder, QLoRA trainer, adapter evaluator
  results/               # JSON results per run (small files only)
  requirements.txt
```

## 6. Code style

- Type hints on public functions.
- Small functions, one job each.
- Use `pathlib`, not string paths.
- Use `logging`, not stray `print`, except for final result tables.
- Add a short docstring to every script that says what it does and how to run it.

## 7. Definition of done for any task

A task is done only when:
- the code runs (smoke test passes),
- the "Done when" line for that step is met,
- the result is saved in `results/` (if it produces numbers),
- `docs/LOG.md` is updated if any decision was made.

Then **stop and report** in a short message: what was built, how to run it, what the result is, anything unclear.
