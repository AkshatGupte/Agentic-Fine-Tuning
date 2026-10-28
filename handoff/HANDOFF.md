# Stage A Handoff

## Current status

Stage A is complete. Do not begin the next stage (organization-specific data, agents, synthetic data, deployment, or chat UI) unless the user explicitly asks.

Read `AGENTS.md`, `docs/PROTOCOL.md`, and `docs/TASKS.md` before making any new project change. The protocol settings are frozen unless the user approves and the decision is logged.

## What was built

- Fixed Spider splits: 3,000 training examples and 300 development-evaluation examples, sampled with seed 42.
- SQLite read-only executor and execution-accuracy scorer with unit tests.
- Prompt-only baseline harness using two prompts; Prompt B is frozen.
- Prompt-B SFT data builder and QLoRA trainer/evaluator.
- 1.5B and 7B QLoRA runs on Kaggle.

## Final results

| Run | EX | Valid SQL |
|---|---:|---:|
| 1.5B, Prompt A | 12.7% (38/300) | 15.0% |
| 1.5B, Prompt B | 59.3% (178/300) | 77.7% |
| 1.5B, Prompt B + QLoRA | 59.7% (179/300) | 85.0% |
| 7B, Prompt B | 72.6% (217/299) | 95.3% |
| 7B, Prompt B + QLoRA | 72.7% (218/300) | 94.0% |

Interpretation: fixed few-shot prompting and the larger base model drove the meaningful gains. The QLoRA adapters improved SQL validity for the 1.5B model, but did not show a meaningful execution-accuracy improvement in either single adapter run. The 7B prompt-only baseline had one gold-query timeout, so its denominator is 299 rather than 300; do not over-interpret its one-answer difference from the adapter result.

## Preserved assets

- Full run results are in `results/`.
- Per-example predictions are in `results/predictions/`.
- Both minimal adapter packages are saved in the user's private Kaggle dataset `stage-a-sql-adapters`.
- The dataset contains only these required files for each adapter:
  - `adapter_model.safetensors`
  - `adapter_config.json`
  - `adapter_info.json`

Kaggle paths used during the runs:

```text
Spider data: /kaggle/input/datasets/akshatgupte/spider-data-2
Adapters:    /kaggle/input/datasets/akshatgupte/stage-a-sql-adapters/<adapter-folder>
```

## Important frozen configuration

- Development model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Larger comparison model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Prompt: B
- Seed: 42
- Metric: execution accuracy (EX)
- Generation: greedy decoding, `max_new_tokens=256`
- SQL execution: read-only SQLite and 10-second query timeout

## Repository state at handoff

The only expected uncommitted changes are deletion of the redundant Step 5 smoke artifacts:

```text
results/2026-10-24_step5_promptB_qwen2.5-coder-7b_smoke.json
results/predictions/step5_promptB_qwen2.5-coder-7b_smoke.jsonl
```

Commit those deletions together with this handoff file if the user wants the repository clean. Check `git status` first rather than assuming the state has not changed.

## Useful files

- `AGENTS.md` — project scope and working rules.
- `docs/PROTOCOL.md` — frozen experiment settings.
- `docs/TASKS.md` — all Stage A items are marked complete.
- `docs/LOG.md` — decisions, settings, and result summary.
- `docs/KAGGLE.md` — Kaggle operating notes.
- `harness/` — scoring, prompt construction, baseline runner.
- `train/` — SFT builder, QLoRA trainer, adapter evaluator.

