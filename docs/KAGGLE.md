# KAGGLE.md: Running the Project on Kaggle

The code is written and smoke-tested in **VS Code**. The real GPU runs happen on **Kaggle notebooks**. This file explains how to move between the two.

> Kaggle rules change (GPU type, weekly hours, session length, internet access). Check your account and the Kaggle docs for the current limits. The numbers below are from memory and only a guide.

## 1. What to expect on Kaggle (verify)

- GPUs: usually 2 x T4 (16 GB each) or 1 x P100 (16 GB).
- Weekly GPU quota: about 30 hours.
- One session: about 9 to 12 hours maximum.
- You need a **verified phone number** to turn on GPU and internet.
- **`/kaggle/working`** is cleared when the session ends (unless you save a version or output). Do not rely on it for long-term storage.
- **`/kaggle/input`** is read-only. Datasets and models you attach appear here.

## 2. GPU rules that affect the code

- T4 and P100 do **not** support bf16. Use **fp16**.
- A model is loaded on one GPU by default. With 2 x T4, do not assume both are used unless the code is set up for it. Simple plan: use **one GPU**, keep the code simple.
- Long prompts use a lot of memory. Keep `max_seq_length` at 2048 or lower. For 7B, try 1024 to 1536.
- Use 4-bit loading (`bitsandbytes`) for QLoRA.

## 3. How to get the code onto Kaggle

### Option A: Clone from GitHub (easiest)

1. Push the repo from VS Code to GitHub (private repo is fine).
2. In the Kaggle notebook, turn **Internet ON** (Settings panel) and attach the `GITHUB_TOKEN` secret (Add-ons, Secrets).
3. Run the clone cell from **section 8, Cell 1**. It reads the token from Kaggle Secrets and removes it from the saved git settings after cloning.
4. Install only what is needed (section 8, Cell 2). Do not run `pip install -r requirements.txt` on Kaggle unless the agent says it is needed, because it can clash with Kaggle's preinstalled `torch` and `transformers`.
5. After you change code in VS Code and push, update the notebook with the "Updating the code" cell at the end of section 8.

Do not edit code files inside the Kaggle notebook. Edit in VS Code and push, otherwise `git pull` can conflict.

The token should be a **fine-grained, read-only token** limited to this one repo. Never paste a token into a cell, a file, or a chat message.

### Option B: Upload as a Kaggle Dataset (no internet needed)

1. Zip the code folder (without data, weights, or results).
2. Create a Kaggle dataset from the zip.
3. Attach it to the notebook. It appears under `/kaggle/input/<dataset-name>/`.
4. Copy it into the working folder:

```python
!cp -r /kaggle/input/<dataset-name>/repo /kaggle/working/repo
%cd /kaggle/working/repo
```

When code changes, upload a **new version** of the dataset and re-attach it.

## 4. Data on Kaggle

- Spider: upload as a **private Kaggle dataset**. It will appear at `/kaggle/input/<spider-dataset>/`.
- Set `DATA_ROOT` in `configs/base.yaml` (or pass it as an argument) to that path.
- DATA_ROOT = /kaggle/input/datasets/akshatgupte/spider-data-2
- Do not copy the data into git.

## 5. Models on Kaggle

Two ways to get a model:
- **Download from Hugging Face** (needs Internet ON). Store your token in **Kaggle Secrets** as `HF_TOKEN`.
- **Attach the model** from Kaggle's Models / Datasets, then point the config to `/kaggle/input/...`.

Downloaded model files go to the cache and are lost when the session ends, so they download again next time. Attaching a model avoids that.

**Current choice:** download `Qwen/Qwen2.5-Coder-1.5B-Instruct` from Hugging Face at run time (Internet ON). The model is public, so no `HF_TOKEN` is needed. Add `HF_TOKEN` to Kaggle Secrets only if downloads get rate-limited. The files (about 3 GB) download again in each new session.

## 6. Saving your work (very important)

Sessions end and `/kaggle/working` is cleared. So:

1. Save checkpoints often (set `save_steps` in the training config).
2. Write adapters and results to `/kaggle/working/outputs/`.
3. Before the session ends, **save the adapter as a Kaggle Dataset** (use "Save Version" and keep outputs, or create a dataset from the output folder).
4. The GitHub token is **read-only**, so Kaggle cannot push to your repo. To keep results, download the `results/*.json` files (and `results/predictions/`) from the notebook's output, copy them into the VS Code repo, and commit them from there.
5. Training can **resume** from the latest checkpoint. If a session ends mid-run, attach the saved checkpoint as input and use the resume option.

## 7. Suggested notebook layout

One notebook per job keeps things clear:

| Notebook | Job |
|---|---|
| `01_setup_check` | clone repo, install, load Spider, print one example |
| `02_check_gold` | run `harness/check_gold.py` |
| `03_baseline` | run `harness/run_baseline.py` for Prompt A and B |
| `04_train` | run `train/train_qlora.py` |
| `05_eval_adapter` | run `train/eval_adapter.py` |
| `06_7b_try` | the 7B experiments |

Each notebook should be short: setup cells, then one command that runs a script from the repo. Keep the logic in the repo scripts, not in notebook cells.

## 8. Standard first cells (copy and adjust)

Before running: turn **Internet ON**, attach your Spider dataset, and attach the `GITHUB_TOKEN` secret (Add-ons, Secrets). For Steps 1 and 2, keep the **GPU off** to save quota.

### Cell 1: Get the code (private repo)

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")

!git clone https://{token}@github.com/AkshatGupte/Agentic-Fine-Tuning.git /kaggle/working/repo
# Remove the token from the saved git settings
!git -C /kaggle/working/repo remote set-url origin https://github.com/AkshatGupte/Agentic-Fine-Tuning.git
%cd /kaggle/working/repo
```

### Cell 2: Install only what is needed

```python
# Steps 1 and 2 only need this. Do not install the full requirements.txt on Kaggle
# unless the agent says it is needed, because it can clash with Kaggle's preinstalled torch/transformers.
!pip install -q pyyaml
```

### Cell 3: Check the data path

```python
import os
DATA_ROOT = "/kaggle/input/datasets/akshatgupte/spider-data-2"
print(os.listdir(DATA_ROOT))   # must show: database, train_spider.json, dev.json, tables.json
```

If those four items are not listed, the path is one folder too high or too low. List the folder and fix `DATA_ROOT`.

### Cell 4: Check the GPU (only when the GPU is ON)

```python
!nvidia-smi
```

With the GPU off, this command fails. That is expected.

### Cell 5: Run the Step 1 check

```python
!python scripts/smoke_load.py --data-root {DATA_ROOT}
```

### Updating the code after you push changes from VS Code

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git -C /kaggle/working/repo pull https://{token}@github.com/AkshatGupte/Agentic-Fine-Tuning.git main
```

Never print the token or paste it into a cell, a file, or a chat message.

## 9. Typical problems

| Problem | Likely fix |
|---|---|
| `CUDA out of memory` | shorter max length, smaller batch with more accumulation, gradient checkpointing, smaller LoRA rank |
| `bf16 not supported` error | use fp16 in the config |
| Model download fails | turn Internet ON, check `HF_TOKEN` in Secrets, or attach the model |
| Files vanished after restart | `/kaggle/working` was cleared; reload from the saved dataset or checkpoint |
| Very slow generation | use batch generation, left padding, and shorter `max_new_tokens` |
| Package version conflicts | pin versions in `requirements.txt` after the first working run |
| `git clone` asks for a password or says "not found" | The secret is not attached to this notebook, the token expired, or it was not given access to this repo. Re-check all three. |
| `Secret not found` error | The label must be exactly `GITHUB_TOKEN`, and the secret must be attached in this notebook. |
| `nvidia-smi` fails | The GPU is off. Expected for Steps 1 and 2. Turn it on in Settings for Step 3 onward. |
| `git pull` says local changes would be overwritten | You edited files on Kaggle. Re-clone into a fresh folder, or run `git checkout .` first. |

## 10. Workflow loop (summary)

1. Agent writes code in VS Code and runs the **smoke test** locally.
2. Commit and push to GitHub.
3. On Kaggle: `git pull`, run the notebook.
4. Copy result files back to the repo, commit them.
5. Tell the agent the results and any errors. Agent fixes code locally. Repeat.
