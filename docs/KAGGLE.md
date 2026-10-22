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
2. In the Kaggle notebook, turn **Internet ON** (Settings panel).
3. In the first cell:

```python
!git clone https://github.com/<user>/<repo>.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q -r requirements.txt
```

4. After you change code in VS Code and push, run this in the notebook to update:

```python
%cd /kaggle/working/repo
!git pull
```

For a **private** repo, use a GitHub token stored in **Kaggle Secrets** (Add-ons, Secrets). Never paste a token into a notebook cell or into a file in the repo.

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
- Do not copy the data into git.

## 5. Models on Kaggle

Two ways to get a model:
- **Download from Hugging Face** (needs Internet ON). Store your token in **Kaggle Secrets** as `HF_TOKEN`.
- **Attach the model** from Kaggle's Models / Datasets, then point the config to `/kaggle/input/...`.

Downloaded model files go to the cache and are lost when the session ends, so they download again next time. Attaching a model avoids that.

## 6. Saving your work (very important)

Sessions end and `/kaggle/working` is cleared. So:

1. Save checkpoints often (set `save_steps` in the training config).
2. Write adapters and results to `/kaggle/working/outputs/`.
3. Before the session ends, **save the adapter as a Kaggle Dataset** (use "Save Version" and keep outputs, or create a dataset from the output folder).
4. Download the `results/*.json` files and copy them into the VS Code repo, so the results live in git.
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

```python
# 1. Get the code
!git clone https://github.com/<user>/<repo>.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q -r requirements.txt

# 2. Check the GPU
!nvidia-smi

# 3. Check the data path
import os
print(os.listdir("/kaggle/input"))
```

## 9. Typical problems

| Problem | Likely fix |
|---|---|
| `CUDA out of memory` | shorter max length, smaller batch with more accumulation, gradient checkpointing, smaller LoRA rank |
| `bf16 not supported` error | use fp16 in the config |
| Model download fails | turn Internet ON, check `HF_TOKEN` in Secrets, or attach the model |
| Files vanished after restart | `/kaggle/working` was cleared; reload from the saved dataset or checkpoint |
| Very slow generation | use batch generation, left padding, and shorter `max_new_tokens` |
| Package version conflicts | pin versions in `requirements.txt` after the first working run |

## 10. Workflow loop (summary)

1. Agent writes code in VS Code and runs the **smoke test** locally.
2. Commit and push to GitHub.
3. On Kaggle: `git pull`, run the notebook.
4. Copy result files back to the repo, commit them.
5. Tell the agent the results and any errors. Agent fixes code locally. Repeat.
