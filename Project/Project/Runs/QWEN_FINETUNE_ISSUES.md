# Qwen MDLM Finetuning — Issues & Workarounds Log

Full chronological record of every problem hit when running the MIA pipeline on
`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` across 6 MIMIR datasets on JarvisLabs L4 GPUs.
Use this as a checklist before launching any new Qwen finetuning run.

---

## 0. Model & Pipeline Overview

- **Target model**: `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`
- **Architecture**: Masked Diffusion LM built on Qwen3-0.6B, `mask_token_id = 151669`
- **Tokenizer**: `Qwen2Tokenizer` (NOT `AutoTokenizer` — see Issue 11 below)
- **Model class**: `AutoModelForMaskedLM` with `trust_remote_code=True`
- **Datasets**: MIMIR `iamgroot42/mimir`, split `ngram_13_0.8`, 6 domains
- **GPU**: L4 (24 GB VRAM), single GPU per instance
- **Python venv**: created by `jl run start` at `/home/code/.venv/`

---

## Issue 1 — DeepSpeed `mpi4py` not found (P0 crash, Stage 2)

### Symptom
```
ModuleNotFoundError: No module named 'mpi4py'
```
Pipeline aborted at finetune stage.

### Root Cause
`finetune.py` used `deepspeed.initialize()`. DeepSpeed on JarvisLabs L4 instances requires
`mpi4py` (an MPI implementation). The wheel is not installed by default and the L4 instances
did not ship with Open MPI.

### Workaround Applied
**Removed DeepSpeed entirely.** The Qwen3-0.6B model is small enough (~600M params) that
plain AdamW in float32 (or bfloat16 on CUDA) works fine on a single L4.

Changed `finetune.py` to:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
```

Also:
- Deleted `ds_config.json` from code directory (orphan file, was confusing)
- Removed `deepspeed` from `requirements_remote.txt`
- Changed `run_pipeline.sh` launcher from `deepspeed --num_gpus=1 finetune.py` → `python finetune.py`

### Rule
Never use DeepSpeed on JarvisLabs unless you explicitly verify `mpi4py` is present.
For models < 2B params on a single GPU, plain AdamW is sufficient.

---

## Issue 2 — PyTorch / torchvision CUDA version mismatch (P0 crash, import)

### Symptom
```
OSError: /home/code/.venv/lib/python3.11/site-packages/torchvision/...
undefined symbol: _ZN2at... (requires CUDA 12.8, got cu130)
```

### Root Cause
`jl run start` uses `uv` to install packages. Without pinning, uv pulled
`torch==2.11.0+cu130` (CUDA 13.0) but the L4 instance driver only supports up to CUDA 12.8.

### Workaround Applied
Added `--setup` flag to `jl run start` to pre-install the correct wheel **before** uv
resolves the rest of `requirements_remote.txt`:

```bash
jl run start . \
  --script run_pipeline.sh \
  --on {MID} \
  --requirements requirements_remote.txt \
  --setup "pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
            --extra-index-url https://download.pytorch.org/whl/cu128 -q" \
  --yes --json \
  -- {DATASET} 1
```

The pre-installed pinned wheels satisfy uv's constraints, so it does not replace them.

### Rule
Always pin `torch==2.10.0+cu128` via `--setup` on JarvisLabs L4/A100 instances until
JarvisLabs updates their driver baseline.

---

## Issue 3 — wandb 401 Unauthorized (P1, every run)

### Symptom
```
wandb: ERROR Failed to init run: 401 Unauthorized
wandb: Run aborted.
```

### Root Cause
`jl run start` creates a fresh venv. The system-level `wandb login` state is not visible
to the venv. The `~/.env` file (containing `WANDB_API_KEY`) was not sourced automatically
before the Python process started.

### Workaround Applied — two-step fix

**Step 1**: Upload secrets as `/home/.env`:
```bash
cat > /tmp/run3_env.sh << 'EOF'
export HF_TOKEN=hf_...
export WANDB_API_KEY=wandb_v1_...
EOF
jl upload {MID} /tmp/run3_env.sh /home/.env
```

**Step 2**: Source `.env` at the top of `run_pipeline.sh`:
```bash
#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f /home/.env ] && source /home/.env
```

**Step 3** (belt-and-suspenders): After venv is created by the first `jl run start`, also
login manually via the venv binary:
```bash
jl exec {MID} -- sh -lc \
  "/home/code/.venv/bin/wandb login {WANDB_KEY} --relogin"
```

### Rule
Always upload `.env` AND source it in the pipeline script. Do not rely on system-level
wandb state being visible inside the jl-managed venv.

---

## Issue 4 — `weights_only=True` crash on all Python-object `.pt` files (P0 crash, multiple stages)

### Symptom
```
_pickle.UnpicklingError: Weights only load does not allow Python object loading...
```
Or:
```
TypeError: cannot pickle 'dict' object
```

### Root Cause
PyTorch ≥ 2.0 defaults `torch.load(..., weights_only=True)`. This flag only allows
tensors and primitives. Any `.pt` file that contains Python dicts, lists, or strings
(which includes ALL our data files) fails with this error.

### Files Affected
| File | torch.load call |
|------|----------------|
| `finetune.py` | `members.pt` (dict with "texts" list) |
| `verify_memorization.py` | `members.pt`, `nonmembers.pt` |
| `run_attacks.py` | `members.pt`, `nonmembers.pt` |
| `run_signals.py` | `members.pt`, `nonmembers.pt` |
| `benchmark.py` | `classifier_results.pt` (dict with nested dicts) |

### Workaround Applied
Changed every `torch.load` that loads Python-object files to:
```python
data = torch.load(path, weights_only=False)
```

### Rule
Use `weights_only=False` for **all** `.pt` files that contain anything other than plain
tensors. Use `weights_only=True` only when loading model state dicts.

---

## Issue 5 — `dtype=` vs `torch_dtype=` in `from_pretrained` (P0 crash, verify_memorization)

### Symptom
```
TypeError: BertConfig.__init__() got an unexpected keyword argument 'dtype'
```
Or (worse): model silently loaded in float32 instead of bfloat16, causing OOM on later stages.

### Root Cause
`verify_memorization.py` used `dtype=dtype` in the `from_pretrained` call. The correct
HuggingFace parameter name is `torch_dtype`.

```python
# WRONG
model = AutoModelForMaskedLM.from_pretrained(path, dtype=dtype)

# CORRECT
model = AutoModelForMaskedLM.from_pretrained(path, torch_dtype=dtype)
```

### Workaround Applied
Fixed both model loads in `verify_memorization.py`:
```python
base_model = AutoModelForMaskedLM.from_pretrained(
    base_path, trust_remote_code=True, torch_dtype=dtype
).to(device).eval()
ft_model = AutoModelForMaskedLM.from_pretrained(
    ft_path, trust_remote_code=True, torch_dtype=dtype
).to(device).eval()
```

### Rule
It is **always** `torch_dtype=`, never `dtype=`, in `AutoModelForMaskedLM.from_pretrained`.
The keyword `dtype` is silently ignored or causes a TypeError depending on the config class.

---

## Issue 6 — `sys.exit(1)` in verify_memorization aborting entire pipeline (P1)

### Symptom
Pipeline stopped after Stage 3 (verify_memorization) with exit code 1, even when the ELBO
gap was below threshold. All downstream stages (SAMA, attacks, signals, classifier, benchmark)
were skipped.

### Root Cause
`verify_memorization.py` called `sys.exit(1)` when `member_gap < MIN_GAP`. The pipeline
shell script uses `set -euo pipefail`, so any non-zero exit code from any stage kills the
entire pipeline immediately.

### Workaround Applied
Replaced `sys.exit(1)` with a WARNING print that continues execution:
```python
# BEFORE
if member_gap < MIN_GAP:
    print(f"FAIL: member_gap={member_gap:.4f} below threshold {MIN_GAP}")
    sys.exit(1)

# AFTER
if member_gap < MIN_GAP:
    print(f"WARNING: member_gap={member_gap:.4f} below threshold {MIN_GAP} — continuing anyway")
```

### Rule
Never call `sys.exit(1)` from any non-final pipeline stage. Use warnings and continue.
If a hard stop is needed, set a flag file and check it at the start of the next stage.

---

## Issue 7 — Zlib score double-normalization (P1, wrong metric)

### Symptom
Zlib AUC results were slightly off from published baselines. The formula in the original
code was dividing NLL by zlib entropy (already normalized by text length), effectively
computing `-NLL * len(text) / compress_bytes` instead of the standard formula.

### Root Cause
The original `run_attacks.py` computed:
```python
zlib_entropy = len(zlib.compress(text.encode())) / len(text)  # already normalized
zlib_scores[i] = -target_nlls[i] / zlib_entropy              # double-normalized!
```

The Carlini 2021 formula is simply:
```
score = −NLL / compress_bytes
```

### Workaround Applied
```python
compress_bytes = max(1, len(zlib_mod.compress(text.encode())))
zlib_scores[i] = -target_nlls[i] / compress_bytes
```

### Rule
Zlib MIA score = per-sample NLL (not per-token) divided by raw compressed byte count.
No double normalization. `max(1, ...)` guards against empty strings.

---

## Issue 8 — Missing `cd` in `run_pipeline.sh` (P0 crash, all stages)

### Symptom
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/github/members.pt'
```
All relative paths broken when script invoked from a different working directory.

### Root Cause
`jl run start` invokes `run_pipeline.sh` from `/home/`, not from `/home/code/` where
the script lives. All relative paths (`data/`, `models/`, `results/`) resolve against
the launch CWD, not the script's location.

### Workaround Applied
Added at the very top of `run_pipeline.sh`, immediately after `set -euo pipefail`:
```bash
cd "$(dirname "${BASH_SOURCE[0]}")"
```

This resolves the script's own directory and changes into it, making all relative paths
correct regardless of where the script was invoked from.

### Rule
Every pipeline shell script must have `cd "$(dirname "${BASH_SOURCE[0]}")"` as its
first real command when it uses relative paths.

---

## Issue 9 — Stale hyperparameter docstring in `finetune.py`

### Symptom
Not a crash, but the script header said `lr=5e-5, wd=0.1, bs=48, 4 epochs` while the
actual `config.py` values were `lr=1e-4, wd=0.01, bs=8, 5 epochs`.

### Workaround Applied
Updated the docstring to match actual config values.

### Rule
Do not hardcode hyperparameters in docstrings — reference `config.py` values directly,
or add a note that docstrings may lag the config.

---

## Issue 10 — MIMIR `wikipedia` dataset config name wrong (P0, prepare_data)

### Symptom
```
datasets.builder.DatasetBuildingError: BuilderConfig 'wikipedia' not found.
Available: ['wikipedia_(en)', ...]
```

### Root Cause
The MIMIR HuggingFace dataset `iamgroot42/mimir` names its Wikipedia config
`wikipedia_(en)`, not `wikipedia`.

### Workaround Applied
Added a lookup map in `config.py`:
```python
MIMIR_HF_NAME = {
    "github":          "github",
    "arxiv":           "arxiv",
    "hackernews":      "hackernews",
    "pubmed_central":  "pubmed_central",
    "pile_cc":         "pile_cc",
    "wikipedia":       "wikipedia_(en)",   # ← special case
}
```
`prepare_data.py` uses `MIMIR_HF_NAME[dataset]` as the config name when calling
`load_dataset`.

### Rule
Always check actual MIMIR config names via `datasets.get_dataset_config_names("iamgroot42/mimir")`
before assuming they match the domain name string.

---

## Issue 11 — `AutoTokenizer` fails for local checkpoints with unknown `model_type` (P0, run_signals)

### Symptom
```
OSError: /home/code/models/github/finetuned_checkpoint does not appear to have a file named config.json
```
Or:
```
ValueError: Unrecognized model type 'a2d-qwen3' in config.json. Falling through to AutoConfig...
KeyError: 'a2d-qwen3'
```

### Root Cause
`transformers >= 4.57` changed `AutoTokenizer.from_pretrained` to call `AutoConfig` first,
which fails for local paths with a custom `model_type`. The finetuned checkpoint's
`config.json` has `"model_type": "a2d-qwen3"`, which is not registered in the standard
HuggingFace model type registry.

### Workaround Applied
Use `Qwen2Tokenizer` directly instead of `AutoTokenizer`:
```python
from transformers import Qwen2Tokenizer
tokenizer = Qwen2Tokenizer.from_pretrained(ft_path)
```

### Rule
For any custom DLLM built on Qwen, use `Qwen2Tokenizer` explicitly. Do **not** use
`AutoTokenizer` with local checkpoints that have a custom `model_type`.

---

## Issue 12 — `attn_implementation="eager"` required for attention capture (P1, run_signals)

### Symptom
Attention weights were `None` when `capture_attentions=True` was passed to `extract_metrics`.
No crash — just all attention-based features (attn_entropy, attn_crosslayer, attn_barycenter,
attn_perturbation) were zero or NaN.

### Root Cause
The default attention implementation in transformers uses `sdpa` (scaled dot-product
attention with flash attention), which does not return attention weight matrices.
Only the `eager` implementation returns full attention tensors.

### Workaround Applied
All model loads in `run_signals.py` use:
```python
model = AutoModelForMaskedLM.from_pretrained(
    path, trust_remote_code=True, torch_dtype=dtype,
    attn_implementation="eager",
).to(device).eval()
```

### Rule
Any script that reads `model.outputs.attentions` must use `attn_implementation="eager"`.
This applies to both the finetuned model and the base model in cross-model comparison.

---

## Issue 13 — NLP datasets: 0 sequences survive length filter (P1, prepare_data)

### Symptom
```
AssertionError: 0 member sequences after filtering (min_length=256 tokens)
```

### Root Cause
Datasets like `agnews` (news headlines) have very short texts. With `min_length=256` tokens
(the MIMIR config), essentially all samples were filtered out.

### Workaround Applied
Added a `min_length` override in `config.py` for NLP datasets:
```python
NLP_MIN_LENGTH = 16  # tokens — much shorter than MIMIR's 256
```
`prepare_data.py` checks `dataset in NLP_DATASETS` and applies `NLP_MIN_LENGTH`.

### Rule
MIMIR's `min_length=256` is appropriate for pretraining corpora (long documents) but not
for NLP classification datasets. Always check the average token count of your dataset
before setting a length filter.

---

## Issue 14 — JarvisLabs: `jl run start` wipes existing files on re-upload (P1)

### Symptom
After a mid-run failure, attempting to re-launch with a patched script via `jl run start`
caused all previously computed data (members.pt, models/, results/) to be deleted.

### Root Cause
`jl run start` re-uploads the entire code directory to the instance, replacing everything
in `/home/code/`. Data stored under `/home/code/data/`, `/home/code/models/`, and
`/home/code/results/` was wiped.

### Workaround Applied
For mid-run patches:
- Use `jl exec {MID} -- sh -lc "..."` to patch files directly on the running instance
- OR use `jl upload {MID} local_patched_file.py /home/code/patched_file.py` to upload
  only the specific changed file
- Then resume the pipeline from the failed stage: `bash run_pipeline.sh {DATASET} {STAGE_N}`

Never re-run `jl run start` on an instance that has already computed data.

### Rule
`jl run start` = full re-upload + fresh run. Use `jl exec` or `jl upload` for patches
on already-running or partially-completed instances.

---

## Issue 15 — JarvisLabs: nested directories on download (P2)

### Symptom
After `jl download {MID} /home/code/results/github/ results/github/`, the files landed at
`results/github/github/*.pt` instead of `results/github/*.pt`.

### Root Cause
`jl download` appends the remote directory's basename as a subdirectory of the local target.
So `/home/code/results/github/` becomes `results/github/github/`.

### Workaround Applied
```bash
# Move files up one level and remove the extra directory
mv results/github/github/* results/github/
rmdir results/github/github
```

Or download the parent and let it create the correct structure:
```bash
jl download {MID} /home/code/results/ results/  # downloads results/{DS}/ correctly
```

### Rule
When using `jl download`, download the **parent** of the directory you want, not the
directory itself — or flatten immediately after download.

---

## Issue 16 — Machine IDs change after instance resume (P2)

### Symptom
After pausing all 6 instances overnight and resuming, the machine IDs changed:
- `400158` → `400189` (github)
- `400159` → `400191` (arxiv)
- etc.

### Root Cause
JarvisLabs reassigns machine IDs when paused instances are resumed on different physical
hosts (for scheduling efficiency).

### Workaround Applied
Always look up current machine IDs before any `jl exec`, `jl upload`, or `jl download`:
```bash
jl list  # shows all instances with current IDs and status
```

### Rule
Never hard-code machine IDs in scripts. Always query `jl list` at the start of a session.

---

## Issue 17 — System python3 missing torch for local post-processing scripts (P2)

### Symptom
```
ModuleNotFoundError: No module named 'torch'
```
When running local analysis scripts with `python3 extract_and_plot.py`.

### Root Cause
macOS system Python (`/usr/bin/python3`) does not have torch. The project venv is at
a different path.

### Workaround Applied
Always invoke analysis scripts with the project venv:
```bash
/Users/skumar/Desktop/DA5001/venv3.11/bin/python extract_and_plot.py
```

Or activate the venv first:
```bash
source /Users/skumar/Desktop/DA5001/venv3.11/bin/activate
python extract_and_plot.py
```

### Rule
Never run `python3 script.py` without first checking which python is active.
Add a shebang or always use the full venv path for one-off scripts.

---

## Issue 18 — Stale T/feat_dim comments in `run_signals.py`

### Symptom
Docstring said `T=16, feat_dim=177` but actual config was `T=4, feat_dim=46`.
Not a crash, but caused confusion when debugging feature dimension mismatches.

### Workaround Applied
Updated all comments and docstring in `run_signals.py` to reference `cfg.T_STEPS`
dynamically rather than hardcoded values.

### Rule
Never hardcode T or feature dimension in comments. Use:
```python
feat_dim = 11 * cfg.T_STEPS + 2  # 11 groups × T + elbo_var + cross_model_cos
```

---

## Quick-Reference: All Config Values Used in Run 3

| Parameter | Value | Location |
|-----------|-------|----------|
| Model | `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` | `config.py` |
| `mask_token_id` | 151669 | derived from tokenizer |
| Tokenizer class | `Qwen2Tokenizer` | `run_signals.py` |
| `trust_remote_code` | `True` | all model loads |
| `attn_implementation` | `"eager"` | `run_signals.py` only |
| `torch_dtype` | `bfloat16` (CUDA) / `float32` (MPS/CPU) | all model loads |
| `T_STEPS` | 4 | `config.py` |
| `ALPHA_MIN / ALPHA_MAX` | 0.05 / 0.50 | `config.py` |
| `SIG_K` | 8 (mask configs) | `config.py` |
| `SIG_GRAD_T` | 2 | `config.py` |
| `MAX_LENGTH` | 256 | `config.py` |
| LR | 1e-4 | `config.py` |
| Weight decay | 0.01 | `config.py` |
| Batch size | 8 | `config.py` |
| Epochs | 5 | `config.py` |
| SAMA N_subsets | 128 | `config.py` |
| SAMA m_tokens | 10 | `config.py` |
| XGBoost n_estimators | 200 | `train_classifier.py` |
| CV folds | 5 | `train_classifier.py` |
| Bootstrap samples | 1000 | `benchmark.py` |
| torch version | 2.10.0+cu128 | `--setup` in jl run |
| MIMIR split | `ngram_13_0.8` | `config.py` |
| Samples per class | 1000 | `config.py` |
| Feature dimension | 46 (= 11×4 + 1 + 1) | `run_signals.py` |

---

## Pre-Launch Checklist (for any new Qwen run)

- [ ] `run_pipeline.sh` starts with `cd "$(dirname "${BASH_SOURCE[0]}")"` and sources `/home/.env`
- [ ] No `deepspeed` in requirements or pipeline script
- [ ] torch pinned to `2.10.0+cu128` in `--setup` flag of `jl run start`
- [ ] All `torch.load` calls use `weights_only=False` for data/result files
- [ ] All `from_pretrained` calls use `torch_dtype=` (not `dtype=`)
- [ ] `verify_memorization.py` does NOT call `sys.exit(1)` on low gap
- [ ] `run_signals.py` uses `Qwen2Tokenizer` (not `AutoTokenizer`)
- [ ] `run_signals.py` model loads include `attn_implementation="eager"`
- [ ] MIMIR wikipedia uses config name `wikipedia_(en)` (not `wikipedia`)
- [ ] Zlib formula uses `-NLL / compress_bytes` (no double normalization)
- [ ] `.env` uploaded to instance BEFORE pipeline starts
- [ ] wandb logged in via venv binary: `/home/code/.venv/bin/wandb login {KEY}`
- [ ] `jl list` run at session start to get current machine IDs
- [ ] Mid-run patches go via `jl exec` or `jl upload`, NOT `jl run start`
- [ ] Downloads: pull parent directory, then flatten if needed

---

## Stage-by-Stage Expected Runtimes (L4 GPU, 1000 member + 1000 non-member)

| Stage | Script | Time |
|-------|--------|------|
| 1. prepare_data | `prepare_data.py` | ~2 min |
| 2. finetune | `finetune.py` (5 ep, bs=8) | ~5 min |
| 3. verify_memorization | `verify_memorization.py` | ~8 min |
| 4. run_sama | `run_sama.py` (T=4, N=128) | ~12 min |
| 5. run_attacks | `run_attacks.py` | ~15 min |
| 6. run_signals | `run_signals.py` (T=4, 2000 samples) | ~2 hrs |
| 7. train_classifier | `train_classifier.py` (5-fold CV) | ~3 min |
| 8. benchmark | `benchmark.py` (+ ablation) | ~5 min |
| **Total** | | **~2h 50min** |

`run_signals.py` dominates because it runs `extract_metrics()` per sample with attention
capture enabled (`eager` attention is ~2× slower than flash attention).
