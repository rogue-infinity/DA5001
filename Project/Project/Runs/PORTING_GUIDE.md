# MIA Pipeline — Issues Log & Porting Guide
## For replicating onto LLaDA-8B (or any other open-source DLLM)

> Written after completing Run 3 (Full MIMIR, 6 domains, Qwen3-0.6B-MDLM).  
> Every bug listed here was hit in production. Do not skip any section.

---

## Table of Contents

1. [Mental Model — What the Pipeline Expects](#1-mental-model)
2. [Model Loading Issues](#2-model-loading-issues)
3. [Tokenizer Issues](#3-tokenizer-issues)
4. [Data / Split Issues](#4-data--split-issues)
5. [Fine-tuning Issues](#5-fine-tuning-issues)
6. [SAMA Attack Issues](#6-sama-attack-issues)
7. [Signal Extraction Issues](#7-signal-extraction-issues)
8. [PyTorch Serialisation Issues (`weights_only`)](#8-pytorch-serialisation-issues)
9. [Pipeline / Shell Issues](#9-pipeline--shell-issues)
10. [JarvisLabs / Remote Execution Issues](#10-jarvislabs--remote-execution-issues)
11. [wandb / Auth Issues](#11-wandb--auth-issues)
12. [LLaDA-8B Porting Checklist](#12-llada-8b-porting-checklist)
13. [Quick Reference — Config Values That Must Match](#13-quick-reference)

---

## 1. Mental Model

The pipeline assumes the target model is a **Masked Diffusion LM** (MDLM):
- Forward process: linearly mask tokens at rate `t ~ U(ε, 1)`
- Training objective: cross-entropy on masked token positions (ELBO)
- At inference: feed noised `input_ids` (with mask tokens), get logit predictions

Everything downstream (SAMA, signal extraction, loss attack) builds on this.
If your model is autoregressive (causal LM), **the whole pipeline is wrong** — you need a completely different set of signals.

**Verify your model is truly masked diffusion before doing anything else:**
```python
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
# Should work without errors
# Check: model has a mask_token_id
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
assert tokenizer.mask_token_id is not None, "No mask token — not an MLM!"
print("mask_token_id:", tokenizer.mask_token_id)
```

---

## 2. Model Loading Issues

### 2.1 `dtype=` vs `torch_dtype=` in `from_pretrained`

**Bug hit:** `verify_memorization.py` used `dtype=dtype` instead of `torch_dtype=dtype`.

**Symptom:** Model loads silently in float32 regardless of what you pass, wasting VRAM and slowing everything down. Sometimes raises a `TypeError` depending on the transformers version.

**Fix — always use:**
```python
model = AutoModelForMaskedLM.from_pretrained(
    path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,   # ← correct kwarg
)
```

**Note:** Newer transformers (≥4.57) emit a deprecation warning:
```
`torch_dtype` is deprecated! Use `dtype` instead.
```
This warning is confusing and contradicts older code. For now, `torch_dtype` still works. If you update transformers again, swap to `dtype=`. Check the version and be consistent.

### 2.2 `trust_remote_code=True` is mandatory

Both Qwen3-MDLM and LLaDA have custom model architectures not in the transformers registry. Without `trust_remote_code=True`, loading fails silently or with an obscure `KeyError` on the model type.

```python
# Always pass this for any non-standard DLLM:
model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True, ...)
```

### 2.3 `attn_implementation="eager"` for signal extraction

The signal extractor (`run_signals.py`) captures per-head attention matrices. Flash Attention 2 (`sdpa` or `flash_attention_2`) does not return attention weights — it silently returns `None` for `attentions`.

**Fix:** Always pass `attn_implementation="eager"` when loading models for signal extraction:
```python
model = AutoModelForMaskedLM.from_pretrained(
    path,
    trust_remote_code=True,
    torch_dtype=dtype,
    attn_implementation="eager",   # ← needed for attn capture
)
```
This is only needed in `run_signals.py`. For fine-tuning and attacks, the default (flash/sdpa) is fine and faster.

### 2.4 `AutoModelForMaskedLM` vs `AutoModelForCausalLM`

SAMA's built-in `RatioAttack` class hardcodes `AutoModelForCausalLM` for the reference model. This crashes on any DLLM.

**Fix:** Do not use SAMA's built-in `RatioAttack`. Instead, compute the ratio attack manually using `compute_nlloss()` from SAMA's utils, loading both models with `AutoModelForMaskedLM`. (Already done in `run_attacks.py`.)

### 2.5 `dllm` package installation

The Qwen3-MDLM model references the `dllm` package for its model architecture. It must be installed **without its dependencies** because `dllm` pins `datasets==4.2.0` which conflicts with MIMIR loading (which needs `datasets<3.0`).

```bash
pip install --no-deps -e ./dllm
```

**For LLaDA:** Check if LLaDA requires any custom package similarly. LLaDA models typically reference `llada` or are pure HuggingFace compatible. Check `config.json`'s `auto_map` field:
```bash
cat /path/to/model/config.json | grep auto_map
```
If there's an `auto_map`, the model needs `trust_remote_code=True` and potentially a custom package.

---

## 3. Tokenizer Issues

### 3.1 `AutoTokenizer` fails for local checkpoints on transformers ≥ 4.57

**Bug hit:** When loading a locally saved checkpoint (after fine-tuning), `AutoTokenizer.from_pretrained(local_path)` fails with:
```
KeyError: 'a2d-qwen3'   # or similar unknown model_type
```

**Root cause:** `AutoTokenizer` first calls `AutoConfig` which fails on unknown `model_type` in the local `config.json`.

**Fix:** Load the tokenizer class directly, bypassing `AutoConfig`:
```python
from transformers import Qwen2Tokenizer
tokenizer = Qwen2Tokenizer.from_pretrained(local_path)
```

**For LLaDA-8B:** LLaDA is based on LLaMA-3, so:
```python
from transformers import PreTrainedTokenizerFast
# or
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
# Should work from HF directly. For local paths, use the explicit class if needed.
```

### 3.2 Verify `mask_token_id` before any diffusion code

Different models use different mask token IDs. The entire SAMA attack and signal extraction depends on using the correct one.

| Model | mask_token_id |
|-------|--------------|
| Qwen3-0.6B-MDLM | **151669** |
| LLaDA-8B | **126336** (verify!) |
| MDLM (generic) | depends on tokenizer |

**Always print and assert:**
```python
mask_token_id = tokenizer.mask_token_id
print(f"mask_token_id: {mask_token_id}")
assert mask_token_id is not None, "Tokenizer has no mask_token_id!"
```

If `mask_token_id` is `None`, the model's tokenizer doesn't define a mask token. You may need to add it:
```python
tokenizer.add_special_tokens({"mask_token": "[MASK]"})
model.resize_token_embeddings(len(tokenizer))
mask_token_id = tokenizer.mask_token_id
```

---

## 4. Data / Split Issues

### 4.1 Length filtering silently shrinks the split below 1000/1000

**Bug hit:** `min_len = max_length = 256` was filtering out ~12.5% of MIMIR samples, giving 875/875 instead of 1000/1000. The pipeline ran without errors — only discovered when checking sample counts.

**Fix for MIMIR domains:**
```python
# In prepare_data.py:
min_len = 1 if dataset in MIMIR_DATASETS else 16
```

**General rule:** Never set `min_len` equal to `max_length`. Truncation at tokenization handles long texts; short texts are valid. Always print the actual count after filtering:
```python
print(f"Members: {len(member_texts)}, Non-members: {len(nonmember_texts)}")
assert len(member_texts) == N, f"Expected {N}, got {len(member_texts)}"
```

### 4.2 MIMIR HF config names differ from domain names

The HF dataset config name is not always the same as the domain string:

```python
MIMIR_HF_NAME = {
    "wikipedia": "wikipedia_(en)",  # ← NOT "wikipedia"
    "arxiv": "arxiv",
    "github": "github",
    "hackernews": "hackernews",
    "pubmed_central": "pubmed_central",
    "pile_cc": "pile_cc",
}
```

Using the wrong config name silently loads the wrong domain. Always map explicitly.

### 4.3 MIMIR split name

Always use `split="ngram_13_0.8"` — this is the deduplicated split. Other splits may contain near-duplicates that inflate MIA performance artificially.

---

## 5. Fine-tuning Issues

### 5.1 Hyperparameters — must match the validated Run 2 config

Previous runs used wrong hyperparameters (lr=5e-5, wd=0.1, T=16) copied from a paper config for A100×3 + DeepSpeed. These caused SAMA AUC to drop from ~0.80 to ~0.65.

**Correct config for single L4 (24GB):**
```python
FT_LR          = 1e-4
FT_WD          = 0.01
FT_BATCH_SIZE  = 8
FT_EPOCHS      = 5
```

**For LLaDA-8B on a single A100-80GB:**
- Batch size 1 or 2 (8B model is ~16GB in bf16 weights alone)
- Gradient accumulation steps: 8 or 16 to match effective bs=8 or 16
- Same lr=1e-4 is a reasonable starting point
- Enable gradient checkpointing (mandatory for 8B)
- Consider 3 epochs instead of 5 (8B memorises faster)

### 5.2 Batch size OOM

**Bug hit:** Original config had `FT_BATCH_SIZE=48` copied from a DeepSpeed/A100×3 config. This OOMs immediately on L4.

**L4 (24GB) limits:**
- Qwen3-0.6B bf16: bs=8 works fine with gradient checkpointing
- Never assume batch size from a paper config — it's always hardware-specific

**A100-80GB limits for LLaDA-8B:**
- Recommend starting at bs=1, grad_accum=8
- Test with `--max_steps 5` first before a full run

### 5.3 Gradient checkpointing

Always enable for any model on L4 or smaller. For 8B it's mandatory on A100 too:
```python
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
```

### 5.4 T_STEPS — the most critical hyperparameter for SAMA

`T_STEPS` controls the number of diffusion timesteps sampled during SAMA and signal extraction. **This must be 4**, not 16.

- T=16 caused SAMA AUC to drop from ~0.80 → ~0.65 because the vectorised subset computation becomes inefficient and the score alignment degrades.
- T=4 with α ∈ {0.05, 0.20, 0.35, 0.50} matched Run 2 results exactly.

```python
T_STEPS   = 4       # ← hardcode this, do not derive from anything else
ALPHA_MIN = 0.05
ALPHA_MAX = 0.50
```

---

## 6. SAMA Attack Issues

### 6.1 SAMA `sys.path` setup

SAMA is not a pip-installable package. It must be cloned and added to `sys.path` manually. Two paths must be added:
```python
sys.path.insert(0, sama_root)                        # for top-level imports
sys.path.insert(0, os.path.join(sama_root, "attack")) # for attack.attacks.*
```

The `run_pipeline.sh` handles this by cloning to `/home/SAMA` and exporting `SAMA_ROOT`. Confirm the clone succeeded before running:
```bash
[ -d "$SAMA_ROOT/attack" ] || { echo "SAMA not found!"; exit 1; }
```

### 6.2 SAMA expects `AutoModelForMaskedLM`, not CausalLM

When using `SamaAttack` directly, it internally loads the reference model. For Qwen3-MDLM and LLaDA this works because the `config.json` has the correct `architectures` field. But always verify:
```python
from attack.attacks.sama import SamaAttack
attack = SamaAttack(target_model, ref_model, mask_token_id=mask_token_id)
```
If SAMA raises a `ValueError` about model type, you may need to patch `attack/attacks/sama.py` to accept `AutoModelForMaskedLM`.

### 6.3 SAMA `compute_nlloss` — the `shift_logits` parameter

MDLM/Qwen3 does **not** shift logits (unlike autoregressive models). Always pass:
```python
nll = compute_nlloss(
    model, input_ids, attention_mask,
    shift_logits=False,   # ← critical
    mc_num=4,
    mask_id=mask_token_id,
)
```
Passing `shift_logits=True` will offset predictions by one position and produce garbage NLL values.

### 6.4 Zlib attack formula

The original code had a non-standard double-normalization:
```python
# WRONG (double-normalizes by text length):
zlib_entropy = len(compress(text.encode())) / len(text)
score = -NLL / zlib_entropy  # = -NLL * len(text) / compress_bytes
```

**Correct (Carlini et al. 2021):**
```python
# CORRECT:
compress_bytes = max(1, len(zlib.compress(text.encode())))
score = -NLL / compress_bytes
```

---

## 7. Signal Extraction Issues

### 7.1 Feature dimension depends on T_STEPS — update all hardcoded sizes

If you change `T_STEPS`, the feature dimension changes: `feat_dim = 11 * T + 2`.

| T | feat_dim |
|---|---------|
| 4  | **46**  |
| 8  | 90      |
| 16 | 178     |

Every place that hardcodes the feature dimension must be updated. In particular:
- The `except` fallback in `run_signals.py` that appends a zero vector
- Any assertion on `X.shape[1]`
- The `GROUPS` dictionary slices

**Fix:** Always compute from `T_STEPS`, never hardcode:
```python
feat_dim = 11 * cfg.T_STEPS + 1 + 1  # 11 groups × T + elbo_var + cross_model_cos
```

### 7.2 `extract_metrics` `max_length` parameter — check if it exists

The `mdlm_metrics_extractor.py::extract_metrics()` function may or may not accept `max_length` depending on version. Always inspect before calling:
```python
import inspect
sig = inspect.signature(extract_metrics)
kwargs = {}
if "max_length" in sig.parameters:
    kwargs["max_length"] = cfg.MAX_LENGTH
bundle = extract_metrics(model, tokenizer, text, **kwargs)
```

### 7.3 Signal extraction is slow — plan accordingly

~3.5s per sample on L4 for T=4. For 2000 samples:
```
2000 × 3.5s ≈ 7000s ≈ 115 min ≈ ~2 hours
```

For LLaDA-8B the per-sample time will be significantly higher (larger model = more compute per forward pass). Estimate:
- 8B/0.6B ≈ 13× more parameters
- But signal extraction is mostly dominated by T×K forward passes, not model size alone
- Expect 8–15s per sample → 4–8 hours for 2000 samples

Consider reducing K (mask configs per timestep) from 8 to 4 to halve the time, at some cost to signal quality.

### 7.4 GROUPS slice dictionary must be updated if T changes

The GROUPS dict in `run_signals.py` uses hardcoded arithmetic based on T:
```python
T = cfg.T_STEPS  # must match what extract_metrics actually produces
GROUPS = {
    "elbo_traj":    slice(0, T),
    "elbo_var":     slice(T, T+1),
    "dldt":         slice(T+1, 2*T+1),
    # ... etc
}
```
If you change T, the slices automatically recalculate (they use `T` as a variable, not a literal). Verify the slices by checking `sum(s.stop - s.start for s in GROUPS.values()) == feat_dim`.

---

## 8. PyTorch Serialisation Issues

### 8.1 `weights_only=True` crashes on any `.pt` file containing Python objects

PyTorch ≥ 2.0 added `weights_only=True` as a security measure. It only allows loading raw tensors — any file containing Python dicts, lists, strings, or custom objects will raise:
```
_pickle.UnpicklingError: Weights only load failed
```

**Files that WILL fail with `weights_only=True`:**
- `members.pt` / `nonmembers.pt` — contain `{"texts": List[str], "input_ids": Tensor, ...}`
- `classifier_results.pt` — contains `{"metrics_xgb": dict, ...}`
- `ablation_results.pt` — contains nested dicts
- `signal_names.pt` — contains `List[str]`
- `benchmark.pt` — contains dicts

**Files that are SAFE with `weights_only=True`:**
- `X.pt` — pure tensor `[N, D]`
- `y.pt` — pure tensor `[N]`

**Rule:** Use `weights_only=False` everywhere in this pipeline. Only use `True` for files you know contain only tensors:
```python
# Safe for any file:
obj = torch.load(path, weights_only=False)
```

---

## 9. Pipeline / Shell Issues

### 9.1 Missing `cd` at top of `run_pipeline.sh`

**Bug hit:** When `jl run` invokes `run_pipeline.sh`, the working directory is not necessarily the script's directory. All relative paths (`data/`, `results/`, `logs/`, `models/`) resolve incorrectly.

**Fix — first lines of any pipeline shell script:**
```bash
#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"   # ← always add this
```

### 9.2 `set -euo pipefail` + `sys.exit(1)` aborts the whole pipeline

**Bug hit:** `verify_memorization.py` called `sys.exit(1)` when the ELBO gap was below threshold. Because `run_pipeline.sh` uses `set -euo pipefail`, any non-zero exit code from any stage aborts all downstream stages.

**Fix:** Replace fatal exits in any stage with warnings that still exit 0:
```python
# WRONG:
if member_gap < MIN_GAP:
    sys.exit(1)

# CORRECT:
if member_gap < MIN_GAP:
    print(f"WARNING: gap {member_gap:.4f} < {MIN_GAP}. Continuing anyway.")
# no exit — pipeline continues
```
Only use `sys.exit(1)` for genuine unrecoverable errors (missing file, wrong config).

### 9.3 DeepSpeed — remove completely

An earlier version of the pipeline included DeepSpeed for distributed training. For single-GPU L4, DeepSpeed is not needed and its presence causes:
- Import errors if not installed
- Incorrect batch size assumptions (DeepSpeed batches are per-GPU; total bs was 48×3=144 in the original config)
- Config file confusion (`ds_config.json` parameters override Python-level settings)

**Remove all DeepSpeed references.** Use plain PyTorch training loop with `AdamW`.

### 9.4 `benchmark.py` variable name bug

**Bug hit:** Line 96 used `args.results_dir` instead of the local variable `results_dir`. Since `args` has no `results_dir` attribute, this raised `AttributeError` and crashed stage 8.

**Pattern:** After parsing args, always immediately assign local variables:
```python
args = parser.parse_args()
results_dir = os.path.join(args.results_base, args.dataset)
# Always use results_dir, never args.results_dir
```

---

## 10. JarvisLabs / Remote Execution Issues

### 10.1 `jl run start .` wipes `/home/code/`

**Bug hit:** We used `jl run start .` to restart from stage 8 only — this re-uploads the entire code directory and **wipes the remote `/home/code/`**, destroying all previously computed signals, classifier results, etc.

**Rule:**
- `jl run start .` — only use for the **very first** run, or when you want a clean slate
- For patching a file and restarting from a late stage — use `jl exec` + `jl upload`:
  ```bash
  jl upload <id> ./fixed_benchmark.py /home/code/benchmark.py
  jl exec <id> -- sh -lc 'cd /home/code && bash run_pipeline.sh arxiv 8'
  ```

### 10.2 `jl resume` returns a **new** `machine_id`

After pausing and resuming, the machine ID changes. Always capture the new ID from the JSON response:
```bash
NEW_ID=$(jl resume 400158 --yes --json | python3 -c "import sys,json; print(json.load(sys.stdin)['machine_id'])")
```
Using the old ID for subsequent `jl exec`/`jl download` will fail silently or hit a different machine.

### 10.3 torch version pinning — prevent uv from pulling cu130

JarvisLabs instances come with PyTorch pre-installed, but `uv pip install -r requirements.txt` will upgrade to the latest version (cu130 as of 2026) which may break CUDA compatibility or introduce API changes.

**Fix in `run_pipeline.sh` via `--setup`:**
```bash
jl run . --script run_pipeline.sh \
  --requirements requirements_remote.txt \
  --setup "pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
           --extra-index-url https://download.pytorch.org/whl/cu128 -q" \
  --on <id> --yes --json -- <dataset>
```
The `--setup` runs after `--requirements`, overriding whatever uv installed.

### 10.4 Secrets must be in `/home/.env`, not `/home/code/.env`

`run_pipeline.sh` sources `~/.env` (which is `/home/.env` on JarvisLabs containers). Files in `/home/code/.env` are NOT sourced automatically and are wiped by `jl run start .`.

```bash
# Write locally:
cat > /tmp/run3_env.sh << 'EOF'
export HF_TOKEN=hf_...
export WANDB_API_KEY=wandb_v1_...
EOF

# Upload before starting run:
jl upload <id> /tmp/run3_env.sh /home/.env
```

---

## 11. wandb / Auth Issues

### 11.1 wandb login must happen inside the venv

`wandb login` from the system shell doesn't propagate into the venv that `jl run` creates. The fix in `run_pipeline.sh`:
```bash
if [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login "$WANDB_API_KEY" --relogin 2>&1 | head -3 || true
fi
```
This runs inside the pipeline script, after the venv is activated.

### 11.2 HF_TOKEN scope

The Hugging Face token needs at least **read** access to:
- `iamgroot42/mimir` (MIMIR dataset)
- `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` (or your target model)

If using a gated model, ensure the token has been granted access on the HF model page first.

---

## 12. LLaDA-8B Porting Checklist

Work through this section top-to-bottom before starting a run.

### 12.1 Model identification

```python
MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
# or the base model:
# MODEL_PATH = "GSAI-ML/LLaDA-8B-Base"
```

Verify it loads as a masked LM:
```python
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(model.__class__.__name__)  # Should be something like LLaDAForMaskedLM
```

### 12.2 Find and hardcode `mask_token_id`

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("mask_token_id:", tok.mask_token_id)
# Expected: 126336 for LLaDA — VERIFY THIS
```

Update `config.py`:
```python
# For LLaDA:
MASK_TOKEN_ID = 126336  # verify from the above
```

And pass it explicitly everywhere — do not rely on `tokenizer.mask_token_id` being available from local checkpoints.

### 12.3 GPU requirements for 8B

| GPU | VRAM | Fits LLaDA-8B bf16? | Batch size |
|-----|------|----------------------|-----------|
| L4 | 24 GB | ❌ No (model alone ~16GB, no room for activations) | — |
| A100-40GB | 40 GB | ⚠️ Tight (with grad checkpointing + bs=1) | bs=1, grad_accum=8 |
| A100-80GB | 80 GB | ✅ Comfortable | bs=2, grad_accum=4 |
| H100-80GB | 80 GB | ✅ Comfortable | bs=4, grad_accum=2 |

**Recommended: A100-80GB** (JarvisLabs IN2, ~$120/hr). Each domain will take longer than the 0.6B runs — estimate 6–8 hours per domain.

### 12.4 Batch size and gradient accumulation

Since LLaDA-8B won't support bs=8 on a single GPU, use gradient accumulation to maintain the same effective batch size:

```python
# config.py for LLaDA-8B:
FT_BATCH_SIZE        = 2        # physical batch per GPU step
FT_GRAD_ACCUM_STEPS  = 4        # effective bs = 2 × 4 = 8
FT_LR                = 1e-4
FT_WD                = 0.01
FT_EPOCHS            = 5        # may reduce to 3 — 8B memorises faster
```

Update `finetune.py` to support accumulation:
```python
optimizer.zero_grad()
for accum_step in range(grad_accum_steps):
    loss = mdlm_loss(...) / grad_accum_steps
    loss.backward()
optimizer.step()
```

### 12.5 Tokenizer loading for local checkpoints

LLaDA-8B uses a LLaMA-3 tokenizer. After saving the fine-tuned checkpoint locally, loading with `AutoTokenizer` may fail (same issue as §3.1). Use:
```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(local_checkpoint_path)
# or directly:
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
# then save locally and reload from local — test this first
```

### 12.6 SAMA compatibility with LLaDA

SAMA was originally validated on LLaDA and MDLM, so it should work. However:
- Confirm `compute_nlloss(..., shift_logits=False, mask_id=126336)` produces sensible NLL values (should be roughly 2–8 nats for natural text)
- LLaDA has a longer context window (2048 vs 256) — consider increasing `MAX_LENGTH` to 512 or 1024, but be aware this dramatically increases compute time

### 12.7 Feature extraction compatibility

The `mdlm_metrics_extractor.py` should work for LLaDA with no changes, as it only uses:
- `model(input_ids=noised, attention_mask=mask)` forward pass
- `.logits` output
- `.attentions` output (when `output_attentions=True`)
- `.hidden_states` output (when `output_hidden_states=True`)

These are standard HuggingFace outputs. The only thing to verify: does LLaDA return `attentions` in eager mode?
```python
out = model(input_ids, attention_mask, output_attentions=True, output_hidden_states=True)
print("attentions:", out.attentions is not None)  # must be True
print("hidden_states:", out.hidden_states is not None)  # must be True
```

### 12.8 Update `config.py` for LLaDA

```python
# config.py — LLaDA-8B version
MODEL_PATH     = "GSAI-ML/LLaDA-8B-Instruct"
MASK_TOKEN_ID  = 126336        # VERIFY from tokenizer

FT_LR          = 1e-4
FT_WD          = 0.01
FT_BATCH_SIZE  = 2
FT_GRAD_ACCUM  = 4             # effective bs = 8
FT_EPOCHS      = 5

T_STEPS        = 4
ALPHA_MIN      = 0.05
ALPHA_MAX      = 0.50
SAMA_N_SUBSETS = 128
SAMA_M_TOKENS  = 10
SAMA_MC        = 4

SIG_K          = 8
SIG_GRAD_T     = 2

MAX_LENGTH     = 512            # LLaDA supports longer context — optional increase
MIMIR_N        = 1_000
```

### 12.9 Memory estimation for signal extraction

For LLaDA-8B at MAX_LENGTH=512, each `extract_metrics` call does T×K forward passes:
- T=4, K=8 → 32 forward passes per sample
- Each pass: seq_len=512, model=8B → ~2.5GB activation memory peak
- With `eager` attention + `output_hidden_states=True`, activation memory can be 4–6GB

This means **signal extraction may not fit on A100-40GB** with eager attention on longer sequences. Options:
1. Use A100-80GB or H100-80GB
2. Reduce K from 8 to 4 (halves compute, loses some signal quality)
3. Keep MAX_LENGTH=256 (same as 0.6B runs)

### 12.10 Expected performance ballpark

Based on the Qwen3-0.6B results, expect LLaDA-8B to have:
- **Higher SAMA AUC** (larger model memorises more strongly → clearer signal)
- **Similar or better XGB AUC** (same feature groups, likely stronger ELBO gap)
- **Higher ELBO gap** (8B can fit member documents more precisely in 5 epochs)

If LLaDA-8B SAMA AUC is *lower* than Qwen3-0.6B, that's a red flag — check:
1. Is `mask_token_id` correct?
2. Is `shift_logits=False`?
3. Is `T_STEPS=4` (not 16)?
4. Did fine-tuning actually converge (check `verify_memorization` ELBO gap)?

---

## 13. Quick Reference

### Config values that must match between all 8 stages

```python
T_STEPS        = 4       # SAMA, run_signals, run_sama — must all use same T
ALPHA_MIN      = 0.05    # SAMA and run_signals — must match
ALPHA_MAX      = 0.50    # SAMA and run_signals — must match
SAMA_N_SUBSETS = 128     # run_sama only
SAMA_M_TOKENS  = 10      # run_sama only
SAMA_MC        = 4       # run_attacks (mc_num)
SIG_K          = 8       # run_signals (n_mask_configs)
SIG_GRAD_T     = 2       # run_signals (grad_timesteps)
MAX_LENGTH     = 256     # prepare_data, run_attacks, run_signals — must match
MIMIR_N        = 1_000   # prepare_data
```

### Checklist before launching a new model

- [ ] Model loads with `AutoModelForMaskedLM` (not CausalLM)
- [ ] `trust_remote_code=True` confirmed
- [ ] `mask_token_id` printed and verified (not None)
- [ ] `tokenizer.mask_token_id == MASK_TOKEN_ID` in config
- [ ] `torch_dtype=` (not `dtype=`) in all `from_pretrained` calls
- [ ] `attn_implementation="eager"` in `run_signals.py` model load
- [ ] `shift_logits=False` in all `compute_nlloss` calls
- [ ] `weights_only=False` in all `torch.load` calls
- [ ] `cd "$(dirname "${BASH_SOURCE[0]}")"` at top of `run_pipeline.sh`
- [ ] No `sys.exit(1)` in verify_memorization (changed to warning)
- [ ] `/home/.env` uploaded with `HF_TOKEN` and `WANDB_API_KEY`
- [ ] `MIMIR_HF_NAME` dict correct for new datasets (if adding any)
- [ ] `dllm` (or equivalent) installed `--no-deps`
- [ ] SAMA cloned and `SAMA_ROOT` exported
- [ ] Smoke test: run with `--max_steps 5 --n_samples 20` before full run
- [ ] GPU has enough VRAM for the chosen model + batch size

### Pre-launch smoke test command
```bash
bash run_pipeline.sh github 1  # full run
# OR for smoke test:
python finetune.py --dataset github --max_steps 10
python run_signals.py --dataset github --n_samples 10
```

### Files produced per domain (what to download after run)
```
results/{ds}/
  X.pt                  # features [N, 46]
  y.pt                  # labels [N]
  X_per_group.pt        # ablation slices
  signal_names.pt       # feature names
  raw_signals.pt        # trajectory tensors
  verify_elbo.pt        # per-sample ELBOs
  sama_scores.pt        # SAMA attack scores
  loss_scores.pt        # Loss baseline
  zlib_scores.pt        # Zlib baseline
  ratio_scores.pt       # Ratio baseline
  classifier_results.pt # XGB/MLP/LGB OOF + importances
  ablation_results.pt   # LOSO + solo AUC per group
  benchmark.pt          # final table
```

---

*Last updated: April 2026 — Run 3 Full MIMIR (Qwen3-0.6B-MDLM)*
