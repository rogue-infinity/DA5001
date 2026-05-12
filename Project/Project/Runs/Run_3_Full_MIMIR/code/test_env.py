"""
test_env.py — Pre-flight environment check for Run_3_Full_MIMIR.
Run this before launching JarvisLabs to confirm every dependency + local
module loads correctly and the model/tokenizer is reachable.

Usage:
    cd Run_3_Full_MIMIR/code/
    python test_env.py               # full check (loads model)
    python test_env.py --no_model    # skip HF model download (fast)
    python test_env.py --no_wandb    # skip wandb login check
"""

import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

PASS, FAIL, SKIP = "✅", "❌", "⏭️ "

results = []

def check(name, fn):
    try:
        msg = fn()
        results.append((PASS, name, msg or ""))
        print(f"  {PASS}  {name}{(' — ' + msg) if msg else ''}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL}  {name} — {e}")

def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no_model",  action="store_true", help="Skip HF model download")
parser.add_argument("--no_wandb",  action="store_true", help="Skip wandb check")
parser.add_argument("--no_sama",   action="store_true", help="Skip SAMA import check")
args = parser.parse_args()

print("\n🔍  Run_3_Full_MIMIR — Environment Pre-flight Check\n")

# ── 1. Core Python deps ───────────────────────────────────────
section("1 · Core dependencies")

check("torch",           lambda: __import__("torch") and f"v{__import__('torch').__version__}")
check("numpy",           lambda: __import__("numpy") and f"v{__import__('numpy').__version__}")
check("transformers",    lambda: __import__("transformers") and f"v{__import__('transformers').__version__}")
check("datasets",        lambda: __import__("datasets") and f"v{__import__('datasets').__version__}")
check("accelerate",      lambda: __import__("accelerate") and f"v{__import__('accelerate').__version__}")
check("scikit-learn",    lambda: __import__("sklearn") and f"v{__import__('sklearn').__version__}")
check("xgboost",         lambda: __import__("xgboost") and f"v{__import__('xgboost').__version__}")
check("lightgbm",        lambda: __import__("lightgbm") and f"v{__import__('lightgbm').__version__}")
check("wandb",           lambda: __import__("wandb") and f"v{__import__('wandb').__version__}")
check("tqdm",            lambda: __import__("tqdm") and "ok")
check("huggingface_hub", lambda: __import__("huggingface_hub") and "ok")
check("sentencepiece",   lambda: __import__("sentencepiece") and "ok")
check("tabulate",        lambda: __import__("tabulate") and "ok")
check("matplotlib",      lambda: __import__("matplotlib") and f"v{__import__('matplotlib').__version__}")
check("seaborn",         lambda: __import__("seaborn") and f"v{__import__('seaborn').__version__}")

# ── 2. Optional heavy deps ────────────────────────────────────
section("2 · Optional deps")

check("umap-learn", lambda: __import__("umap") and "ok")
check("peft",       lambda: __import__("peft") and f"v{__import__('peft').__version__}")

# ── 3. GPU / device ───────────────────────────────────────────
section("3 · Device")

def gpu_check():
    import torch
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        return f"{n}× GPU — {', '.join(names)}"
    elif torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU only"

check("device",      gpu_check)
check("bf16 support",lambda: __import__("torch").cuda.is_bf16_supported() if __import__("torch").cuda.is_available() else "N/A (no CUDA)")

# ── 4. Local modules ──────────────────────────────────────────
section("4 · Local modules (Run_3 code)")

check("config.py",                lambda: __import__("config") and f"{len(__import__('config').ALL_DATASETS)} datasets configured")
check("mdlm_metrics_extractor",   lambda: __import__("mdlm_metrics_extractor") and "ok")
check("dllm package",             lambda: __import__("dllm") and "ok")

# ── 5. SAMA ───────────────────────────────────────────────────
section("5 · SAMA repo")

if args.no_sama:
    print(f"  {SKIP}  SAMA import check skipped (--no_sama)")
else:
    def sama_check():
        import types, sys, os
        candidates = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../SAMA")),
            "/home/SAMA",
            os.path.abspath("SAMA"),
        ]
        sama_root = next((p for p in candidates if os.path.isdir(os.path.join(p, "attack"))), None)
        if not sama_root:
            raise RuntimeError(f"SAMA not found. Checked: {candidates}")
        sys.path.insert(0, sama_root)
        sys.path.insert(0, os.path.join(sama_root, "attack"))
        # Stub attack.run so SamaAttack import doesn't fail without tabulate/models
        if "attack.run" not in sys.modules:
            stub = types.ModuleType("attack.run")
            stub.init_model = lambda *a, **kw: (None, None, None)
            sys.modules["attack.run"] = stub
        from attack.attacks.sama import SamaAttack  # noqa
        return f"found at {sama_root}"

    check("SAMA repo + SamaAttack", sama_check)

# ── 6. Config sanity ──────────────────────────────────────────
section("6 · Config values")

def cfg_check():
    import config as c
    assert c.T_STEPS == 4,           f"T_STEPS={c.T_STEPS}, expected 4"
    assert c.ALPHA_MIN == 0.05,      f"ALPHA_MIN={c.ALPHA_MIN}"
    assert c.ALPHA_MAX == 0.50,      f"ALPHA_MAX={c.ALPHA_MAX}"
    assert c.FT_LR == 1e-4,          f"FT_LR={c.FT_LR}"
    assert c.FT_EPOCHS == 5,         f"FT_EPOCHS={c.FT_EPOCHS}"
    assert c.FT_BATCH_SIZE == 8,     f"FT_BATCH_SIZE={c.FT_BATCH_SIZE}"
    assert c.SAMA_N_SUBSETS == 128,  f"SAMA_N_SUBSETS={c.SAMA_N_SUBSETS}"
    assert c.SAMA_MC == 4,           f"SAMA_MC={c.SAMA_MC}"
    assert len(c.ALL_DATASETS) == 9, f"ALL_DATASETS has {len(c.ALL_DATASETS)} entries"
    return "T=4, α=[5%,50%], lr=1e-4, wd=0.01, bs=8, 5ep, 6 MIMIR datasets"

check("config values", cfg_check)

# ── 7. Tokenizer + model (optional) ──────────────────────────
section("7 · HuggingFace model")

if args.no_model:
    print(f"  {SKIP}  Model load skipped (--no_model)")
else:
    def tok_check():
        from transformers import AutoTokenizer
        import config as c
        model_path = os.environ.get("MODEL_PATH", c.MODEL_PATH)
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        mid = tok.mask_token_id
        assert mid is not None, "mask_token_id is None"
        return f"mask_token_id={mid}"

    def model_check():
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import config as c
        model_path = os.environ.get("MODEL_PATH", c.MODEL_PATH)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        m = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        return f"{n_params:.0f}M params, dtype={dtype}"

    check("tokenizer loads",  tok_check)
    check("model loads",      model_check)

# ── 8. Quick metrics extractor smoke test ─────────────────────
section("8 · Metrics extractor smoke test (T=2, K=2)")

def smoke_extract():
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from mdlm_metrics_extractor import extract_metrics
    import config as c

    model_path = os.environ.get("MODEL_PATH", c.MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    tok   = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()

    text = "The neural network was trained on a large corpus of text data."
    t0 = time.time()
    bundle = extract_metrics(model, tok, text,
                             n_timesteps=2, n_mask_configs=2,
                             grad_timesteps=1, capture_attentions=True,
                             alpha_min=c.ALPHA_MIN, alpha_max=c.ALPHA_MAX)
    fv = bundle.to_feature_vector()
    elapsed = time.time() - t0

    expected_dim = 11 * 2 + 1  # T=2 → 23 (without cross-model cosine)
    assert fv.shape[0] == expected_dim, f"feature dim={fv.shape[0]}, expected {expected_dim}"
    assert torch.isfinite(fv).all(),    "feature vector contains NaN/Inf"
    return f"fv shape={fv.shape}, finite=True, elapsed={elapsed:.1f}s"

if args.no_model:
    print(f"  {SKIP}  Smoke test skipped (--no_model)")
else:
    check("extract_metrics (T=2, K=2)", smoke_extract)

# ── 9. wandb ─────────────────────────────────────────────────
section("9 · Weights & Biases")

if args.no_wandb:
    print(f"  {SKIP}  wandb check skipped (--no_wandb)")
else:
    def wandb_check():
        import wandb
        status = wandb.api.api_key
        if not status:
            raise RuntimeError("Not logged in — run: wandb login")
        return f"logged in as {wandb.api.viewer()['entity']}"
    check("wandb login", wandb_check)

# ── Summary ───────────────────────────────────────────────────
section("Summary")
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"\n  {PASS} Passed: {passed}   {FAIL} Failed: {failed}\n")

if failed:
    print("  Failed checks:")
    for status, name, msg in results:
        if status == FAIL:
            print(f"    • {name}: {msg}")
    sys.exit(1)
else:
    print("  All checks passed — ready to launch! 🚀")
