"""
smoke_test.py
=============
End-to-end smoke test for the MDLM signal extraction pipeline.

Loads the model from ./dllm/ (local weights), reloads with eager attention,
runs extract_metrics() on 2 hardcoded strings with minimal settings (T=3,
K=2, grad_timesteps=1), logs every signal to wandb, and prints PASS/FAIL
for each signal group based on the shape annotations in MetricsBundle.

Usage
-----
    python smoke_test.py
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import torch
import wandb
from transformers import AutoModelForMaskedLM

from mdlm_metrics_extractor import MetricsBundle, extract_metrics
from mdlm_qwen3_test import load_model_and_tokenizer, select_device

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = (
    "/Users/skumar/.cache/huggingface/hub"
    "/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1"
    "/snapshots/39c33255701becfc61f3052d4b86c80b6d36603f"
)

TEXTS = [
    "The patient was diagnosed with a rare form of lymphoma.",
    "Masked diffusion language models learn to reconstruct corrupted sequences.",
]

T      = 3   # timesteps  — enough to exercise every signal path
K      = 2   # mask configs per timestep
GRAD_T = 1   # grad timesteps (expensive; 1 is sufficient for smoke)


# ── PASS/FAIL checks ───────────────────────────────────────────────────────────

def _ok(x: Optional[torch.Tensor]) -> bool:
    """True iff x is a finite, non-None tensor."""
    return x is not None and bool(x.isfinite().all())

def _nonzero(x: torch.Tensor) -> bool:
    return bool(x.abs().sum() > 0)


def check_bundle(bundle: MetricsBundle) -> dict[str, bool]:
    """
    PASS/FAIL for each signal group.
    Shapes are taken directly from the MetricsBundle docstring.
    """
    results: dict[str, bool] = {}

    # ── ELBO trajectory  [T] ─────────────────────────────────────────────────
    et = bundle.elbo_per_t
    results["elbo_per_t"] = (
        et.shape == (T,) and _ok(et) and _nonzero(et)
    )

    # ── ELBO variance  (scalar float) ────────────────────────────────────────
    ev = torch.tensor(bundle.elbo_variance)
    results["elbo_variance"] = bool(ev.isfinite())

    # ── Loss curve shape  dL/dt, d²L/dt²  [T] ────────────────────────────────
    results["loss_shape"] = (
        bundle.dldt.shape == (T,)
        and bundle.d2ldt2.shape == (T,)
        and _ok(bundle.dldt)
        and _ok(bundle.d2ldt2)
    )

    # ── Prediction entropy  [T] ───────────────────────────────────────────────
    pe = bundle.pred_entropy_per_t
    results["pred_entropy"] = (
        pe.shape == (T,) and _ok(pe) and _nonzero(pe)
    )

    # ── Multi-mask consistency  [T] ───────────────────────────────────────────
    mc = bundle.mask_consistency_per_t
    results["mask_consistency"] = (
        mc.shape == (T,) and _ok(mc)
    )

    # ── Hidden state norms  [T, Nl] ───────────────────────────────────────────
    hn = bundle.hidden_norms
    results["hidden_norms"] = (
        hn.ndim == 2 and hn.shape[0] == T and _ok(hn) and _nonzero(hn)
    )

    # ── Hidden cosine similarity  [T, Nl] ─────────────────────────────────────
    hc = bundle.hidden_cosine_sim
    results["hidden_cosine_sim"] = (
        hc.ndim == 2 and hc.shape[0] == T and _ok(hc)
    )

    # ── AttenMIA: per-head entropy  [T] ──────────────────────────────────────
    ae = bundle.attn_entropy_per_t
    if ae is not None:
        results["attn_entropy"] = (
            ae.shape == (T,) and _ok(ae) and _nonzero(ae)
        )
    else:
        results["attn_entropy"] = False

    # ── AttenMIA: cross-layer correlation  [T] ────────────────────────────────
    cl = bundle.attn_crosslayer_per_t
    if cl is not None:
        results["attn_crosslayer"] = (
            cl.shape == (T,) and _ok(cl)
        )
    else:
        results["attn_crosslayer"] = False

    # ── AttenMIA: barycentric drift  [T] ──────────────────────────────────────
    bary = bundle.attn_barycenter_per_t
    if bary is not None:
        results["attn_barycenter"] = (
            bary.shape == (T,) and _ok(bary)
        )
    else:
        results["attn_barycenter"] = False

    # ── AttenMIA: perturbation variance  [T] ──────────────────────────────────
    ap = bundle.attn_perturbation_per_t
    if ap is not None:
        results["attn_perturbation"] = (
            ap.shape == (T,) and _ok(ap)
        )
    else:
        results["attn_perturbation"] = False

    # ── Gradient norms  [GRAD_T] ──────────────────────────────────────────────
    gn = bundle.grad_norms
    results["grad_norms"] = (
        gn.shape == (GRAD_T,) and _ok(gn) and _nonzero(gn)
    )

    # ── Feature vector  [11*T + 1] ────────────────────────────────────────────
    fv = bundle.to_feature_vector()
    expected_dim = 11 * T + 1
    results["feature_vector"] = (
        fv.shape == (expected_dim,) and _ok(fv) and fv.norm() > 0
    )

    return results


# ── wandb logging ──────────────────────────────────────────────────────────────

def log_bundle_to_wandb(bundle: MetricsBundle, prefix: str) -> None:
    """
    Log every wandb-loggable scalar from bundle under the given prefix.
    Per-t signals are logged as separate keys t0/t1/…
    """
    d: dict[str, float] = {}

    # Scalar
    d[f"{prefix}/elbo_variance"] = float(bundle.elbo_variance)

    for i in range(T):
        tag = f"t{i}"
        d[f"{prefix}/elbo_per_t/{tag}"]            = bundle.elbo_per_t[i].item()
        d[f"{prefix}/pred_entropy_per_t/{tag}"]    = bundle.pred_entropy_per_t[i].item()
        d[f"{prefix}/mask_consistency_per_t/{tag}"]= bundle.mask_consistency_per_t[i].item()
        d[f"{prefix}/hidden_norms_mean/{tag}"]     = bundle.hidden_norms[i].float().mean().item()
        d[f"{prefix}/hidden_cosine_sim_mean/{tag}"]= bundle.hidden_cosine_sim[i].float().mean().item()

        if bundle.attn_entropy_per_t is not None:
            d[f"{prefix}/attn_entropy_per_t/{tag}"]     = bundle.attn_entropy_per_t[i].item()
        if bundle.attn_crosslayer_per_t is not None:
            d[f"{prefix}/attn_crosslayer_per_t/{tag}"]  = bundle.attn_crosslayer_per_t[i].item()
        if bundle.attn_barycenter_per_t is not None:
            d[f"{prefix}/attn_barycenter_per_t/{tag}"]  = bundle.attn_barycenter_per_t[i].item()
        if bundle.attn_perturbation_per_t is not None:
            d[f"{prefix}/attn_perturbation_per_t/{tag}"]= bundle.attn_perturbation_per_t[i].item()

    # Gradient norms (mean over the GRAD_T subset)
    d[f"{prefix}/grad_norms_mean"] = bundle.grad_norms.float().mean().item()

    # Feature vector summary
    fv = bundle.to_feature_vector()
    d[f"{prefix}/feature_vector_norm"] = fv.norm().item()
    d[f"{prefix}/feature_vector_dim"]  = float(fv.shape[0])

    wandb.log(d)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Device + model load ────────────────────────────────────────────────
    device = select_device()

    log.info("Loading model and tokenizer from %s …", MODEL_PATH)
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, device)

    # Reload with attn_implementation="eager" so attention tensors are returned.
    # Exact pattern from mdlm_metrics_extractor.py:main() lines 1182-1189.
    log.info("Reloading with attn_implementation='eager' for attention capture …")
    dtype = next(model.parameters()).dtype
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()

    # ── 2. Init wandb ─────────────────────────────────────────────────────────
    run = wandb.init(project="da5001-mia", name="smoke-test")
    log.info("wandb run: %s", run.url)

    # ── 3-5. Extract, log, check each sample ──────────────────────────────────
    all_results: list[dict[str, bool]] = []

    for i, text in enumerate(TEXTS):
        sample_id = i + 1
        log.info("")
        log.info("══ Sample %d / %d ══════════════════════════════", sample_id, len(TEXTS))
        log.info("  text: %r", text[:70])

        bundle = extract_metrics(
            model             = model,
            tokenizer         = tokenizer,
            text              = text,
            n_timesteps       = T,
            n_mask_configs    = K,
            grad_timesteps    = GRAD_T,
            capture_attentions= True,
            seed              = 42,
        )

        fv = bundle.to_feature_vector()
        log.info("  feature vector: shape=%s  norm=%.4f", tuple(fv.shape), fv.norm().item())

        log_bundle_to_wandb(bundle, prefix=f"sample_{sample_id}")

        checks = check_bundle(bundle)
        all_results.append(checks)

        print(f"\n{'─'*50}")
        print(f"  Sample {sample_id} PASS/FAIL  (T={T}, K={K}, grad_T={GRAD_T})")
        print(f"{'─'*50}")
        for name, ok in checks.items():
            status = "PASS" if ok else "FAIL"
            # Attach shape info for context
            extra = ""
            if name == "hidden_norms":
                extra = f"  shape={tuple(bundle.hidden_norms.shape)}"
            elif name == "hidden_cosine_sim":
                extra = f"  shape={tuple(bundle.hidden_cosine_sim.shape)}"
            elif name == "attn_entropy" and bundle.attn_entropy_per_t is not None:
                extra = f"  shape={tuple(bundle.attn_entropy_per_t.shape)}"
            elif name == "grad_norms":
                extra = f"  shape={tuple(bundle.grad_norms.shape)}"
            elif name == "feature_vector":
                fv_shape = tuple(bundle.to_feature_vector().shape)
                extra = f"  shape={fv_shape}  norm={bundle.to_feature_vector().norm():.4f}"
            print(f"  [{status}] {name}{extra}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    all_pass = all(all(r.values()) for r in all_results)
    if all_pass:
        print("  OVERALL: ALL PASS")
    else:
        print("  OVERALL: FAILURES DETECTED")
        for i, r in enumerate(all_results):
            failed = [k for k, v in r.items() if not v]
            if failed:
                print(f"  Sample {i+1} failures: {', '.join(failed)}")
    print(f"{'═'*50}\n")

    run.finish()


if __name__ == "__main__":
    main()
