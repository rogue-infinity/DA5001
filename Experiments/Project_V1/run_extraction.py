"""
run_extraction.py
=================
Orchestrator for MIA metrics extraction.

Collects all signals from metrics/* in a single shared forward-pass loop:
  - One forward pass per (t, k) cell captures logits + hidden states + attention.
  - Each metric module receives the already-computed tensors — no extra forwards.
  - Gradient norms run in a separate phase (requires_grad) after the main loop.

Usage
-----
    python run_extraction.py
    python run_extraction.py --text "The patient was diagnosed..." \\
                              --timesteps 20 --mask_configs 8 --save bundle.pt
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from metrics import (
    MetricsBundle,
    apply_masking,
    attention_mean,
    compute_baseline_dirs,
    compute_elbo_derivatives,
    compute_elbo_loss,
    cosine_sim_to_baseline,
    forward_with_hooks,
    gradient_norm,
    layer_norms,
    mask_consistency,
    print_summary,
    stack_attention_maps,
    token_entropy,
    aggregate_entropy,
)
from shared import (
    RunLogger,
    build_logger,
    load_model_and_tokenizer,
    select_device,
)

log = build_logger("run_extraction")


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction orchestrator
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_metrics(
    model:              AutoModelForMaskedLM,
    tokenizer:          AutoTokenizer,
    text:               str,
    n_timesteps:        int  = 20,
    n_mask_configs:     int  = 8,
    grad_timesteps:     int  = 4,
    capture_attentions: bool = True,
    seed:               int  = 42,
) -> MetricsBundle:
    """
    Full metrics extraction for a single input text.

    Loop structure
    --------------
    Phase 1 — Baseline:
        One forward pass on the unmasked sequence to get t=0 hidden directions.

    Phase 2 — Main loop (decorated @torch.no_grad()):
        For each t in timestep_grid:
            For each k in n_mask_configs:
                apply_masking → forward_with_hooks → fan out to metric functions
            Aggregate K results for this t.

    Phase 3 — Derivatives (CPU numpy):
        compute_elbo_derivatives(elbo_per_t, timestep_grid)

    Phase 4 — Gradient norms (torch.enable_grad() inside gradient_norm):
        gradient_norm() internally uses torch.enable_grad(), which correctly
        overrides this function's @torch.no_grad() decorator.
    """
    t_start = time.time()
    device  = next(model.parameters()).device
    mask_id = tokenizer.mask_token_id
    rng     = torch.Generator(device=device).manual_seed(seed)

    # ── Phase 1: tokenise + baseline hidden dirs ──────────────────────────────
    token_ids = tokenizer.encode(text, return_tensors="pt").to(device)   # [1, L]
    L = token_ids.size(1)
    log.info("Input: %d tokens  |  text[:60]: %r", L, text[:60])

    _, h0_states, _ = forward_with_hooks(model, token_ids, capture_attentions=False)
    baseline_dirs   = compute_baseline_dirs(h0_states)
    hidden_available = len(baseline_dirs) > 0
    if not hidden_available:
        log.warning("Hidden states not available — hidden_norms and cosine_sim will be zeros.")

    timestep_grid = torch.linspace(1.0 / n_timesteps, 1.0, n_timesteps)  # [T]

    # Accumulators (one entry per timestep after the inner K loop)
    elbo_per_t:         list[float]                    = []
    entropy_per_t:      list[float]                    = []
    entropy_full_per_t: list[torch.Tensor]             = []
    consistency_per_t:  list[float]                    = []
    attn_per_t:         list[Optional[torch.Tensor]]   = []
    hidden_norm_per_t:  list[torch.Tensor]             = []
    cosine_sim_per_t:   list[torch.Tensor]             = []

    # ── Phase 2: main loop ────────────────────────────────────────────────────
    for t_idx, t in enumerate(timestep_grid.tolist()):

        # Per-(t,k) accumulators — reset each timestep
        losses_k:      list[float]               = []
        entropies_k:   list[torch.Tensor]        = []   # each [L]
        predictions_k: list[torch.Tensor]        = []   # each [L] argmax
        attn_k:        list[Optional[torch.Tensor]] = []
        hidden_norm_k: list[torch.Tensor]        = []
        hidden_dir_k:  list[list[torch.Tensor]]  = []   # [K][Nl] mean token dirs

        for k in range(n_mask_configs):
            z_t = apply_masking(token_ids, mask_id, t, rng=rng)
            logits, h_states, attns = forward_with_hooks(
                model, z_t, capture_attentions=capture_attentions
            )
            masked_pos = (z_t[0] == mask_id)   # [L] bool

            if masked_pos.sum() == 0:
                # Edge case: all tokens happen to be unmasked (shouldn't occur at t>0)
                losses_k.append(0.0)
                entropies_k.append(torch.zeros(L, device=device))
                predictions_k.append(token_ids[0].clone())
                attn_k.append(None)
                if h_states:
                    hidden_norm_k.append(layer_norms(h_states))
                    hidden_dir_k.append([h[0].mean(dim=0) if h.ndim == 3 else h.mean(dim=0)
                                         for h in h_states])
                continue

            # ── Signal: ELBO loss ─────────────────────────────────────────────
            losses_k.append(compute_elbo_loss(logits, token_ids[0], masked_pos))

            # ── Signal: prediction entropy ────────────────────────────────────
            entropies_k.append(token_entropy(logits, masked_pos))

            # ── Signal: consistency (needs argmax predictions) ────────────────
            predictions_k.append(logits[0].argmax(dim=-1))

            # ── Signal: attention maps ────────────────────────────────────────
            attn_k.append(attention_mean(attns))   # [Nl, L, L] or None

            # ── Signal: hidden state norms + cosine sim ───────────────────────
            if h_states:
                hidden_norm_k.append(layer_norms(h_states))
                hidden_dir_k.append([
                    h[0].mean(dim=0) if h.ndim == 3 else h.mean(dim=0)
                    for h in h_states
                ])

        # ── Aggregate K results for this timestep ─────────────────────────────

        # ELBO
        elbo_per_t.append(float(np.mean(losses_k)))

        # Entropy
        ent_stack = torch.stack(entropies_k).mean(dim=0)   # [L]
        entropy_full_per_t.append(ent_stack)
        entropy_per_t.append(aggregate_entropy(ent_stack))

        # Consistency
        consistency_per_t.append(mask_consistency(predictions_k))

        # Attention: mean across K configs
        valid_attn = [a for a in attn_k if a is not None]
        attn_per_t.append(
            torch.stack(valid_attn).mean(dim=0) if valid_attn else None
        )

        # Hidden norms
        if hidden_norm_k and all(h.numel() > 0 for h in hidden_norm_k):
            hidden_norm_per_t.append(torch.stack(hidden_norm_k).mean(dim=0))
        else:
            hidden_norm_per_t.append(torch.zeros(1, device=device))

        # Cosine similarity
        cosine_sim_per_t.append(
            cosine_sim_to_baseline(hidden_dir_k, baseline_dirs, device)
            if hidden_available else torch.zeros(1, device=device)
        )

        if (t_idx + 1) % 5 == 0 or t_idx == 0:
            log.info(
                "  t=%.3f | L(t)=%.4f | H=%.4f | consist=%.4f",
                t, elbo_per_t[-1], entropy_per_t[-1], consistency_per_t[-1],
            )

    # ── Phase 3: convert accumulators → tensors ───────────────────────────────
    elbo_tensor        = torch.tensor(elbo_per_t,        dtype=torch.float32)
    entropy_tensor     = torch.tensor(entropy_per_t,     dtype=torch.float32)
    entropy_full_mat   = torch.stack(entropy_full_per_t)          # [T, L]
    consistency_tensor = torch.tensor(consistency_per_t, dtype=torch.float32)
    hidden_norm_mat    = torch.stack(hidden_norm_per_t)           # [T, Nl]
    cosine_sim_mat     = torch.stack(cosine_sim_per_t)            # [T, Nl]

    dldt, d2ldt2 = compute_elbo_derivatives(elbo_tensor, timestep_grid)
    attn_tensor  = stack_attention_maps(attn_per_t)               # [T, Nl, L, L] or None

    # ── Phase 4: gradient norms (enable_grad inside gradient_norm) ───────────
    grad_t_indices = np.linspace(0, n_timesteps - 1, grad_timesteps, dtype=int)
    grad_t_grid    = timestep_grid[grad_t_indices]
    log.info("Computing gradient norms at %d timesteps …", grad_timesteps)

    grad_norm_list = [
        gradient_norm(model, token_ids, mask_id, float(timestep_grid[i].item()), rng=rng)
        for i in grad_t_indices
    ]
    grad_norms = torch.tensor(grad_norm_list, dtype=torch.float32)

    elapsed = time.time() - t_start
    log.info("Extraction complete in %.1fs", elapsed)

    return MetricsBundle(
        text                   = text,
        token_ids              = token_ids[0].tolist(),
        seq_len                = L,
        timestep_grid          = timestep_grid,
        elbo_per_t             = elbo_tensor,
        dldt                   = dldt,
        d2ldt2                 = d2ldt2,
        elbo_variance          = float(elbo_tensor.var().item()),
        pred_entropy_per_t     = entropy_tensor,
        pred_entropy_full      = entropy_full_mat,
        mask_consistency_per_t = consistency_tensor,
        attention_maps         = attn_tensor,
        hidden_norms           = hidden_norm_mat,
        hidden_cosine_sim      = cosine_sim_mat,
        grad_t_grid            = grad_t_grid,
        grad_norms             = grad_norms,
        elapsed_seconds        = elapsed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract MDLM MIA signals from a text input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",          default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1")
    p.add_argument("--text",
                   default="The patient was diagnosed with a rare form of lymphoma and "
                           "prescribed an experimental immunotherapy protocol.")
    p.add_argument("--timesteps",      type=int,  default=20)
    p.add_argument("--mask_configs",   type=int,  default=8)
    p.add_argument("--grad_timesteps", type=int,  default=4)
    p.add_argument("--no_attentions",  action="store_true",
                   help="Skip attention capture (faster, less memory).")
    p.add_argument("--save",           type=str,  default=None,
                   help="Path to save MetricsBundle via torch.save.")
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--debug",          action="store_true")
    return p.parse_args()


def main() -> None:
    import logging
    args   = parse_args()
    if args.debug:
        log.setLevel(logging.DEBUG)

    device = select_device()
    log.info("Model: %s  |  Device: %s", args.model, device)
    log.info("T=%d  K=%d  grad_T=%d  attn=%s",
             args.timesteps, args.mask_configs, args.grad_timesteps, not args.no_attentions)

    # Load with eager attention when capturing attention maps
    attn_impl = "eager" if not args.no_attentions else None
    model, tokenizer = load_model_and_tokenizer(args.model, device, attn_impl)

    bundle = extract_metrics(
        model              = model,
        tokenizer          = tokenizer,
        text               = args.text,
        n_timesteps        = args.timesteps,
        n_mask_configs     = args.mask_configs,
        grad_timesteps     = args.grad_timesteps,
        capture_attentions = not args.no_attentions,
        seed               = args.seed,
    )

    print_summary(bundle)

    save_path = None
    if args.save:
        save_path = Path(args.save)
        torch.save(bundle, save_path)
        log.info("Bundle saved → %s", save_path)

    RunLogger().log_extraction(
        model_name     = args.model,
        text           = args.text,
        n_timesteps    = args.timesteps,
        n_mask_configs = args.mask_configs,
        grad_timesteps = args.grad_timesteps,
        bundle         = bundle,
        save_path      = save_path,
    )


if __name__ == "__main__":
    main()
