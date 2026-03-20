"""
metrics/gradients.py
====================
Gradient norm signal — ‖∇_θ L(x₀, t)‖₂.

This module owns the full gradient computation lifecycle:
  1. Apply masking at the requested timestep.
  2. Enable grad on all parameters.
  3. Forward pass → CE loss at masked positions → backward.
  4. Compute total gradient norm.
  5. Zero grad and disable requires_grad.

torch.enable_grad() is used as a context manager inside gradient_norm(),
which correctly overrides any outer @torch.no_grad() decorator on the caller.
This is the standard PyTorch pattern for mixing no-grad and grad regions.

Imports from: metrics.forward, metrics.masking, shared.logger
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from metrics.forward import forward_with_hooks
from metrics.masking import apply_masking
from shared.logger import build_logger

_log = build_logger("metrics.gradients")


def gradient_norm(
    model:      AutoModelForMaskedLM,
    token_ids:  torch.Tensor,          # [1, L] original unmasked ids
    mask_id:    int,
    mask_ratio: float,                 # t value (= fraction to mask)
    rng:        Optional[torch.Generator] = None,
) -> float:
    """
    Compute ‖∇_θ L(x₀, t)‖₂ at a single timestep.

    Returns the total gradient norm as a float.  Returns 0.0 if no positions
    are masked.

    Memory note: activations are released before backward() via explicit del.
    """
    z_t = apply_masking(token_ids, mask_id, mask_ratio, rng=rng)

    model.zero_grad()
    for p in model.parameters():
        p.requires_grad_(True)

    try:
        with torch.enable_grad():
            logits_g, _, _ = forward_with_hooks(model, z_t, capture_attentions=False)

            masked_pos = (z_t[0] == mask_id)
            if masked_pos.sum() == 0:
                return 0.0

            target   = token_ids[0]
            log_p    = F.log_softmax(logits_g[0], dim=-1)
            loss     = F.nll_loss(log_p, target, reduction="none")[masked_pos].mean()

            # Release activation graph before backward to reduce peak memory
            del logits_g
            loss.backward()

        total_norm = torch.sqrt(
            sum(
                p.grad.norm() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
        ).item()
    finally:
        model.zero_grad()
        for p in model.parameters():
            p.requires_grad_(False)

    _log.info("  t=%.3f | ‖∇_θ L‖ = %.4f", mask_ratio, total_norm)
    return total_norm
