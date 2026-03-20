"""
metrics/entropy.py
==================
Per-token prediction entropy signal.

Signal: H(p_θ(·|z_t)) — Shannon entropy of the model's softmax distribution
at each masked token position.  High entropy ↔ uncertain prediction;
low entropy ↔ confident prediction.

No project-level imports.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_entropy(
    logits:           torch.Tensor,   # [1, L, V]
    masked_positions: torch.Tensor,   # bool [L]
) -> torch.Tensor:
    """
    Compute Shannon entropy H(softmax(logits[pos])) at each masked position.

    Non-masked positions are zeroed out so the caller can safely mean/sum
    over the full [L] vector without contamination.

    Returns:
        [L] — entropy per position (0 at non-masked positions).
    """
    probs     = F.softmax(logits[0], dim=-1)        # [L, V]
    log_probs = F.log_softmax(logits[0], dim=-1)    # [L, V]
    H         = -(probs * log_probs).sum(dim=-1)    # [L]
    return H * masked_positions.float()


def aggregate_entropy(entropy_vec: torch.Tensor) -> float:
    """
    Scalar summary for one timestep: mean entropy over non-zero positions.

    Args:
        entropy_vec : [L] — output of token_entropy().

    Returns:
        float — mean over masked positions; 0.0 if all zeros.
    """
    nonzero = entropy_vec[entropy_vec > 0]
    return nonzero.mean().item() if nonzero.numel() > 0 else 0.0
