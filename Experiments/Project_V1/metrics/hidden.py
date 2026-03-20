"""
metrics/hidden.py
=================
Hidden-state geometry signals.

Signals:
  - ‖h_l‖₂ : mean L2 norm over token dimension per layer.
  - cosine similarity of mean token direction to the t=0 (unmasked) baseline.

Both signals track how much the internal representation changes as the
masking ratio increases.

No project-level imports.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def layer_norms(hidden_states: list[torch.Tensor]) -> torch.Tensor:
    """
    Mean ‖h‖₂ over the token dimension for each layer.

    Handles two shapes emitted by different model variants:
        [1, L, d]  — standard HuggingFace output_hidden_states
        [L, d]     — hook-captured tensors (no batch dim)

    Returns:
        [Nl] — one scalar per layer.
    """
    norms = []
    for h in hidden_states:
        h2d = h[0] if h.ndim == 3 else h    # [L, d]
        norms.append(h2d.norm(dim=-1).mean())
    return torch.stack(norms)


def compute_baseline_dirs(hidden_states: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Extract mean token direction [d_model] per layer from an unmasked pass.

    Call once on the output of forward_with_hooks(token_ids, capture_attentions=False)
    to get the t=0 reference directions.

    Returns:
        list of Nl tensors each [d_model].
    """
    dirs = []
    for h in hidden_states:
        h2d = h[0] if h.ndim == 3 else h    # [L, d]
        dirs.append(h2d.mean(dim=0))          # [d]
    return dirs


def cosine_sim_to_baseline(
    hidden_dir_k:  list[list[torch.Tensor]],  # [K, Nl] mean token dirs at current t
    baseline_dirs: list[torch.Tensor],         # [Nl] mean token dirs at t=0
    device:        torch.device,
) -> torch.Tensor:
    """
    Cosine similarity between the current mean token direction and the t=0
    baseline, per layer.

    Args:
        hidden_dir_k  : For each of K mask configs, a list of Nl per-layer
                        mean token directions [d_model].
        baseline_dirs : Nl per-layer mean token directions from the unmasked
                        forward pass (output of compute_baseline_dirs).
        device        : Target device for the output tensor.

    Returns:
        [Nl] — cosine similarity per layer, averaged across K configs.
        Returns zeros tensor [1] if either input is empty.
    """
    if not baseline_dirs or not hidden_dir_k:
        return torch.zeros(1, device=device)

    Nl = len(baseline_dirs)

    # Average hidden dirs across K configs for each layer
    mean_dirs = []
    for l in range(Nl):
        layer_dirs = [
            hidden_dir_k[k][l]
            for k in range(len(hidden_dir_k))
            if l < len(hidden_dir_k[k])
        ]
        if layer_dirs:
            mean_dirs.append(torch.stack(layer_dirs).mean(dim=0))  # [d]
        else:
            mean_dirs.append(baseline_dirs[l])   # fallback to baseline

    cos_sims = [
        F.cosine_similarity(h0.unsqueeze(0), ht.unsqueeze(0)).item()
        for h0, ht in zip(baseline_dirs, mean_dirs)
    ]
    return torch.tensor(cos_sims, device=device)  # [Nl]
