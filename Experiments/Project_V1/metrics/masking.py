"""
metrics/masking.py
==================
Random masking of input token sequences.
No project-level imports.
"""

from __future__ import annotations

from typing import Optional

import torch


def apply_masking(
    token_ids:  torch.Tensor,        # [1, L]
    mask_id:    int,
    mask_ratio: float,               # fraction of tokens to mask ∈ (0, 1]
    rng:        Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Replace a random `mask_ratio` fraction of positions with `mask_id`.

    Sampling is without replacement.  Returns a new [1, L] tensor;
    the input is not modified.
    """
    L      = token_ids.size(1)
    n_mask = max(1, int(round(mask_ratio * L)))
    perm   = torch.randperm(L, generator=rng, device=token_ids.device)
    z_t    = token_ids.clone()
    z_t[0, perm[:n_mask]] = mask_id
    return z_t
