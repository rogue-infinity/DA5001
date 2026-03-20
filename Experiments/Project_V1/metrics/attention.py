"""
metrics/attention.py
====================
Attention-map aggregation signals.

Signal: per-layer attention weights averaged over heads, collected at each
timestep t.  Captures how attention patterns shift as more tokens are masked.

No project-level imports.
"""

from __future__ import annotations

from typing import Optional

import torch


def attention_mean(
    attentions: list[torch.Tensor],   # Nl tensors each [1, Nh, L, L]
) -> Optional[torch.Tensor]:
    """
    Average attention weights over heads for each layer.

    Args:
        attentions : list of Nl tensors each [1, Nh, L, L].

    Returns:
        [Nl, L, L] — head-averaged maps, or None if list is empty.
    """
    if not attentions:
        return None
    return torch.stack([a[0].mean(dim=0) for a in attentions])  # [Nl, L, L]


def stack_attention_maps(
    attn_per_t: list[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    """
    Stack per-timestep attention maps into a single tensor.

    Args:
        attn_per_t : list of T entries, each [Nl, L, L] or None.

    Returns:
        [T, Nl, L, L] if all entries are non-None, else None.
    """
    if all(a is not None for a in attn_per_t):
        return torch.stack(attn_per_t)    # [T, Nl, L, L]
    return None
