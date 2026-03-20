"""
metrics/consistency.py
======================
Multi-mask consistency signal.

Signal: agreement fraction across K independent mask configurations at the
same timestep t.  High agreement ↔ robust predictions; low agreement ↔
the model is sensitive to which tokens are masked.

No project-level imports.
"""

from __future__ import annotations

import torch


def mask_consistency(predictions_k: list[torch.Tensor]) -> float:
    """
    Compute mode-agreement fraction across K argmax predictions.

    For each token position, find the most-common predicted token id
    across all K configurations.  Consistency = mean fraction of configs
    that agree with the mode.

    Args:
        predictions_k : list of K tensors each [L] — argmax token id
                        predicted by the model under each mask config.

    Returns:
        float in [0, 1] — 1.0 means all K configs give identical predictions
        at every position.
    """
    if not predictions_k:
        return 0.0

    preds   = torch.stack(predictions_k)            # [K, L]
    mode    = preds.cpu().mode(dim=0).values.to(preds.device)  # [L]
    agree   = (preds == mode.unsqueeze(0)).float()  # [K, L]
    return agree.mean().item()
