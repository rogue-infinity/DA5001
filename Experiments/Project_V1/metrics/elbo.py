"""
metrics/elbo.py
===============
ELBO loss trajectory and finite-difference derivatives.

Signals computed here:
  - L(t)   : mean cross-entropy at masked positions (one forward pass)
  - dL/dt  : first derivative via central finite differences
  - d²L/dt²: second derivative
  - Var_t[L(t)]: variance over the timestep grid

No project-level imports.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_elbo_loss(
    logits:           torch.Tensor,   # [1, L, V]
    target_ids:       torch.Tensor,   # [L]   original unmasked token ids
    masked_positions: torch.Tensor,   # bool  [L]
) -> float:
    """
    Mean cross-entropy at masked positions for one (z_t, x_0) pair.

    Returns 0.0 if no positions are masked (shouldn't happen in normal use
    but guards against edge cases at t≈0).
    """
    if masked_positions.sum() == 0:
        return 0.0
    log_p   = F.log_softmax(logits[0], dim=-1)           # [L, V]
    ce_full = F.nll_loss(log_p, target_ids, reduction="none")  # [L]
    return ce_full[masked_positions].mean().item()


def compute_elbo_derivatives(
    elbo_per_t:    torch.Tensor,   # [T]  L(t) values
    timestep_grid: torch.Tensor,   # [T]  t values
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Finite-difference derivatives of L(t) over the timestep axis.

    Uses numpy.gradient (central differences for interior points,
    first-order one-sided at boundaries).

    Returns:
        dldt   : [T]  dL/dt
        d2ldt2 : [T]  d²L/dt²
    """
    t_np      = timestep_grid.detach().cpu().numpy()
    elbo_np   = elbo_per_t.detach().cpu().numpy()
    dldt_np   = np.gradient(elbo_np,  t_np)
    d2ldt2_np = np.gradient(dldt_np,  t_np)
    return (
        torch.from_numpy(dldt_np.copy()).float(),
        torch.from_numpy(d2ldt2_np.copy()).float(),
    )
