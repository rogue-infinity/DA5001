"""
metrics/bundle.py
=================
MetricsBundle — typed container for all MIA signals extracted from one text.
Also provides print_summary() for human-readable display.
No project-level imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MetricsBundle:
    """
    All signals for a single input sequence.

    Shape abbreviations:
        T   = number of timestep samples
        L   = sequence length (tokens, no padding)
        K   = independent mask configurations per timestep
        Nl  = number of transformer layers
        Nh  = number of attention heads
    """

    # ── Input ──────────────────────────────────────────────────────────────
    text:       str
    token_ids:  list[int]
    seq_len:    int                   # L

    # ── ELBO trajectory  [T] ───────────────────────────────────────────────
    timestep_grid: torch.Tensor       # t ∈ [0,1], shape [T]
    elbo_per_t:    torch.Tensor       # L(t),       shape [T]

    # ── Loss curve shape  [T] ──────────────────────────────────────────────
    dldt:   torch.Tensor              # dL/dt   (central finite diff) [T]
    d2ldt2: torch.Tensor              # d²L/dt²                       [T]

    # ── ELBO variance (scalar) ─────────────────────────────────────────────
    elbo_variance: float              # Var_t[L(t)]

    # ── Per-token prediction entropy  [T] and [T, L] ──────────────────────
    pred_entropy_per_t: torch.Tensor  # mean over masked positions  [T]
    pred_entropy_full:  torch.Tensor  # full matrix                 [T, L]

    # ── Multi-mask consistency  [T] ────────────────────────────────────────
    mask_consistency_per_t: torch.Tensor  # mode-agreement fraction  [T]

    # ── Attention maps  [T, Nl, L, L] or None ─────────────────────────────
    attention_maps: Optional[torch.Tensor]

    # ── Hidden state geometry  [T, Nl] ────────────────────────────────────
    hidden_norms:      torch.Tensor   # mean ‖h_l‖₂ per layer   [T, Nl]
    hidden_cosine_sim: torch.Tensor   # cos-sim to t=0 baseline  [T, Nl]

    # ── Gradient norms  [T_grad] ───────────────────────────────────────────
    grad_t_grid: torch.Tensor         # subset of timestep_grid  [T_grad]
    grad_norms:  torch.Tensor         # ‖∇_θ L‖₂                [T_grad]

    # ── Timing ─────────────────────────────────────────────────────────────
    elapsed_seconds: float


def print_summary(bundle: MetricsBundle) -> None:
    """Pretty-print all extracted metrics to stdout."""
    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  MDLM Metrics Summary")
    print(f"  Text  : {bundle.text[:70]!r}")
    print(f"  Tokens: {bundle.seq_len}  |  Elapsed: {bundle.elapsed_seconds:.1f}s")
    print(sep)

    print("\n── ELBO trajectory L(t) ──────────────────────────────────────────")
    print(f"  Variance : {bundle.elbo_variance:.6f}")
    for t, l in zip(bundle.timestep_grid.tolist(), bundle.elbo_per_t.tolist()):
        print(f"  t={t:.3f}  L(t)={l:.4f}")

    print("\n── Loss curve shape  dL/dt ───────────────────────────────────────")
    for t, d1, d2 in zip(
        bundle.timestep_grid.tolist(), bundle.dldt.tolist(), bundle.d2ldt2.tolist()
    ):
        print(f"  t={t:.3f}  dL/dt={d1:+.4f}  d²L/dt²={d2:+.4f}")

    print("\n── Prediction entropy H(p_θ(·|z_t)) ─────────────────────────────")
    for t, h in zip(bundle.timestep_grid.tolist(), bundle.pred_entropy_per_t.tolist()):
        print(f"  t={t:.3f}  H={h:.4f}")

    print("\n── Multi-mask consistency ────────────────────────────────────────")
    for t, c in zip(bundle.timestep_grid.tolist(), bundle.mask_consistency_per_t.tolist()):
        print(f"  t={t:.3f}  agreement={c:.4f}")

    print("\n── Hidden state ‖h‖₂ (mean over tokens) ─────────────────────────")
    T, Nl = bundle.hidden_norms.shape
    for t_idx in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        t_val  = bundle.timestep_grid[t_idx].item()
        norms  = bundle.hidden_norms[t_idx].tolist()
        norms_str = "  ".join(f"L{l}:{n:.2f}" for l, n in enumerate(norms[:6]))
        print(f"  t={t_val:.3f}  {norms_str} ...")

    print("\n── Hidden cosine similarity to t=0 ──────────────────────────────")
    for t_idx in [0, T // 2, T - 1]:
        t_val   = bundle.timestep_grid[t_idx].item()
        cos     = bundle.hidden_cosine_sim[t_idx].tolist()
        cos_str = "  ".join(f"L{l}:{c:.3f}" for l, c in enumerate(cos[:6]))
        print(f"  t={t_val:.3f}  {cos_str} ...")

    print("\n── Gradient norms ‖∇_θ L‖₂ ──────────────────────────────────────")
    for t, g in zip(bundle.grad_t_grid.tolist(), bundle.grad_norms.tolist()):
        print(f"  t={t:.3f}  ‖∇_θ L‖={g:.4f}")

    if bundle.attention_maps is not None:
        T_a, Nl_a, _, _ = bundle.attention_maps.shape
        print(f"\n── Attention maps  [T={T_a}, Nl={Nl_a}, L, L] ──────────────")
        print("  (use --save to serialise for full inspection)")
    else:
        print("\n── Attention maps: not captured ─────────────────────────────")

    print(f"\n{sep}\n")
