"""
mdlm_metrics_extractor.py
=========================
Production-grade metrics extraction harness for MDLM / dLLM models.

Extracts every signal from the MIA signal taxonomy (see Table in the SAMA
paper + the local signal-catalogue PDF), organised by how the signal is
computed and what it measures:

  Signal category        | What we capture
  -----------------------|--------------------------------------------------
  ELBO loss trajectory   | Per-timestep L(t) across t ∈ [0,1]
  Loss curve shape       | dL/dt, d²L/dt² (finite differences over t grid)
  ELBO variance          | Var_t[L(t)] — consistency of reconstruction
  Score / logit entropy  | H(p_θ(·|z_t)) per masked token at each t
  Prediction entropy     | Same as above; MDLM/LLaDA specific name
  Multi-mask consistency | Var of predictions across K mask configs at same t
  Attention patterns     | Layer-wise attention maps at selected mask ratios
  Hidden states          | Per-layer hidden-state norms + cosine geometry
  Gradient norms         | ‖∇_θ L(x₀, t)‖ at selected t values

All signals are captured inside a single instrumented forward pass (or a
small number of forward passes), returning a typed MetricsBundle dataclass
that can be serialised to disk or consumed downstream by an attack classifier.

Usage
-----
    python mdlm_metrics_extractor.py
    python mdlm_metrics_extractor.py --text "The patient was diagnosed with..." \\
                                      --timesteps 20 --mask_configs 8 --save metrics.pt

Architecture note
-----------------
The script does NOT modify the generation loop in mdlm_qwen3_test.py.
Instead, it re-uses load_model_and_tokenizer() and prepare_batch() from that
module, adding a thin instrumentation layer on top.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ── Import helpers from the existing test harness ─────────────────────────────
# If running from the same directory as mdlm_qwen3_test.py, this just works.
# If the module is elsewhere, add its parent to sys.path before this import.
try:
    from mdlm_qwen3_test import (
        load_model_and_tokenizer,
        prepare_batch,
        select_device,
        model_dtype,
    )
except ModuleNotFoundError:
    # Graceful fallback: inline minimal versions so the file is self-contained.
    import os, importlib.util  # noqa: E401
    _here = Path(__file__).parent
    _spec = importlib.util.spec_from_file_location(
        "mdlm_qwen3_test", _here / "mdlm_qwen3_test.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    load_model_and_tokenizer = _mod.load_model_and_tokenizer
    prepare_batch            = _mod.prepare_batch
    select_device            = _mod.select_device
    model_dtype              = _mod.model_dtype


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
class _ColouredFormatter(logging.Formatter):
    """Adds ANSI colour codes by level so the log is easy to scan at a glance."""
    _COLOURS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname:<8}{self._RESET}"
        return super().format(record)


def _build_logger(level: int = logging.INFO) -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ColouredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger = logging.getLogger("mdlm_extractor")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# Initialised at INFO; main() promotes to DEBUG when --debug is passed.
log = _build_logger(logging.INFO)


def _dbg_tensor(label: str, t: Optional[torch.Tensor]) -> None:
    """Log a single tensor's shape / stats at DEBUG level — zero cost when disabled."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    if t is None:
        log.debug("    %-30s  None", label)
    else:
        mn, mx, mu = t.min().item(), t.max().item(), t.float().mean().item()
        log.debug("    %-30s  shape=%-20s  min=%+.4f  max=%+.4f  mean=%+.4f",
                  label, str(tuple(t.shape)), mn, mx, mu)


def _dbg_list(label: str, tensors: list[torch.Tensor]) -> None:
    """Log summary of a list of tensors at DEBUG level."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    if not tensors:
        log.debug("    %-30s  [] (empty)", label)
        return
    shapes = [tuple(t.shape) for t in tensors]
    log.debug("    %-30s  len=%d  shapes[0]=%s … shapes[-1]=%s",
              label, len(tensors), shapes[0], shapes[-1])


# ──────────────────────────────────────────────────────────────────────────────
# Typed result bundle
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MetricsBundle:
    """
    Container for all extracted signals for a single input sequence.

    Shapes use the following abbreviations:
        T   = number of timestep samples (--timesteps)
        L   = sequence length (prompt tokens only, no padding)
        K   = number of independent mask configurations (--mask_configs)
        Nl  = number of transformer layers
        Nh  = number of attention heads
    """

    # ── Input metadata ─────────────────────────────────────────────────────
    text:          str                        # raw input string
    token_ids:     list[int]                  # tokenised input (no padding)
    seq_len:       int                        # L

    # ── ELBO loss trajectory  [T] ──────────────────────────────────────────
    timestep_grid: torch.Tensor               # t values in [0,1], shape [T]
    elbo_per_t:    torch.Tensor               # L(t) at each t,     shape [T]

    # ── Loss curve shape  [T] ─────────────────────────────────────────────
    dldt:          torch.Tensor               # dL/dt  (central finite diff) [T]
    d2ldt2:        torch.Tensor               # d²L/dt² [T]

    # ── ELBO variance (scalar) ─────────────────────────────────────────────
    elbo_variance: float                      # Var_t[L(t)]

    # ── Per-token prediction entropy  [T, L] ──────────────────────────────
    # H(p_θ(·|z_t)) averaged over masked tokens at each timestep
    pred_entropy_per_t: torch.Tensor          # shape [T]
    # Full per-token, per-timestep matrix (for detailed analysis)
    pred_entropy_full:  torch.Tensor          # shape [T, L]

    # ── Multi-mask consistency  [T] ───────────────────────────────────────
    # Variance of argmax predictions across K mask configs at each t
    mask_consistency_per_t: torch.Tensor      # shape [T]

    # ── Attention patterns  [T, Nl, L, L] ────────────────────────────────
    # Averaged over heads; one snapshot per timestep
    attention_maps: Optional[torch.Tensor]    # shape [T, Nl, L, L] or None

    # ── Hidden state geometry  [T, Nl] ────────────────────────────────────
    hidden_norms:      torch.Tensor           # ‖h_l‖₂ per layer, shape [T, Nl]
    hidden_cosine_sim: torch.Tensor           # cos-sim to t=0 hidden, [T, Nl]

    # ── Gradient norms  [T_grad] ──────────────────────────────────────────
    # Computed only at a small subset of timesteps (expensive)
    grad_t_grid:   torch.Tensor               # subset of timestep_grid
    grad_norms:    torch.Tensor               # ‖∇_θ L‖₂ per param group [T_grad]

    # ── Timing ────────────────────────────────────────────────────────────
    elapsed_seconds: float


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_masking(
    token_ids: torch.Tensor,        # [1, L]
    mask_id:   int,
    mask_ratio: float,              # fraction of tokens to mask ∈ (0,1)
    rng:       Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Randomly mask a `mask_ratio` fraction of positions in token_ids.
    Returns a new [1, L] tensor with selected positions replaced by mask_id.
    """
    L = token_ids.size(1)
    n_mask = max(1, int(round(mask_ratio * L)))
    # Sample without replacement
    perm = torch.randperm(L, generator=rng, device=token_ids.device)
    mask_positions = perm[:n_mask]
    z_t = token_ids.clone()
    z_t[0, mask_positions] = mask_id
    return z_t


def _patch_model_config_for_outputs(model: AutoModelForMaskedLM) -> None:
    """
    Some custom trust_remote_code models gate hidden-state and attention
    output on config flags rather than on the forward-pass kwargs.
    Patching config before the first call ensures they are always emitted.
    This is a no-op on well-behaved models.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    # SDPA does not support output_attentions; downgrade to eager attention
    if getattr(cfg, '_attn_implementation', None) == 'sdpa':
        cfg._attn_implementation = 'eager'
    for attr in ("output_hidden_states", "output_attentions"):
        if not getattr(cfg, attr, False):
            setattr(cfg, attr, True)
    # Some architectures nest the real transformer under .model or .bert etc.
    for child_name in ("model", "transformer", "bert", "roberta", "encoder"):
        child = getattr(model, child_name, None)
        if child is not None and hasattr(child, "config"):
            if getattr(child.config, '_attn_implementation', None) == 'sdpa':
                child.config._attn_implementation = 'eager'
            for attr in ("output_hidden_states", "output_attentions"):
                if not getattr(child.config, attr, False):
                    setattr(child.config, attr, True)


def _forward_with_hooks(
    model:      AutoModelForMaskedLM,
    input_ids:  torch.Tensor,          # [1, L]
    capture_attentions: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    Run one forward pass and return:
        logits        : [1, L, V]
        hidden_states : list of Nl tensors, each [1, L, d_model]
        attentions    : list of Nl tensors, each [1, Nh, L, L], or []

    Strategy
    --------
    1. Patch model.config so any models that honour the output flags do so.
    2. Call model() with output_hidden_states / output_attentions kwarg.
    3. If the model still returns None (the A2DQwen3 custom model ignores
       these kwargs entirely), fall back to two separate forward-hook passes:

       Hidden states — generic hook on all modules; accepts rank-3 float
       tensors matching [batch, L, d_model].

       Attention weights — TARGETED hook placed only on modules that match
       the path pattern  model.layers.<N>.self_attn  (top-level attention
       module, not its children).  Qwen3Attention returns a 2-tuple:
           output[0] : context vector   [1, L, d_model]   (rank-3)
           output[1] : attention weights [1, Nh, L, L]    (rank-4)  ← this
       Hooks are keyed by layer index so the result list is always ordered
       layer-0 … layer-Nl.
    """
    L = input_ids.shape[1]

    # ── Step 1: config patch ─────────────────────────────────────────────────
    _patch_model_config_for_outputs(model)

    # ── Step 2: standard forward ─────────────────────────────────────────────
    log.debug("  forward pass: input_ids shape=%s", tuple(input_ids.shape))
    out = model(
        input_ids,
        output_hidden_states=True,
        output_attentions=capture_attentions,
    )
    logits = out.logits  # [1, L, V]
    _dbg_tensor("logits", logits)

    needs_hook_hidden = out.hidden_states is None
    needs_hook_attn   = capture_attentions and (out.attentions is None)

    log.debug("  out.hidden_states present: %s | out.attentions present: %s",
              not needs_hook_hidden, not needs_hook_attn)

    # ── Step 3: hook-based fallback ───────────────────────────────────────────
    if needs_hook_hidden or needs_hook_attn:
        log.debug("  falling back to forward hooks  (hidden=%s  attn=%s)",
                  needs_hook_hidden, needs_hook_attn)

        captured_hidden: list[torch.Tensor] = []
        # layer_idx → attn_weights [1, Nh, L, L]
        layer_attn_map:  dict[int, torch.Tensor] = {}
        hooks: list = []

        # ── Hook A: generic hidden-state capture ─────────────────────────────
        if needs_hook_hidden:
            def _hidden_hook(module, inp, output):
                t = output[0] if isinstance(output, tuple) else output
                if (
                    isinstance(t, torch.Tensor)
                    and t.is_floating_point()
                    and t.ndim == 3
                    and t.shape[0] == input_ids.shape[0]
                    and t.shape[1] == L
                ):
                    captured_hidden.append(t.detach())
            for module in model.modules():
                hooks.append(module.register_forward_hook(_hidden_hook))

        # ── Hook B: targeted self_attn attention-weight capture ───────────────
        # Qwen3Attention.forward() returns (context [1,L,d], attn_weights [1,Nh,L,L])
        # We must read output[1], NOT output[0].
        if needs_hook_attn:
            def _make_attn_hook(layer_idx: int):
                def _attn_hook(module, inp, output):
                    if not (isinstance(output, tuple) and len(output) >= 2):
                        log.debug("    layer %d self_attn: output is not a 2-tuple (%s)",
                                  layer_idx, type(output))
                        return
                    weights = output[1]
                    if weights is None:
                        log.debug("    layer %d self_attn: output[1] is None", layer_idx)
                        return
                    if not (
                        isinstance(weights, torch.Tensor)
                        and weights.is_floating_point()
                        and weights.ndim == 4
                        and weights.shape[0] == input_ids.shape[0]
                        and weights.shape[2] == L
                        and weights.shape[3] == L
                    ):
                        log.debug("    layer %d self_attn: output[1] shape %s does not match "
                                  "[%d, *, %d, %d]", layer_idx,
                                  tuple(weights.shape) if isinstance(weights, torch.Tensor) else "N/A",
                                  input_ids.shape[0], L, L)
                        return
                    layer_attn_map[layer_idx] = weights.detach()
                    log.debug("    layer %d attn captured: shape=%s  heads=%d",
                              layer_idx, tuple(weights.shape), weights.shape[1])
                return _attn_hook

            n_attn_hooks = 0
            for name, module in model.named_modules():
                parts = name.split(".")
                # Match exactly: model.layers.<N>.self_attn  (4 components)
                if (
                    len(parts) == 4
                    and parts[0] == "model"
                    and parts[1] == "layers"
                    and parts[2].isdigit()
                    and parts[3] == "self_attn"
                ):
                    idx = int(parts[2])
                    hooks.append(module.register_forward_hook(_make_attn_hook(idx)))
                    n_attn_hooks += 1
            log.debug("  registered %d targeted self_attn hooks", n_attn_hooks)

        # ── Run the second forward pass with all hooks active ─────────────────
        model(input_ids)
        for h in hooks:
            h.remove()

        # ── De-duplicate hidden states (hook fires on sub-modules too) ─────────
        def _dedup(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
            seen, out_list = set(), []
            for t in tensors:
                key = (t.data_ptr(), t.shape)
                if key not in seen:
                    seen.add(key)
                    out_list.append(t)
            return out_list

        if needs_hook_hidden:
            hidden_states = _dedup(captured_hidden)
            log.debug("  hidden hook: %d raw → %d unique tensors",
                      len(captured_hidden), len(hidden_states))
            _dbg_list("hidden_states", hidden_states)
        else:
            hidden_states = list(out.hidden_states)

        if needs_hook_attn:
            if layer_attn_map:
                attentions = [layer_attn_map[i] for i in sorted(layer_attn_map)]
                log.debug("  attn hook: captured %d/%d layers",
                          len(attentions), max(layer_attn_map) + 1 if layer_attn_map else 0)
                _dbg_list("attentions", attentions)
            else:
                attentions = []
                log.warning("  attention hooks fired but captured nothing — "
                            "check that model was loaded with attn_implementation='eager'")
        else:
            attentions = list(out.attentions) if out.attentions is not None else []
    else:
        hidden_states = list(out.hidden_states)
        attentions = (
            list(out.attentions)
            if capture_attentions and out.attentions is not None
            else []
        )

    log.debug("  _forward_with_hooks done: hidden=%d  attn=%d",
              len(hidden_states), len(attentions))
    return logits, hidden_states, attentions


def _token_entropy(logits: torch.Tensor, masked_positions: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token Shannon entropy H(softmax(logits)) at masked positions.

    Args:
        logits           : [1, L, V]
        masked_positions : bool tensor [L] — True where token was masked

    Returns:
        entropies : [L] — entropy at each position (0 at non-masked positions)
    """
    probs = F.softmax(logits[0], dim=-1)               # [L, V]
    log_probs = F.log_softmax(logits[0], dim=-1)       # [L, V]
    H = -(probs * log_probs).sum(dim=-1)               # [L]
    H = H * masked_positions.float()
    return H


def _layer_norms(hidden_states: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute mean ‖h‖₂ over token dimension for each layer.

    Handles two tensor shapes emitted by different model variants:
        [1, L, d]  — standard HuggingFace output_hidden_states format
        [L, d]     — hook-captured tensors (no batch dimension)

    Returns:
        norms : [Nl]
    """
    norms = []
    for h in hidden_states:
        # Normalise to [L, d] regardless of whether batch dim is present
        h2d = h[0] if h.ndim == 3 else h   # [L, d]
        norms.append(h2d.norm(dim=-1).mean())
    return torch.stack(norms)


def _attention_mean(attentions: list[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Average attention weights over heads for each layer.

    Args:
        attentions : list of Nl tensors, each [1, Nh, L, L]

    Returns:
        avg_attn : [Nl, L, L], or None if list is empty
    """
    if not attentions:
        return None
    return torch.stack([a[0].mean(dim=0) for a in attentions])  # [Nl, L, L]


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction engine
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_metrics(
    model:        AutoModelForMaskedLM,
    tokenizer:    AutoTokenizer,
    text:         str,
    n_timesteps:  int  = 20,
    n_mask_configs: int = 8,
    grad_timesteps: int = 4,
    capture_attentions: bool = True,
    seed:         int  = 42,
) -> MetricsBundle:
    """
    Full metrics extraction for a single input text.

    Procedure
    ---------
    1.  Tokenise the input (no padding; single sequence).
    2.  Define a uniform grid of mask ratios  t ∈ {1/T, 2/T, …, 1}.
        In MDLM the mask ratio IS the continuous time t.
    3.  For each t in the grid:
        a. Sample K independent random mask configurations.
        b. For each configuration run a forward pass capturing logits,
           hidden states, and (optionally) attention weights.
        c. Aggregate across K configurations:
           - mean cross-entropy loss → L(t)
           - mean per-token prediction entropy
           - variance of argmax predictions → multi-mask consistency
           - mean attention maps (per layer)
           - mean hidden state norms
    4.  Compute finite-difference derivatives of L(t) over t.
    5.  Compute gradient norms at a small subset of t values
        (requires grad; handled in a separate with-grad context).
    6.  Return MetricsBundle.

    Args
    ----
    model              : Loaded MDLM model in eval() mode.
    tokenizer          : Corresponding tokenizer.
    text               : Raw input string to analyse.
    n_timesteps        : Number of t values to evaluate (T).
    n_mask_configs     : Independent mask draws per timestep (K).
    grad_timesteps     : How many t values to compute gradients at (expensive).
    capture_attentions : If True, store per-layer attention maps.
    seed               : RNG seed for reproducibility.

    Returns
    -------
    MetricsBundle with all signals populated.
    """
    t_start = time.time()
    device   = next(model.parameters()).device
    mask_id  = tokenizer.mask_token_id
    rng      = torch.Generator(device=device).manual_seed(seed)

    # ── 1. Tokenise ──────────────────────────────────────────────────────────
    token_ids = tokenizer.encode(text, return_tensors="pt").to(device)  # [1, L]
    L = token_ids.size(1)
    log.info("Input: %d tokens  |  text[:60]: %r", L, text[:60])
    log.debug("  token_ids: %s", token_ids[0].tolist())

    # ── 2. Timestep grid ─────────────────────────────────────────────────────
    # t=0 → fully observed; t=1 → fully masked.
    # We sample T points uniformly in (0, 1] (avoid t=0 where no mask exists).
    timestep_grid = torch.linspace(1.0 / n_timesteps, 1.0, n_timesteps)  # [T]

    # ── Accumulators ─────────────────────────────────────────────────────────
    elbo_per_t:             list[float]         = []
    entropy_per_t:          list[float]         = []
    entropy_full_per_t:     list[torch.Tensor]  = []  # each [L]
    consistency_per_t:      list[float]         = []
    attn_per_t:             list[Optional[torch.Tensor]] = []   # each [Nl,L,L]
    hidden_norm_per_t:      list[torch.Tensor]  = []  # each [Nl]

    # Baseline hidden norms at t≈0 (no masking) — used for cosine similarity
    with torch.no_grad():
        _, h0_states, _ = _forward_with_hooks(
            model, token_ids, capture_attentions=False
        )

    hidden_available = len(h0_states) > 0
    if not hidden_available:
        log.warning(
            "Hidden states unavailable from this model — "
            "hidden_norms and hidden_cosine_sim will be zero tensors."
        )
        h0_dirs: list[torch.Tensor] = []
    else:
        h0_dirs = [h[0].mean(dim=0) for h in h0_states]  # list of [d_model]

    cosine_sim_per_t: list[torch.Tensor] = []             # each [Nl]

    # ── 3. Main loop over timesteps ───────────────────────────────────────────
    for t_idx, t in enumerate(timestep_grid.tolist()):
        mask_ratio = t   # in MDLM, t IS the fraction of tokens masked

        # Per-config accumulators
        losses_k:       list[float]         = []
        entropies_k:    list[torch.Tensor]  = []   # each [L]
        predictions_k:  list[torch.Tensor]  = []   # each [L]  (argmax token)
        attn_k:         list[Optional[torch.Tensor]] = []
        hidden_norm_k:  list[torch.Tensor]  = []
        hidden_dir_k:   list[list[torch.Tensor]] = []  # [K, Nl, d_model]

        for k in range(n_mask_configs):
            z_t = _apply_masking(token_ids, mask_id, mask_ratio, rng=rng)
            log.debug("  t=%.3f  k=%d/%d  masked_tokens=%d/%d",
                      t, k + 1, n_mask_configs,
                      (z_t[0] == mask_id).sum().item(), L)

            logits, hidden, attentions = _forward_with_hooks(
                model, z_t, capture_attentions=capture_attentions
            )  # logits: [1, L, V]
            log.debug("    returned: logits=%s  hidden=%d  attn=%d",
                      tuple(logits.shape), len(hidden), len(attentions))

            # ── Cross-entropy loss at masked positions only ───────────────
            masked_pos = (z_t[0] == mask_id)                  # [L]  bool
            if masked_pos.sum() == 0:
                losses_k.append(0.0)
                entropies_k.append(torch.zeros(L, device=device))
                predictions_k.append(token_ids[0].clone())
                attn_k.append(None)
                if hidden:
                    hidden_norm_k.append(_layer_norms(hidden))
                    hidden_dir_k.append([h[0].mean(dim=0) for h in hidden])
                continue

            # CE loss: compare model output against original (unmasked) tokens
            target  = token_ids[0]                            # [L]
            log_p   = F.log_softmax(logits[0], dim=-1)        # [L, V]
            ce_full = F.nll_loss(log_p, target, reduction="none")  # [L]
            # Average over masked positions only
            ce_masked = ce_full[masked_pos].mean().item()
            losses_k.append(ce_masked)

            # ── Per-token prediction entropy ──────────────────────────────
            H = _token_entropy(logits, masked_pos)            # [L]
            entropies_k.append(H)

            # ── Argmax prediction (for consistency metric) ────────────────
            predictions_k.append(logits[0].argmax(dim=-1))   # [L]

            # ── Attention + hidden ────────────────────────────────────────
            attn_k.append(_attention_mean(attentions))        # [Nl,L,L] or None
            if hidden:
                hidden_norm_k.append(_layer_norms(hidden))
                hidden_dir_k.append([h[0].mean(dim=0) for h in hidden])

        # ── Aggregate over K configs ──────────────────────────────────────
        elbo_per_t.append(float(np.mean(losses_k)))

        # Mean entropy over K (shape [L])
        ent_mean = torch.stack(entropies_k).mean(dim=0)       # [L]
        entropy_full_per_t.append(ent_mean)
        entropy_per_t.append(ent_mean[ent_mean > 0].mean().item())

        # Multi-mask consistency: fraction of positions where all K configs
        # agree on the same argmax prediction
        preds_tensor = torch.stack(predictions_k)             # [K, L]
        # Variance of token IDs is not semantically great; use mode agreement:
        mode_pred = preds_tensor.cpu().mode(dim=0).values.to(preds_tensor.device)  # [L] most common
        agree = (preds_tensor == mode_pred.unsqueeze(0)).float().mean(dim=0)  # [L]
        # Consistency score per t: mean agreement across all positions
        consistency_per_t.append(agree.mean().item())

        # Mean attention
        valid_attn = [a for a in attn_k if a is not None]
        if valid_attn:
            attn_per_t.append(torch.stack(valid_attn).mean(dim=0))   # [Nl,L,L]
        else:
            attn_per_t.append(None)

        # Mean hidden norms (guard: hidden_norm_k may be empty if model
        # doesn't expose hidden states at all)
        if hidden_norm_k and all(h.numel() > 0 for h in hidden_norm_k):
            hidden_norm_per_t.append(
                torch.stack(hidden_norm_k).mean(dim=0)       # [Nl]
            )
        else:
            hidden_norm_per_t.append(torch.zeros(1, device=device))

        # Cosine similarity to t=0 hidden directions (only if available)
        if hidden_available and hidden_dir_k and h0_dirs:
            cos_sims = []
            Nl = len(h0_dirs)
            mean_dirs_k = [
                torch.stack([
                    hidden_dir_k[k][l]
                    for k in range(len(hidden_dir_k))
                    if l < len(hidden_dir_k[k])
                ]).mean(dim=0)
                for l in range(Nl)
            ]
            for h0, ht in zip(h0_dirs, mean_dirs_k):
                cos = F.cosine_similarity(h0.unsqueeze(0), ht.unsqueeze(0)).item()
                cos_sims.append(cos)
            cosine_sim_per_t.append(torch.tensor(cos_sims, device=device))  # [Nl]
        else:
            cosine_sim_per_t.append(torch.zeros(1, device=device))

        if (t_idx + 1) % 5 == 0 or t_idx == 0:
            log.info(
                "  t=%.3f | L(t)=%.4f | H_mean=%.4f | consistency=%.4f",
                t, elbo_per_t[-1], entropy_per_t[-1], consistency_per_t[-1],
            )

    # ── 4. Convert accumulators to tensors ───────────────────────────────────
    elbo_tensor       = torch.tensor(elbo_per_t,        dtype=torch.float32)
    entropy_tensor    = torch.tensor(entropy_per_t,     dtype=torch.float32)
    entropy_full_mat  = torch.stack(entropy_full_per_t)          # [T, L]
    consistency_tensor = torch.tensor(consistency_per_t, dtype=torch.float32)
    hidden_norm_mat   = torch.stack(hidden_norm_per_t)           # [T, Nl]
    cosine_sim_mat    = torch.stack(cosine_sim_per_t)            # [T, Nl]

    # Attention: [T, Nl, L, L] or None
    if all(a is not None for a in attn_per_t):
        attn_tensor = torch.stack(attn_per_t)                    # [T, Nl, L, L]
    else:
        attn_tensor = None

    # ── 5. Finite-difference derivatives of L(t) over t ─────────────────────
    t_np     = timestep_grid.numpy()
    elbo_np  = elbo_tensor.numpy()
    dldt_np  = np.gradient(elbo_np, t_np)                       # central diffs
    d2ldt2_np = np.gradient(dldt_np, t_np)
    dldt    = torch.from_numpy(dldt_np.copy()).float()
    d2ldt2  = torch.from_numpy(d2ldt2_np.copy()).float()

    # ── 6. Gradient norms (requires_grad pass) ───────────────────────────────
    # We sample `grad_timesteps` t values spread across [0,1]
    grad_t_indices = np.linspace(0, n_timesteps - 1, grad_timesteps, dtype=int)
    grad_t_grid    = timestep_grid[grad_t_indices]              # [T_grad]
    grad_norm_list: list[float] = []

    log.info("Computing gradient norms at %d timesteps …", grad_timesteps)
    for t_idx in grad_t_indices:
        t = float(timestep_grid[t_idx].item())
        mask_ratio = t
        z_t = _apply_masking(token_ids, mask_id, mask_ratio, rng=rng)

        # Fresh forward with gradient tracking
        # Use a subset of parameters to keep memory manageable
        model.zero_grad()
        for p in model.parameters():
            p.requires_grad_(True)

        with torch.enable_grad():
            logits_g, _, _ = _forward_with_hooks(
                model, z_t, capture_attentions=False
            )

            masked_pos_g = (z_t[0] == mask_id)
            if masked_pos_g.sum() == 0:
                grad_norm_list.append(0.0)
                model.zero_grad()
                for p in model.parameters():
                    p.requires_grad_(False)
                continue

            target_g = token_ids[0]
            log_p_g  = F.log_softmax(logits_g[0], dim=-1)
            loss_g   = F.nll_loss(log_p_g, target_g, reduction="none")[masked_pos_g].mean()
            loss_g.backward()

        total_grad_norm = torch.sqrt(
            sum(
                p.grad.norm() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
        ).item()
        grad_norm_list.append(total_grad_norm)

        model.zero_grad()
        for p in model.parameters():
            p.requires_grad_(False)

        log.info("  t=%.3f | ‖∇_θ L‖ = %.4f", t, total_grad_norm)

    grad_norms = torch.tensor(grad_norm_list, dtype=torch.float32)

    elapsed = time.time() - t_start
    log.info("Metrics extraction complete in %.1fs", elapsed)

    return MetricsBundle(
        text                  = text,
        token_ids             = token_ids[0].tolist(),
        seq_len               = L,
        timestep_grid         = timestep_grid,
        elbo_per_t            = elbo_tensor,
        dldt                  = dldt,
        d2ldt2                = d2ldt2,
        elbo_variance         = float(elbo_tensor.var().item()),
        pred_entropy_per_t    = entropy_tensor,
        pred_entropy_full     = entropy_full_mat,
        mask_consistency_per_t = consistency_tensor,
        attention_maps        = attn_tensor,
        hidden_norms          = hidden_norm_mat,
        hidden_cosine_sim     = cosine_sim_mat,
        grad_t_grid           = grad_t_grid,
        grad_norms            = grad_norms,
        elapsed_seconds       = elapsed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(bundle: MetricsBundle) -> None:
    """
    Pretty-print a human-readable summary of all extracted metrics.
    """
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
        bundle.timestep_grid.tolist(),
        bundle.dldt.tolist(),
        bundle.d2ldt2.tolist(),
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
    # Print a few representative t values
    for t_idx in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        t_val = bundle.timestep_grid[t_idx].item()
        norms = bundle.hidden_norms[t_idx].tolist()
        norms_str = "  ".join(f"L{l}:{n:.2f}" for l, n in enumerate(norms[:6]))
        print(f"  t={t_val:.3f}  {norms_str} ...")

    print("\n── Hidden cosine similarity to t=0 ──────────────────────────────")
    for t_idx in [0, T // 2, T - 1]:
        t_val = bundle.timestep_grid[t_idx].item()
        cos   = bundle.hidden_cosine_sim[t_idx].tolist()
        cos_str = "  ".join(f"L{l}:{c:.3f}" for l, c in enumerate(cos[:6]))
        print(f"  t={t_val:.3f}  {cos_str} ...")

    print("\n── Gradient norms ‖∇_θ L‖₂ ──────────────────────────────────────")
    for t, g in zip(bundle.grad_t_grid.tolist(), bundle.grad_norms.tolist()):
        print(f"  t={t:.3f}  ‖∇_θ L‖={g:.4f}")

    if bundle.attention_maps is not None:
        T_a, Nl_a, _, _ = bundle.attention_maps.shape
        print(f"\n── Attention maps  shape=[T={T_a}, Nl={Nl_a}, L, L] ─────────")
        print("  (stored; use --save to serialise for full inspection)")
    else:
        print("\n── Attention maps: not captured (--no_attentions) ──────────")

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract MDLM membership-inference signals from a text input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        help="HuggingFace model identifier.",
    )
    p.add_argument(
        "--text",
        default="The patient was diagnosed with a rare form of lymphoma and "
                "prescribed an experimental immunotherapy protocol.",
        help="Input text to analyse.",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=20,
        help="Number of t values to evaluate on the [0,1] grid (T).",
    )
    p.add_argument(
        "--mask_configs",
        type=int,
        default=8,
        help="Independent mask configurations per timestep (K).",
    )
    p.add_argument(
        "--grad_timesteps",
        type=int,
        default=4,
        help="Timesteps at which to compute gradient norms (expensive).",
    )
    p.add_argument(
        "--no_attentions",
        action="store_true",
        help="Skip attention map capture (faster, less memory).",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="If provided, serialise the MetricsBundle to this path via torch.save.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible masking.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging: prints tensor shapes, hook counts, "
             "per-step internals, and attention capture diagnostics.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Promote logger to DEBUG if requested ──────────────────────────────────
    if args.debug:
        log.setLevel(logging.DEBUG)
        log.debug("DEBUG logging enabled — verbose tensor/hook diagnostics active")

    device = select_device()

    log.info("Loading model : %s", args.model)
    log.info("Device        : %s", device)
    log.info("Timesteps     : %d  |  mask_configs: %d  |  grad_timesteps: %d",
             args.timesteps, args.mask_configs, args.grad_timesteps)
    log.info("Capture attn  : %s", not args.no_attentions)

    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # SDPA attention modules never expose output[1] as attention weights.
    # Reload with eager attention so Qwen3Attention (not Qwen3SdpaAttention)
    # is instantiated — it returns (context, attn_weights) tuples that our
    # targeted self_attn hook captures from output[1].
    if not args.no_attentions:
        log.info("Reloading with attn_implementation='eager' so attention weights "
                 "are accessible via forward hooks (output[1] of each self_attn) …")
        from transformers import AutoModelForMaskedLM as _AMML
        dtype = next(model.parameters()).dtype
        model = _AMML.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device).eval()
        log.debug("Model class after reload: %s", type(model).__name__)
        # Verify at least one self_attn module is Qwen3Attention (not Sdpa)
        attn_types = {type(m).__name__ for n, m in model.named_modules() if "self_attn" in n and "." not in n.split("self_attn")[-1].lstrip(".")}
        log.debug("self_attn module types found: %s", attn_types)

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

    if args.save:
        save_path = Path(args.save)
        torch.save(bundle, save_path)
        log.info("MetricsBundle saved to %s", save_path)


if __name__ == "__main__":
    main()