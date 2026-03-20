"""
metrics/forward.py
==================
Instrumented forward pass — exposes hidden states and attention weights even
for custom trust_remote_code models that ignore output_hidden_states /
output_attentions kwargs.

Strategy:
  1. Patch model.config flags so cooperative models emit the outputs.
  2. Try a standard forward with kwargs.
  3. If the model still returns None for either output, fall back to
     registered forward hooks:
     - Hidden states: generic hook on all modules accepting rank-3 float tensors.
     - Attention: targeted hook on model.layers.<N>.self_attn (Qwen3 path).
       Qwen3Attention.forward returns (context [1,L,d], attn_weights [1,Nh,L,L]).
       We capture output[1].

No project-level imports other than shared.logger.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM

from shared.logger import build_logger

_log = build_logger("metrics.forward")


def patch_model_for_outputs(model: AutoModelForMaskedLM) -> None:
    """
    Mutate model.config (and sub-model configs) to enable hidden-state and
    attention output flags.  Downgrades SDPA → eager so attention weights
    are emitted.  Idempotent.
    """
    def _patch_cfg(cfg) -> None:
        if getattr(cfg, "_attn_implementation", None) == "sdpa":
            cfg._attn_implementation = "eager"
        for attr in ("output_hidden_states", "output_attentions"):
            if not getattr(cfg, attr, False):
                setattr(cfg, attr, True)

    cfg = getattr(model, "config", None)
    if cfg is not None:
        _patch_cfg(cfg)

    for child_name in ("model", "transformer", "bert", "roberta", "encoder"):
        child = getattr(model, child_name, None)
        if child is not None and hasattr(child, "config"):
            _patch_cfg(child.config)


def forward_with_hooks(
    model:              AutoModelForMaskedLM,
    input_ids:          torch.Tensor,      # [1, L]
    capture_attentions: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    Run one forward pass and return:
        logits        : [1, L, V]
        hidden_states : list of Nl tensors, each [1, L, d_model]
        attentions    : list of Nl tensors, each [1, Nh, L, L], or []

    Falls back to forward hooks when the model ignores output kwargs.
    """
    L = input_ids.shape[1]

    patch_model_for_outputs(model)

    _log.debug("  forward: input_ids shape=%s", tuple(input_ids.shape))
    out = model(input_ids, output_hidden_states=True, output_attentions=capture_attentions)
    logits = out.logits

    needs_hook_hidden = out.hidden_states is None
    needs_hook_attn   = capture_attentions and (out.attentions is None)
    _log.debug("  hidden_states present=%s  attentions present=%s",
               not needs_hook_hidden, not needs_hook_attn)

    if not (needs_hook_hidden or needs_hook_attn):
        hidden_states = list(out.hidden_states)
        attentions    = list(out.attentions) if (capture_attentions and out.attentions) else []
        _log.debug("  hook fallback not needed")
        return logits, hidden_states, attentions

    # ── Hook-based fallback ───────────────────────────────────────────────────
    _log.debug("  using hook fallback  (hidden=%s  attn=%s)",
               needs_hook_hidden, needs_hook_attn)

    captured_hidden: list[torch.Tensor] = []
    layer_attn_map:  dict[int, torch.Tensor] = {}
    hooks: list = []

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

    if needs_hook_attn:
        def _make_attn_hook(layer_idx: int):
            def _attn_hook(module, inp, output):
                if not (isinstance(output, tuple) and len(output) >= 2):
                    return
                weights = output[1]
                if weights is None:
                    return
                if not (
                    isinstance(weights, torch.Tensor)
                    and weights.is_floating_point()
                    and weights.ndim == 4
                    and weights.shape[0] == input_ids.shape[0]
                    and weights.shape[2] == L
                    and weights.shape[3] == L
                ):
                    return
                layer_attn_map[layer_idx] = weights.detach()
                _log.debug("    layer %d attn: shape=%s", layer_idx, tuple(weights.shape))
            return _attn_hook

        n_hooks = 0
        for name, module in model.named_modules():
            parts = name.split(".")
            if (
                len(parts) == 4
                and parts[0] == "model"
                and parts[1] == "layers"
                and parts[2].isdigit()
                and parts[3] == "self_attn"
            ):
                hooks.append(module.register_forward_hook(_make_attn_hook(int(parts[2]))))
                n_hooks += 1
        _log.debug("  registered %d self_attn hooks", n_hooks)

    model(input_ids)
    for h in hooks:
        h.remove()

    # De-duplicate hidden states (hook fires on sub-modules too)
    def _dedup(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        seen, out_list = set(), []
        for t in tensors:
            key = (t.data_ptr(), t.shape)
            if key not in seen:
                seen.add(key)
                out_list.append(t)
        return out_list

    hidden_states = _dedup(captured_hidden) if needs_hook_hidden else list(out.hidden_states)
    _log.debug("  hidden: %d raw → %d unique", len(captured_hidden), len(hidden_states))

    if needs_hook_attn:
        if layer_attn_map:
            attentions = [layer_attn_map[i] for i in sorted(layer_attn_map)]
        else:
            attentions = []
            _log.warning("  attention hooks captured nothing — "
                         "ensure model uses attn_implementation='eager'")
    else:
        attentions = list(out.attentions) if out.attentions is not None else []

    return logits, hidden_states, attentions
