"""
mdlm_shared.py
==============
Shared utilities for the MDLM project scripts.

Provides:
  - Device/dtype helpers        (select_device, model_dtype)
  - Model + tokenizer loading   (load_model_and_tokenizer)
  - Prompt batch preparation    (prepare_batch)
  - Unified coloured logger     (build_logger)
  - Persistent run logger       (RunLogger)

Usage
-----
    from mdlm_shared import (
        select_device, model_dtype,
        load_model_and_tokenizer, prepare_batch,
        build_logger, RunLogger,
    )
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

class _ColouredFormatter(logging.Formatter):
    """ANSI colour codes by log level for easy visual scanning."""
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


def build_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a coloured stdout logger with the given name.

    Calling build_logger with the same name twice returns the same logger
    (idempotent); useful when both scripts are imported in the same process.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured — don't double-add handlers
        logger.setLevel(level)
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ColouredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Device / dtype
# ──────────────────────────────────────────────────────────────────────────────

_log = build_logger("mdlm_shared")


def select_device() -> torch.device:
    """
    Prefer CUDA > MPS > CPU.
    MPS does not support bfloat16, so float32 is used there.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        _log.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        _log.info("Device: MPS (Apple Silicon) — using float32 (bf16 unsupported)")
    else:
        dev = torch.device("cpu")
        _log.info("Device: CPU")
    return dev


def model_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 on CUDA; float32 everywhere else."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32


# ──────────────────────────────────────────────────────────────────────────────
# Model + tokenizer loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    attn_implementation: Optional[str] = None,
) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """
    Download (or load from HF cache) the MDLM model and tokenizer.

    Args:
        model_name          : HuggingFace model identifier.
        device              : Target torch.device.
        attn_implementation : Optional override (e.g. "eager" to expose
                              attention weights via forward hooks).

    Notes
    -----
    - trust_remote_code=True is required for custom modelling classes.
    - .eval() disables dropout; callers should use @torch.no_grad() or
      torch.enable_grad() as appropriate.
    """
    dtype = model_dtype(device)
    _log.info("Loading tokenizer from %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        _log.info("  pad_token set to eos_token (%r)", tokenizer.eos_token)

    kwargs: dict = dict(torch_dtype=dtype, trust_remote_code=True)
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    _log.info("Loading model in %s precision%s …", dtype,
              f" (attn={attn_implementation})" if attn_implementation else "")
    t0 = time.time()
    model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs).to(device).eval()
    _log.info(
        "  Model loaded in %.1fs  |  params: %.0fM",
        time.time() - t0,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Prompt batch preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_batch(
    messages_list: list[list[dict]],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Apply the chat template to each conversation and left-pad into a batch.

    enable_thinking=False is recommended for stable, reproducible outputs;
    thinking mode is an autoregressive feature that doesn't integrate cleanly
    with the diffusion generation loop.

    Returns
    -------
    prompt_tensor : LongTensor [batch, max_prompt_len] — right-aligned prompts.
    prompt_lens   : LongTensor [batch]                 — actual length of each.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    token_ids_list = [
        tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        for msgs in messages_list
    ]

    prompt_lens = torch.tensor([len(ids) for ids in token_ids_list], dtype=torch.long)
    max_len = int(prompt_lens.max().item())

    prompt_tensor = torch.full((len(token_ids_list), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(token_ids_list):
        prompt_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    return prompt_tensor.to(device), prompt_lens.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Persistent run logger
# ──────────────────────────────────────────────────────────────────────────────

class RunLogger:
    """
    Appends one JSON record per run to ``<log_dir>/runs.jsonl``.

    Each record is a self-contained JSON object (one per line) so the file
    can be loaded for analysis with::

        import pandas as pd
        df = pd.read_json("logs/runs.jsonl", lines=True)

    or plain Python::

        import json
        runs = [json.loads(l) for l in open("logs/runs.jsonl")]
    """

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self._path = Path(log_dir) / "runs.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── internal ──────────────────────────────────────────────────────────────

    def _write(self, record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _log.info("Run logged → %s", self._path)

    # ── public API ────────────────────────────────────────────────────────────

    def log_generation(
        self,
        *,
        model_name: str,
        config: object,          # GenerationConfig dataclass
        elapsed: float,
        tokens_generated: int,
    ) -> None:
        """
        Log one generation run.

        Args:
            model_name       : HuggingFace model identifier used.
            config           : GenerationConfig dataclass instance.
            elapsed          : Wall-clock seconds for the generation call.
            tokens_generated : Total tokens produced (batch_size * max_new_tokens).
        """
        cfg_dict = (
            {k: v for k, v in vars(config).items()}
            if hasattr(config, "__dict__")
            else {}
        )
        record = {
            "script":           "mdlm_qwen3_test",
            "model":            model_name,
            "config":           cfg_dict,
            "elapsed_s":        round(elapsed, 3),
            "tokens_generated": tokens_generated,
            "tok_per_s":        round(tokens_generated / elapsed, 2) if elapsed > 0 else None,
        }
        self._write(record)

    def log_extraction(
        self,
        *,
        model_name: str,
        text: str,
        n_timesteps: int,
        n_mask_configs: int,
        grad_timesteps: int,
        bundle,                  # MetricsBundle dataclass
        save_path: Optional[str] = None,
    ) -> None:
        """
        Log one metrics-extraction run (scalar summary only — the full
        MetricsBundle tensors are saved separately via --save if requested).

        Args:
            model_name      : HuggingFace model identifier used.
            text            : Input text (first 120 chars stored).
            n_timesteps     : T parameter used.
            n_mask_configs  : K parameter used.
            grad_timesteps  : Number of gradient timesteps computed.
            bundle          : MetricsBundle returned by extract_metrics().
            save_path       : Path where the .pt bundle was saved, or None.
        """
        record = {
            "script":          "mdlm_metrics_extractor",
            "model":           model_name,
            "text_preview":    text[:120],
            "seq_len":         bundle.seq_len,
            "config": {
                "n_timesteps":    n_timesteps,
                "n_mask_configs": n_mask_configs,
                "grad_timesteps": grad_timesteps,
            },
            "metrics": {
                "elbo_variance":         round(bundle.elbo_variance, 6),
                "elbo_mean":             round(float(bundle.elbo_per_t.mean()), 6),
                "elbo_min":              round(float(bundle.elbo_per_t.min()),  6),
                "elbo_max":              round(float(bundle.elbo_per_t.max()),  6),
                "entropy_mean":          round(float(bundle.pred_entropy_per_t.mean()), 6),
                "entropy_max":           round(float(bundle.pred_entropy_per_t.max()),  6),
                "consistency_mean":      round(float(bundle.mask_consistency_per_t.mean()), 6),
                "consistency_min":       round(float(bundle.mask_consistency_per_t.min()),  6),
                "grad_norm_mean":        round(float(bundle.grad_norms.mean()), 6),
                "grad_norm_max":         round(float(bundle.grad_norms.max()),  6),
                "dldt_abs_mean":         round(float(bundle.dldt.abs().mean()), 6),
                "d2ldt2_abs_mean":       round(float(bundle.d2ldt2.abs().mean()), 6),
                "hidden_norm_mean":      round(float(bundle.hidden_norms.mean()), 6),
                "hidden_cosine_mean":    round(float(bundle.hidden_cosine_sim.mean()), 6),
                "attention_captured":    bundle.attention_maps is not None,
            },
            "elapsed_s":       round(bundle.elapsed_seconds, 3),
            "bundle_path":     str(save_path) if save_path else None,
        }
        self._write(record)
