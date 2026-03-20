"""
shared/logger.py
================
Unified coloured logger and persistent JSON-lines run logger.
No project-level imports.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Console logger
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
    Return a coloured stdout logger.  Idempotent — calling with the same
    name twice returns the existing logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
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
# Persistent run logger
# ──────────────────────────────────────────────────────────────────────────────

class RunLogger:
    """
    Append-only JSON-lines logger.  Each call to ``write()`` adds one
    JSON object (one line) to ``<log_dir>/runs.jsonl``.

    Load for analysis with:

        import pandas as pd
        df = pd.read_json("logs/runs.jsonl", lines=True)

    or plain Python:

        runs = [json.loads(l) for l in open("logs/runs.jsonl")]
    """

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self._path = Path(log_dir) / "runs.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._log = build_logger("run_logger")

    def write(self, record: dict[str, Any]) -> None:
        """Append one record to the JSONL file, stamping it with a UTC timestamp."""
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log.info("Run logged → %s", self._path)

    def log_generation(
        self,
        *,
        model_name: str,
        config: Any,            # GenerationConfig dataclass
        elapsed: float,
        tokens_generated: int,
    ) -> None:
        """Log a generation run — config, latency, throughput."""
        cfg_dict = vars(config) if hasattr(config, "__dict__") else {}
        self.write({
            "script":           "run_generation",
            "model":            model_name,
            "config":           cfg_dict,
            "elapsed_s":        round(elapsed, 3),
            "tokens_generated": tokens_generated,
            "tok_per_s":        round(tokens_generated / elapsed, 2) if elapsed > 0 else None,
        })

    def log_extraction(
        self,
        *,
        model_name: str,
        text: str,
        n_timesteps: int,
        n_mask_configs: int,
        grad_timesteps: int,
        bundle: Any,            # MetricsBundle dataclass
        save_path: str | Path | None = None,
    ) -> None:
        """Log an extraction run — scalar metrics summary + optional bundle path."""
        self.write({
            "script":     "run_extraction",
            "model":      model_name,
            "text_preview": text[:120],
            "seq_len":    bundle.seq_len,
            "config": {
                "n_timesteps":    n_timesteps,
                "n_mask_configs": n_mask_configs,
                "grad_timesteps": grad_timesteps,
            },
            "metrics": {
                "elbo_variance":      round(bundle.elbo_variance, 6),
                "elbo_mean":          round(float(bundle.elbo_per_t.mean()), 6),
                "elbo_min":           round(float(bundle.elbo_per_t.min()),  6),
                "elbo_max":           round(float(bundle.elbo_per_t.max()),  6),
                "entropy_mean":       round(float(bundle.pred_entropy_per_t.mean()), 6),
                "entropy_max":        round(float(bundle.pred_entropy_per_t.max()),  6),
                "consistency_mean":   round(float(bundle.mask_consistency_per_t.mean()), 6),
                "consistency_min":    round(float(bundle.mask_consistency_per_t.min()),  6),
                "grad_norm_mean":     round(float(bundle.grad_norms.mean()), 6),
                "grad_norm_max":      round(float(bundle.grad_norms.max()),  6),
                "dldt_abs_mean":      round(float(bundle.dldt.abs().mean()), 6),
                "d2ldt2_abs_mean":    round(float(bundle.d2ldt2.abs().mean()), 6),
                "hidden_norm_mean":   round(float(bundle.hidden_norms.mean()), 6),
                "hidden_cosine_mean": round(float(bundle.hidden_cosine_sim.mean()), 6),
                "attention_captured": bundle.attention_maps is not None,
            },
            "elapsed_s":  round(bundle.elapsed_seconds, 3),
            "bundle_path": str(save_path) if save_path else None,
        })
