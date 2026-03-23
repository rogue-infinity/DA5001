"""
verify_memorization.py
======================
Load a finetuned checkpoint and confirm memorization by comparing the
MDLM masked-diffusion loss L(t) between member and non-member probe sets.

For each of N samples per group, compute unweighted CE loss at
t ∈ {0.1, 0.3, 0.5, 0.7, 0.9} and print a summary table.

Expected result: member L(t) < non-member L(t) at most t values.

Usage
-----
python verify_memorization.py
python verify_memorization.py --checkpoint checkpoints/epoch_3.pt --n_samples 20
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from shared.device import select_device
from shared.logger import build_logger
from shared.model_io import load_model_and_tokenizer

_log = build_logger("verify_memorization")

T_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_loss_at_t(
    model,
    chunk: torch.LongTensor,    # [L]
    pad_token_id: int,
    mask_token_id: int,
    t: float,
    device: torch.device,
    n_trials: int = 3,
) -> float:
    """
    Estimate L(t) for a single sequence by averaging over n_trials
    independent random masking patterns at noise level t.
    Returns NaN if no tokens are ever masked.
    """
    x = chunk.unsqueeze(0).to(device)  # [1, L]
    is_pad = (x == pad_token_id)

    losses = []
    for _ in range(n_trials):
        rand = torch.rand_like(x, dtype=torch.float)
        mask = (rand < t) & ~is_pad

        if mask.sum() == 0:
            continue

        x_masked = x.clone()
        x_masked[mask] = mask_token_id

        logits = model(x_masked).logits  # [1, L, V]
        ce = F.cross_entropy(logits[0, mask[0]], x[0, mask[0]])
        losses.append(ce.item())

    return sum(losses) / len(losses) if losses else float("nan")


def evaluate_group(
    model,
    chunks: list[torch.LongTensor],
    pad_token_id: int,
    mask_token_id: int,
    t_levels: list[float],
    device: torch.device,
    n_samples: int,
    rng: random.Random,
    group_name: str,
) -> dict[float, list[float]]:
    """
    Evaluate L(t) for each sample in a randomly drawn subset.
    Returns {t: [loss_per_sample]}.
    """
    samples = rng.sample(chunks, min(n_samples, len(chunks)))
    _log.info("[%s] Evaluating %d samples at %d t-levels …", group_name, len(samples), len(t_levels))

    results: dict[float, list[float]] = {t: [] for t in t_levels}

    for i, chunk in enumerate(samples):
        for t in t_levels:
            loss = compute_loss_at_t(model, chunk, pad_token_id, mask_token_id, t, device)
            results[t].append(loss)
        if (i + 1) % 5 == 0:
            _log.info("[%s]  %d/%d done", group_name, i + 1, len(samples))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Table printing
# ──────────────────────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    valid = [v for v in values if v == v]  # filter NaN
    return sum(valid) / len(valid) if valid else float("nan")


def print_table(
    member_results: dict[float, list[float]],
    nonmember_results: dict[float, list[float]],
) -> None:
    header = f"{'t':>6} | {'members':>10} | {'nonmembers':>12} | {'gap (nm-m)':>12}"
    sep    = "-" * len(header)
    print()
    print("Memorization Verification — L(t) Summary")
    print(sep)
    print(header)
    print(sep)
    for t in sorted(member_results.keys()):
        m  = _mean(member_results[t])
        nm = _mean(nonmember_results[t])
        gap = nm - m
        marker = " ✓" if gap > 0 else " ✗"
        print(f"{t:>6.1f} | {m:>10.4f} | {nm:>12.4f} | {gap:>+12.4f}{marker}")
    print(sep)
    print()
    print("✓ = non-member loss > member loss (expected memorization signal)")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify DLLM memorization via L(t) gap between members and non-members."
    )
    parser.add_argument(
        "--model",
        default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        help="HuggingFace model identifier (used for tokenizer config).",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/epoch_3.pt",
        help="Path to finetuned checkpoint .pt file.",
    )
    parser.add_argument("--data_dir",  default="data", help="Directory with probe_*.pt files.")
    parser.add_argument("--n_samples", type=int, default=20, help="Samples per group.")
    parser.add_argument("--log_dir",   default="logs", help="Directory for verify.jsonl output.")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = select_device()
    _log.info("Device: %s", device)

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    pad_id  = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    _log.info("pad_token_id=%s  mask_token_id=%s", pad_id, mask_id)

    if mask_id is None:
        raise ValueError("Tokenizer has no mask_token_id.")

    # Load finetuned weights
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run finetune.py first or pass --checkpoint <path>."
        )
    _log.info("Loading checkpoint from %s …", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _log.info("  Loaded epoch %s  (avg_loss=%s)", ckpt.get("epoch"), ckpt.get("avg_loss"))
    model.eval()

    # Load probe sets
    data_dir = Path(args.data_dir)
    probe_members: list[torch.LongTensor] = torch.load(
        data_dir / "probe_members.pt", weights_only=True
    )
    probe_nonmembers: list[torch.LongTensor] = torch.load(
        data_dir / "probe_nonmembers.pt", weights_only=True
    )
    _log.info("Probe sets: %d members, %d non-members", len(probe_members), len(probe_nonmembers))

    # Evaluate
    member_results = evaluate_group(
        model, probe_members, pad_id, mask_id, T_LEVELS, device,
        args.n_samples, rng, "members"
    )
    nonmember_results = evaluate_group(
        model, probe_nonmembers, pad_id, mask_id, T_LEVELS, device,
        args.n_samples, rng, "nonmembers"
    )

    # Print table
    print_table(member_results, nonmember_results)

    # Save results to logs/verify.jsonl
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    verify_log = log_dir / "verify.jsonl"

    record = {
        "checkpoint": str(ckpt_path),
        "n_samples": args.n_samples,
        "t_levels": T_LEVELS,
        "members":    {str(t): round(_mean(v), 6) for t, v in member_results.items()},
        "nonmembers": {str(t): round(_mean(v), 6) for t, v in nonmember_results.items()},
        "gaps":       {str(t): round(_mean(nonmember_results[t]) - _mean(member_results[t]), 6)
                       for t in T_LEVELS},
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with verify_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    _log.info("Results logged → %s", verify_log)


if __name__ == "__main__":
    main()
