"""
finetune.py
===========
Finetune the DLLM on data/members.pt using the MDLM masked-diffusion objective.

MDLM objective (continuous-time)
---------------------------------
  1. Sample t_b ~ Uniform(t_min, 1)  per sample in the batch
  2. Mask each non-padding token independently with probability t_b
  3. Forward pass → logits
  4. Loss = mean_b [ CE(masked positions) / t_b ]

Outputs
-------
checkpoints/epoch_{n}.pt  — model + optimizer state after each epoch
logs/finetune.jsonl        — per-step loss, per-epoch average, per-epoch sanity gap

Usage
-----
python finetune.py
python finetune.py --epochs 3 --batch_size 8 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from shared.device import select_device, model_dtype
from shared.logger import build_logger
from shared.model_io import load_model_and_tokenizer

_log = build_logger("finetune")

T_MIN = 0.01  # Clamp t away from 0 to keep 1/t finite


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TokenChunkDataset(Dataset):
    """Wraps a list[LongTensor[L]] loaded from a .pt file."""

    def __init__(self, path: str | Path) -> None:
        self.chunks: list[torch.LongTensor] = torch.load(path, weights_only=True)
        _log.info("Loaded dataset from %s  (%d sequences)", path, len(self.chunks))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.LongTensor:
        return self.chunks[idx]


# ──────────────────────────────────────────────────────────────────────────────
# MDLM loss
# ──────────────────────────────────────────────────────────────────────────────

def mdlm_loss(
    model,
    x: torch.LongTensor,          # [B, L]
    pad_token_id: int,
    mask_token_id: int,
    t_values: torch.FloatTensor,  # [B]
) -> tuple[torch.Tensor, int]:
    """
    Compute the MDLM continuous-time loss for a batch.

    Returns
    -------
    loss        : scalar tensor (mean over valid samples)
    n_valid     : number of samples that had at least one masked token
    """
    B, L = x.shape
    device = x.device

    is_pad = (x == pad_token_id)  # [B, L]

    # Sample binary mask: token i in sample b is masked iff rand < t_b
    rand = torch.rand(B, L, device=device)
    mask = (rand < t_values[:, None]) & ~is_pad  # [B, L]

    # Build masked input
    x_masked = x.clone()
    x_masked[mask] = mask_token_id

    # Forward pass (no grad needed for logits, grad flows through backward)
    logits = model(x_masked).logits  # [B, L, V]

    # Per-sample 1/t-weighted CE loss
    total_loss = torch.zeros(1, device=device, dtype=logits.dtype)
    n_valid = 0
    for b in range(B):
        mask_b = mask[b]  # [L]
        if mask_b.sum() == 0:
            continue
        ce_b = F.cross_entropy(logits[b, mask_b], x[b, mask_b])
        total_loss = total_loss + ce_b / t_values[b]
        n_valid += 1

    if n_valid == 0:
        return total_loss.squeeze(), 0

    return (total_loss / n_valid).squeeze(), n_valid


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sanity_check(
    model,
    member_chunks: list[torch.LongTensor],
    nonmember_chunks: list[torch.LongTensor],
    pad_token_id: int,
    mask_token_id: int,
    device: torch.device,
    n: int = 10,
    t_fixed: float = 0.5,
    rng: random.Random = None,
) -> tuple[float, float]:
    """
    Compute unweighted CE loss at t=t_fixed for n members and n non-members.
    Returns (member_loss, nonmember_loss).
    """
    if rng is None:
        rng = random.Random(0)

    def _eval_loss(chunks: list[torch.LongTensor]) -> float:
        samples = rng.sample(chunks, min(n, len(chunks)))
        losses = []
        for chunk in samples:
            x = chunk.unsqueeze(0).to(device)  # [1, L]
            is_pad = (x == pad_token_id)
            rand = torch.rand_like(x, dtype=torch.float)
            mask = (rand < t_fixed) & ~is_pad
            if mask.sum() == 0:
                continue
            x_masked = x.clone()
            x_masked[mask] = mask_token_id
            logits = model(x_masked).logits  # [1, L, V]
            ce = F.cross_entropy(logits[0, mask[0]], x[0, mask[0]])
            losses.append(ce.item())
        return sum(losses) / len(losses) if losses else float("nan")

    model.eval()
    m_loss = _eval_loss(member_chunks)
    nm_loss = _eval_loss(nonmember_chunks)
    model.train()
    return m_loss, nm_loss


# ──────────────────────────────────────────────────────────────────────────────
# JSON-lines logger
# ──────────────────────────────────────────────────────────────────────────────

class FinetuneLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self._path = Path(log_dir) / "finetune.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _ts(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write(self, record: dict) -> None:
        record["ts"] = self._ts()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_step(self, epoch: int, step: int, loss: float, t_mean: float) -> None:
        self._write({"event": "step", "epoch": epoch, "step": step,
                     "loss": round(loss, 6), "t_mean": round(t_mean, 4)})

    def log_epoch(self, epoch: int, avg_loss: float, elapsed_s: float) -> None:
        self._write({"event": "epoch", "epoch": epoch,
                     "avg_loss": round(avg_loss, 6), "elapsed_s": round(elapsed_s, 2)})

    def log_sanity(self, epoch: int, member_loss: float, nonmember_loss: float) -> None:
        gap = nonmember_loss - member_loss
        self._write({"event": "sanity", "epoch": epoch,
                     "member_loss": round(member_loss, 6),
                     "nonmember_loss": round(nonmember_loss, 6),
                     "gap": round(gap, 6)})


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    device = select_device()
    _log.info("Device: %s  dtype: %s", device, model_dtype(device))

    # Directories
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ft_logger = FinetuneLogger(args.log_dir)

    # Model + tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    model.train()

    pad_id  = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    _log.info("pad_token_id=%s  mask_token_id=%s", pad_id, mask_id)

    if mask_id is None:
        raise ValueError(
            "Tokenizer has no mask_token_id. "
            "The MDLM objective requires a [MASK] token."
        )

    # Datasets
    member_ds = TokenChunkDataset(Path(args.data_dir) / "members.pt")
    nonmember_chunks: list[torch.LongTensor] = torch.load(
        Path(args.data_dir) / "nonmembers.pt", weights_only=True
    )
    _log.info("Non-member sequences loaded: %d", len(nonmember_chunks))

    train_loader = DataLoader(
        member_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * (len(train_loader) // args.grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    _log.info(
        "Training: epochs=%d  steps/epoch=%d  total_opt_steps=%d  warmup=%d",
        args.epochs, len(train_loader), total_steps, args.warmup_steps,
    )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        epoch_losses: list[float] = []
        optimizer.zero_grad()

        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)  # [B, L]
            B = x.shape[0]

            # Sample t per sample, clamped to [T_MIN, 1]
            t_vals = torch.empty(B, device=device).uniform_(T_MIN, 1.0)

            loss, n_valid = mdlm_loss(model, x, pad_id, mask_id, t_vals)

            if n_valid == 0:
                _log.warning("Step %d: no masked tokens in batch — skipping", global_step)
                continue

            # Gradient accumulation
            (loss / args.grad_accum).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            loss_val = loss.item()
            t_mean_val = t_vals.mean().item()
            epoch_losses.append(loss_val)

            # Log every 50 steps
            if global_step % 50 == 0 and global_step > 0:
                _log.info(
                    "epoch=%d  step=%d  loss=%.4f  t_mean=%.3f  lr=%.2e",
                    epoch, global_step, loss_val, t_mean_val,
                    scheduler.get_last_lr()[0],
                )
                ft_logger.log_step(epoch, global_step, loss_val, t_mean_val)

        # ── End of epoch ──────────────────────────────────────────────────────
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        elapsed = time.time() - epoch_start
        _log.info(
            "Epoch %d complete | avg_loss=%.4f | elapsed=%.1fs",
            epoch, avg_loss, elapsed,
        )
        ft_logger.log_epoch(epoch, avg_loss, elapsed)

        # Checkpoint
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
                "args": vars(args),
            },
            ckpt_path,
        )
        _log.info("Checkpoint saved → %s", ckpt_path)

        # Sanity check
        _log.info("Running sanity check (t=0.5, n=10) …")
        m_loss, nm_loss = sanity_check(
            model,
            member_ds.chunks,
            nonmember_chunks,
            pad_id,
            mask_id,
            device,
            n=10,
            t_fixed=0.5,
            rng=rng,
        )
        gap = nm_loss - m_loss
        _log.info(
            "Sanity check | member_loss=%.4f | nonmember_loss=%.4f | gap=%.4f",
            m_loss, nm_loss, gap,
        )
        ft_logger.log_sanity(epoch, m_loss, nm_loss)

        model.train()

    _log.info("Training complete. Final checkpoint: %s", ckpt_dir / f"epoch_{args.epochs}.pt")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finetune DLLM on members corpus using MDLM masked-diffusion objective."
    )
    parser.add_argument(
        "--model",
        default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        help="HuggingFace model identifier.",
    )
    parser.add_argument("--data_dir",       default="data",        help="Directory with members.pt / nonmembers.pt.")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory for epoch checkpoints.")
    parser.add_argument("--log_dir",        default="logs",        help="Directory for finetune.jsonl.")
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--grad_accum",     type=int,   default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--warmup_steps",   type=int,   default=100)
    parser.add_argument("--num_workers",    type=int,   default=4,   help="DataLoader worker processes (0 = main process only).")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
