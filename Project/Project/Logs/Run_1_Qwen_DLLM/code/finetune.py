"""
finetune.py — Fine-tune MDLM on member data to induce memorization.

Steps:
  1. Load base model from HF cache snapshot
  2. Save base weights to models/base_checkpoint/
  3. Train inline MDLM loss loop on members.pt × 3 repetitions for N epochs
  4. Save fine-tuned weights to models/finetuned_checkpoint/

Outputs:
  models/base_checkpoint/
  models/finetuned_checkpoint/
"""

import argparse
import math
import os
import random

import torch
import wandb
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1"
    "/snapshots/39c33255701becfc61f3052d4b86c80b6d36603f"
)
TIME_EPSILON = 1e-3


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mdlm_loss(model, input_ids, attention_mask, mask_token_id, device):
    """Inline MDLM ELBO loss for a single batch.

    Linear scheduler: α(t) = 1-t, so p_mask = t.
    CE on masked positions only, normalized by masked token count.
    """
    B, L = input_ids.shape
    t = torch.empty(B, device=device).uniform_(TIME_EPSILON, 1.0)  # [B]

    # p_mask per token = t broadcast to [B, L]
    p_mask = t.unsqueeze(1).expand(B, L)

    # Only mask real (non-padding) positions
    rand = torch.rand(B, L, device=device)
    mask_positions = (rand < p_mask) & (attention_mask == 1)

    if mask_positions.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    noised = input_ids.clone()
    noised[mask_positions] = mask_token_id

    logits = model(input_ids=noised, attention_mask=attention_mask).logits  # [B, L, V]

    # CE on masked positions
    logits_flat = logits[mask_positions]          # [M, V]
    targets_flat = input_ids[mask_positions]      # [M]
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--repetitions", type=int, default=3,
                        help="How many times to repeat training data per epoch")
    args = parser.parse_args()

    wandb.init(project="da5001-mia", name="finetune", config=vars(args))

    device = select_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Load base model
    print(f"Loading base model from {MODEL_PATH}")
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    mask_token_id = tokenizer.mask_token_id
    assert mask_token_id is not None, "tokenizer has no mask_token_id"
    print(f"mask_token_id = {mask_token_id}")

    # Save base checkpoint
    base_path = os.path.join(args.out_dir, "base_checkpoint")
    os.makedirs(base_path, exist_ok=True)
    model.save_pretrained(base_path)
    tokenizer.save_pretrained(base_path)
    print(f"Saved base checkpoint to {base_path}")

    # Load training data
    members = torch.load(os.path.join(args.data_dir, "members.pt"), weights_only=True)
    input_ids = members["input_ids"]          # [N, 128]
    attention_mask = members["attention_mask"]  # [N, 128]
    N = input_ids.shape[0]
    print(f"Training on {N} member samples × {args.repetitions} repetitions = {N * args.repetitions} per epoch")

    # Optimizer
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    global_step = 0
    for epoch in range(args.n_epochs):
        # Repeat and shuffle indices
        indices = list(range(N)) * args.repetitions
        random.shuffle(indices)

        epoch_loss = 0.0
        n_batches = math.ceil(len(indices) / args.batch_size)

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{args.n_epochs}")
        for step in pbar:
            batch_idx = indices[step * args.batch_size : (step + 1) * args.batch_size]
            ids = input_ids[batch_idx].to(device)
            mask = attention_mask[batch_idx].to(device)

            optimizer.zero_grad()
            loss = mdlm_loss(model, ids, mask, mask_token_id, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            wandb.log({"train/loss": loss_val, "step": global_step})

        mean_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1} mean loss: {mean_loss:.4f}")
        wandb.log({"train/epoch_loss": mean_loss, "epoch": epoch + 1})

    # Save fine-tuned checkpoint
    model.eval()
    ft_path = os.path.join(args.out_dir, "finetuned_checkpoint")
    os.makedirs(ft_path, exist_ok=True)
    model.save_pretrained(ft_path)
    tokenizer.save_pretrained(ft_path)
    print(f"Saved finetuned checkpoint to {ft_path}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
