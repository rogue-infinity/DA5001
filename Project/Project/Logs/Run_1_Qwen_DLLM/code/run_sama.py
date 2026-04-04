"""
run_sama.py — SAMA baseline: exact Algorithms 1+2 from arXiv 2601.20125.

Credit: SAMA algorithm from "Membership Inference Attacks Against Fine-tuned
Diffusion Language Models" (arXiv 2601.20125, MIT licensed, github.com/Stry233/SAMA).
This implementation is original; the method is credited to the original authors.

Key algorithmic details (verified against official GitHub source):
  - Mask is CUMULATIVE across steps (not freshly sampled each step)
  - Losses stored only for NEWLY added positions at each step
  - Subset sampling uses ALL cumulative masked positions
  - Alpha schedule: frac = α_min + (α_max - α_min)*(step+1)/(T+1)
  - Harmonic weights: w_s = 1/(s+1), normalized
  - Subset comparison: sum(ℓ_R) > sum(ℓ_T) per subset (sign test)

Parameters: T=16, alpha_min=0.05, alpha_max=0.50, N=128, m=10
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm


# SAMA hyperparameters (from paper)
T = 16
ALPHA_MIN = 0.05
ALPHA_MAX = 0.50
N_SUBSETS = 128
SUBSET_SIZE = 10


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def harmonic_weights(n: int) -> np.ndarray:
    """w_s = 1/(s+1) for s=0..n-1, normalized. Matches GitHub exactly."""
    w = 1.0 / (np.arange(n) + 1)
    return w / w.sum()


def tpr_at_fpr(y_true, y_score, fpr_threshold: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(fpr <= fpr_threshold)[0]
    return float(tpr[idx[-1]]) if len(idx) > 0 else 0.0


@torch.no_grad()
def compute_batch_ce(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     mask_token_id: int) -> torch.Tensor:
    """Forward pass → per-token CE for all positions. Returns [B, L] float32.

    Note: computes CE at every position (including unmasked). The caller
    selects only the relevant positions. This matches the GitHub implementation.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, L, V]
    B, L, V = logits.shape
    ce = F.cross_entropy(
        logits.view(B * L, V),
        input_ids.view(B * L),
        reduction="none",
    ).view(B, L).float()
    return ce


def sama_score_single(
    target_model, ref_model,
    input_ids: torch.Tensor,      # [L]
    attention_mask: torch.Tensor, # [L]
    mask_token_id_target: int,
    mask_token_id_ref: int,
    device: torch.device,
    T: int,
    alpha_min: float,
    alpha_max: float,
    N: int,
    m: int,
    rng: np.random.RandomState,
) -> float:
    """Compute SAMA membership score for one sample. Returns phi ∈ [0,1].

    Exactly follows the official GitHub implementation (Stry233/SAMA):
      - Cumulative mask built step by step
      - Losses stored only for newly added positions per step
      - Subsets sampled from all cumulative masked positions
    """
    L = input_ids.shape[0]
    valid = attention_mask.bool()   # [L] — non-padding positions
    Lb = int(valid.sum().item())    # real sequence length

    if Lb < m:
        return 0.5  # too short to score reliably

    # Running state
    cumulative_mask = torch.zeros(L, dtype=torch.bool, device=device)
    target_losses_full = torch.zeros(L, dtype=torch.float32, device=device)
    ref_losses_full    = torch.zeros(L, dtype=torch.float32, device=device)

    step_scores = []

    for step in range(T):
        # ---- Masking density (matches GitHub linear schedule) ----
        frac = alpha_min + (alpha_max - alpha_min) * (step + 1) / (T + 1)
        desired_total = max(1, int(round(frac * Lb)))
        current_total = int((cumulative_mask & valid).sum().item())
        to_add = max(0, desired_total - current_total)

        if to_add == 0:
            # Already at desired density; still record a score from current state
            pass
        else:
            # Sample new positions from unmasked valid tokens
            unmasked_valid = (~cumulative_mask) & valid
            candidates = torch.where(unmasked_valid)[0]
            to_add = min(to_add, candidates.numel())

            if to_add > 0:
                perm = torch.randperm(candidates.numel(), device=device)
                chosen = candidates[perm[:to_add]]
                new_mask = torch.zeros(L, dtype=torch.bool, device=device)
                new_mask[chosen] = True

                # Build noised inputs with full cumulative mask
                cumulative_mask = cumulative_mask | new_mask
                noised_target = input_ids.clone()
                noised_target[cumulative_mask] = mask_token_id_target
                noised_ref = input_ids.clone()
                noised_ref[cumulative_mask] = mask_token_id_ref

                # Forward passes — both models, batched as [1, L]
                ce_target = compute_batch_ce(
                    target_model,
                    noised_target.unsqueeze(0),
                    attention_mask.unsqueeze(0),
                    mask_token_id_target,
                ).squeeze(0)   # [L]
                ce_ref = compute_batch_ce(
                    ref_model,
                    noised_ref.unsqueeze(0),
                    attention_mask.unsqueeze(0),
                    mask_token_id_ref,
                ).squeeze(0)   # [L]

                # Store only for newly added positions
                target_losses_full[new_mask] = ce_target[new_mask]
                ref_losses_full[new_mask]    = ce_ref[new_mask]

        # ---- Subset binary comparison ----
        masked_positions = torch.where(cumulative_mask)[0]
        total_masked = masked_positions.numel()
        if total_masked < m:
            continue   # not enough positions yet

        t_losses = target_losses_full[masked_positions].cpu().numpy()
        r_losses = ref_losses_full[masked_positions].cpu().numpy()

        # Sample N subsets of size m from total_masked positions
        subset_size = min(m, total_masked)
        idx_matrix = np.vstack([
            rng.choice(total_masked, size=subset_size, replace=False)
            for _ in range(N)
        ]).astype(np.int64)   # [N, m]

        t_sel = t_losses[idx_matrix].sum(axis=1)   # [N]
        r_sel = r_losses[idx_matrix].sum(axis=1)   # [N]
        comparisons = (r_sel > t_sel)               # [N] bool
        beta_t = float(comparisons.mean())
        step_scores.append(beta_t)

    if not step_scores:
        return 0.5

    weights = harmonic_weights(len(step_scores))
    phi = float(np.average(step_scores, weights=weights))
    return phi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit total samples (for dry-run)")
    parser.add_argument("--n_comparisons", type=int, default=N_SUBSETS,
                        help=f"Number of subsets N per masking step (default {N_SUBSETS})")
    parser.add_argument("--T", type=int, default=T,
                        help=f"Number of masking steps (default {T})")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wandb.init(project="da5001-mia", name="sama", config={
        "T": args.T, "alpha_min": ALPHA_MIN, "alpha_max": ALPHA_MAX,
        "N": args.n_comparisons, "m": SUBSET_SIZE,
        **vars(args),
    })

    device = select_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Load both models
    target_path = os.path.join(args.model_dir, "finetuned_checkpoint")
    ref_path    = os.path.join(args.model_dir, "base_checkpoint")

    print("Loading target model (finetuned)...")
    target_model = AutoModelForMaskedLM.from_pretrained(
        target_path, trust_remote_code=True, dtype=dtype
    ).to(device).eval()

    print("Loading reference model (base)...")
    ref_model = AutoModelForMaskedLM.from_pretrained(
        ref_path, trust_remote_code=True, dtype=dtype
    ).to(device).eval()

    target_tok = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
    ref_tok    = AutoTokenizer.from_pretrained(ref_path,    trust_remote_code=True)
    mask_id_target = target_tok.mask_token_id
    mask_id_ref    = ref_tok.mask_token_id

    # Load data
    members    = torch.load(os.path.join(args.data_dir, "members.pt"),    weights_only=True)
    nonmembers = torch.load(os.path.join(args.data_dir, "nonmembers.pt"), weights_only=True)

    all_ids  = torch.cat([members["input_ids"],    nonmembers["input_ids"]],    dim=0)
    all_mask = torch.cat([members["attention_mask"], nonmembers["attention_mask"]], dim=0)
    n_mem    = members["input_ids"].shape[0]
    n_nonmem = nonmembers["input_ids"].shape[0]
    labels   = torch.cat([torch.ones(n_mem), torch.zeros(n_nonmem)])

    if args.n_samples is not None:
        half = args.n_samples // 2
        mem_idx    = list(range(min(half, n_mem)))
        nonmem_idx = list(range(n_mem, n_mem + min(half, n_nonmem)))
        idx = mem_idx + nonmem_idx
        all_ids, all_mask, labels = all_ids[idx], all_mask[idx], labels[idx]
        print(f"Dry-run: {len(mem_idx)} members + {len(nonmem_idx)} non-members")

    total = all_ids.shape[0]
    rng = np.random.RandomState(args.seed)
    scores = []

    print(f"\nRunning SAMA on {total} samples (T={args.T}, N={args.n_comparisons}, m={SUBSET_SIZE})...")
    for i in tqdm(range(total)):
        ids  = all_ids[i].to(device)
        mask = all_mask[i].to(device)

        phi = sama_score_single(
            target_model, ref_model,
            ids, mask,
            mask_id_target, mask_id_ref,
            device,
            T=args.T, alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX,
            N=args.n_comparisons, m=SUBSET_SIZE,
            rng=rng,
        )
        scores.append(phi)

        if (i + 1) % 50 == 0:
            wandb.log({"sama/progress": (i + 1) / total, "sama/sample": i + 1})

    scores_arr = np.array(scores)
    labels_arr = labels.numpy()

    auc    = roc_auc_score(labels_arr, scores_arr)
    tpr_01 = tpr_at_fpr(labels_arr, scores_arr, 0.001)
    tpr_1  = tpr_at_fpr(labels_arr, scores_arr, 0.01)
    tpr_10 = tpr_at_fpr(labels_arr, scores_arr, 0.10)

    print(f"\nSAMA Results:")
    print(f"  AUC:          {auc:.4f}")
    print(f"  TPR@0.1%FPR:  {tpr_01:.4f}")
    print(f"  TPR@1%FPR:    {tpr_1:.4f}")
    print(f"  TPR@10%FPR:   {tpr_10:.4f}")

    wandb.log({
        "sama/auc":          auc,
        "sama/tpr_at_0.1fpr": tpr_01,
        "sama/tpr_at_1fpr":   tpr_1,
        "sama/tpr_at_10fpr":  tpr_10,
    })

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "sama_scores.pt")
    torch.save({"scores": torch.tensor(scores_arr), "labels": labels}, out_path)
    print(f"Saved {out_path}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
