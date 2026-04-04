"""
run_signals.py — Extract white-box MetricsBundle feature vectors for all 600 samples.

For each sample:
  1. extract_metrics() on finetuned model  → feature vector [111]
  2. Cross-model cosine sim (finetuned vs base hidden_norms) → scalar → append → [112]

Outputs:
  results/X.pt  — feature matrix [N, 112]
  results/y.pt  — labels [N]  (1=member, 0=non-member)
"""

import argparse
import os
import sys

import torch
import wandb
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

# Import from project
sys.path.insert(0, os.path.dirname(__file__))
from mdlm_metrics_extractor import extract_metrics

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1"
    "/snapshots/39c33255701becfc61f3052d4b86c80b6d36603f"
)

# extract_metrics settings
T = 10
K = 8
GRAD_T = 2
SEED = 42


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_eager(path: str, dtype, device):
    model = AutoModelForMaskedLM.from_pretrained(
        path, trust_remote_code=True, dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    return model


def cross_model_cosine_sim(ft_bundle, base_bundle) -> float:
    """Cosine similarity between mean hidden norms of finetuned vs base model."""
    # hidden_norms shape: [T, n_layers]
    hn_ft = ft_bundle.hidden_norms    # [T, L]
    hn_base = base_bundle.hidden_norms  # [T, L] (T may differ)

    if hn_ft is None or hn_base is None:
        return 0.0

    # Mean over timesteps → [n_layers]
    v_ft = hn_ft.mean(dim=0)
    v_base = hn_base.mean(dim=0)

    # Align layer dimension (in case they differ — shouldn't, same arch)
    min_layers = min(v_ft.shape[0], v_base.shape[0])
    v_ft = v_ft[:min_layers].float()
    v_base = v_base[:min_layers].float()

    cos_sim = torch.nn.functional.cosine_similarity(
        v_ft.unsqueeze(0), v_base.unsqueeze(0)
    ).item()
    return cos_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit samples per class for dry-run")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    wandb.init(project="da5001-mia", name="signals", config={
        "T": T, "K": K, "grad_timesteps": GRAD_T,
        **vars(args),
    })

    device = select_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    ft_path = os.path.join(args.model_dir, "finetuned_checkpoint")
    base_path = os.path.join(args.model_dir, "base_checkpoint")

    print("Loading finetuned model (eager attn)...")
    ft_model = load_model_eager(ft_path, dtype, device)

    print("Loading base model (eager attn)...")
    base_model = load_model_eager(base_path, dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(ft_path, trust_remote_code=True)

    # Load data
    members = torch.load(os.path.join(args.data_dir, "members.pt"), weights_only=True)
    nonmembers = torch.load(os.path.join(args.data_dir, "nonmembers.pt"), weights_only=True)

    member_texts = members["texts"]
    nonmember_texts = nonmembers["texts"]

    if args.n_samples is not None:
        member_texts = member_texts[: args.n_samples]
        nonmember_texts = nonmember_texts[: args.n_samples]

    all_texts = list(member_texts) + list(nonmember_texts)
    labels = [1] * len(member_texts) + [0] * len(nonmember_texts)
    total = len(all_texts)
    print(f"Extracting signals from {total} samples ({len(member_texts)} members, {len(nonmember_texts)} non-members)...")

    feature_vectors = []
    failed = 0

    for i, text in enumerate(tqdm(all_texts, desc="Extracting signals")):
        try:
            # Primary extraction on finetuned model
            bundle_ft = extract_metrics(
                ft_model, tokenizer, text,
                n_timesteps=T, n_mask_configs=K,
                grad_timesteps=GRAD_T, capture_attentions=True, seed=SEED,
            )
            fv = bundle_ft.to_feature_vector()  # [11*T + 1] = [111]

            # Cross-model feature: base model with lighter settings
            bundle_base = extract_metrics(
                base_model, tokenizer, text,
                n_timesteps=T, n_mask_configs=4,
                grad_timesteps=0, capture_attentions=False, seed=SEED,
            )
            cos_sim = cross_model_cosine_sim(bundle_ft, bundle_base)
            cos_sim_t = torch.tensor([cos_sim], dtype=fv.dtype)

            fv_full = torch.cat([fv, cos_sim_t], dim=0)  # [112]
            feature_vectors.append(fv_full)

            fv_norm = fv_full.norm().item()
            wandb.log({"signals/fv_norm": fv_norm, "signals/sample": i + 1})

        except Exception as e:
            print(f"  WARNING: sample {i} failed: {e}")
            failed += 1
            # Append zeros to keep alignment
            feat_dim = 11 * T + 1 + 1  # 112
            feature_vectors.append(torch.zeros(feat_dim))

    print(f"\nExtraction complete. Failed: {failed}/{total}")

    X = torch.stack(feature_vectors)  # [N, 112]
    y = torch.tensor(labels, dtype=torch.long)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(X, os.path.join(args.out_dir, "X.pt"))
    torch.save(y, os.path.join(args.out_dir, "y.pt"))
    print(f"Saved X.pt {X.shape}, y.pt {y.shape}")

    wandb.log({"signals/total": total, "signals/failed": failed, "signals/feat_dim": X.shape[1]})
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
