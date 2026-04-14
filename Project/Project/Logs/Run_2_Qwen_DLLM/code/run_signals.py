"""
run_signals.py — Extract white-box MetricsBundle feature vectors for all ~2000 samples.

Run_2 change vs Run_1:
  - max_length=256 passed to extract_metrics (was 128)
  - N is now ~2000 (1000 members + 1000 non-members)

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
from transformers import Qwen2Tokenizer
from tqdm import tqdm

# Import from project
sys.path.insert(0, os.path.dirname(__file__))
from mdlm_metrics_extractor import extract_metrics

# extract_metrics settings
T      = 10
K      = 8
GRAD_T = 2
SEED   = 42
MAX_LENGTH = 256


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_eager(path: str, dtype, device):
    model = AutoModelForMaskedLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    return model


def cross_model_cosine_sim(ft_bundle, base_bundle) -> float:
    """Cosine similarity between mean hidden norms of finetuned vs base model."""
    hn_ft   = ft_bundle.hidden_norms
    hn_base = base_bundle.hidden_norms

    if hn_ft is None or hn_base is None:
        return 0.0

    v_ft   = hn_ft.mean(dim=0)
    v_base = hn_base.mean(dim=0)

    min_layers = min(v_ft.shape[0], v_base.shape[0])
    v_ft   = v_ft[:min_layers].float()
    v_base = v_base[:min_layers].float()

    cos_sim = torch.nn.functional.cosine_similarity(
        v_ft.unsqueeze(0), v_base.unsqueeze(0)
    ).item()
    return cos_sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit samples per class for dry-run")
    parser.add_argument("--data_dir",  default="data")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--out_dir",   default="results")
    args = parser.parse_args()

    wandb.init(project="da5001-mia", name="run2-signals", config={
        "T": T, "K": K, "grad_timesteps": GRAD_T, "max_length": MAX_LENGTH,
        **vars(args),
    })

    device = select_device()
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    ft_path   = os.path.join(args.model_dir, "finetuned_checkpoint")
    base_path = os.path.join(args.model_dir, "base_checkpoint")

    print("Loading finetuned model (eager attn)...")
    ft_model = load_model_eager(ft_path, dtype, device)

    print("Loading base model (eager attn)...")
    base_model = load_model_eager(base_path, dtype, device)

    # Use Qwen2Tokenizer directly — transformers>=4.57 AutoTokenizer calls AutoConfig
    # first, which fails for local paths with unknown model_type 'a2d-qwen3'.
    tokenizer = Qwen2Tokenizer.from_pretrained(ft_path)

    # Load data
    members    = torch.load(os.path.join(args.data_dir, "members.pt"),    weights_only=True)
    nonmembers = torch.load(os.path.join(args.data_dir, "nonmembers.pt"), weights_only=True)

    member_texts    = members["texts"]
    nonmember_texts = nonmembers["texts"]

    if args.n_samples is not None:
        member_texts    = member_texts[: args.n_samples]
        nonmember_texts = nonmember_texts[: args.n_samples]

    all_texts = list(member_texts) + list(nonmember_texts)
    labels    = [1] * len(member_texts) + [0] * len(nonmember_texts)
    total     = len(all_texts)
    print(f"Extracting signals from {total} samples ({len(member_texts)} members, {len(nonmember_texts)} non-members)...")

    # Determine if extract_metrics accepts max_length
    import inspect
    _em_sig = inspect.signature(extract_metrics)
    _em_accepts_max_length = "max_length" in _em_sig.parameters

    feature_vectors = []
    failed = 0

    for i, text in enumerate(tqdm(all_texts, desc="Extracting signals")):
        try:
            em_kwargs = dict(
                n_timesteps=T, n_mask_configs=K,
                grad_timesteps=GRAD_T, capture_attentions=True, seed=SEED,
            )
            if _em_accepts_max_length:
                em_kwargs["max_length"] = MAX_LENGTH

            # Primary extraction on finetuned model
            bundle_ft = extract_metrics(ft_model, tokenizer, text, **em_kwargs)
            fv = bundle_ft.to_feature_vector()  # [11*T + 1] = [111]

            # Cross-model feature: base model with lighter settings
            base_kwargs = dict(
                n_timesteps=T, n_mask_configs=4,
                grad_timesteps=0, capture_attentions=False, seed=SEED,
            )
            if _em_accepts_max_length:
                base_kwargs["max_length"] = MAX_LENGTH

            bundle_base = extract_metrics(base_model, tokenizer, text, **base_kwargs)
            cos_sim = cross_model_cosine_sim(bundle_ft, bundle_base)
            cos_sim_t = torch.tensor([cos_sim], dtype=fv.dtype)

            fv_full = torch.cat([fv, cos_sim_t], dim=0)  # [112]
            feature_vectors.append(fv_full)

            wandb.log({"signals/fv_norm": fv_full.norm().item(), "signals/sample": i + 1})

        except Exception as e:
            print(f"  WARNING: sample {i} failed: {e}")
            failed += 1
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
