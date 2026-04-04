"""
prepare_data.py — Pull MIMIR GitHub subset, tokenize, save balanced splits.

MIMIR dataset structure (iamgroot42/mimir, config="github"):
  Each row contains both a member text and a non-member text:
    {member: str, nonmember: str, member_neighbors: list, nonmember_neighbors: list}
  Split used: ngram_13_0.2 (~740 rows → up to 300 members + 300 non-members)

Outputs:
  data/members.pt     — {input_ids [N,128], attention_mask [N,128], texts [N]}
  data/nonmembers.pt  — same structure

Requires:
  HF_TOKEN env var set (dataset is gated at iamgroot42/mimir)
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer
from tqdm import tqdm

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--dllm-hub--Qwen3-0.6B-diffusion-mdlm-v0.1"
    "/snapshots/39c33255701becfc61f3052d4b86c80b6d36603f"
)

MIMIR_SPLIT = "ngram_13_0.2"   # 740 rows, best balance of size and n-gram threshold


def tokenize_texts(texts: list, tokenizer, max_length: int = 128):
    enc = tokenizer(
        list(texts),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=300,
                        help="Number of member and non-member samples each")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit("ERROR: HF_TOKEN env var not set. Export it before running.")

    # Log in globally so all datasets/hub calls authenticate correctly
    login(token=hf_token, add_to_git_credential=False)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"\nLoading MIMIR github split='{MIMIR_SPLIT}'...")
    ds = load_dataset(
        "iamgroot42/mimir",
        "github",
        split=MIMIR_SPLIT,
        token=hf_token,
        trust_remote_code=True,
    )
    print(f"  Total rows: {len(ds)}, columns: {ds.column_names}")

    n = min(args.n_samples, len(ds))
    ds = ds.select(range(n))

    # Each row has both member and nonmember text
    member_texts = ds["member"]
    nonmember_texts = ds["nonmember"]
    print(f"  Using {n} rows → {n} members + {n} non-members")

    for texts, out_name in [(member_texts, "members"), (nonmember_texts, "nonmembers")]:
        print(f"\nTokenizing {out_name} (max_length={args.max_length})...")
        batch_size = 64
        all_input_ids = []
        all_attention_mask = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"  {out_name}"):
            batch = texts[i : i + batch_size]
            ids, mask = tokenize_texts(batch, tokenizer, args.max_length)
            all_input_ids.append(ids)
            all_attention_mask.append(mask)

        input_ids = torch.cat(all_input_ids, dim=0)        # [N, 128]
        attention_mask = torch.cat(all_attention_mask, dim=0)  # [N, 128]

        out_path = os.path.join(args.out_dir, f"{out_name}.pt")
        torch.save({"input_ids": input_ids, "attention_mask": attention_mask, "texts": texts}, out_path)
        print(f"  Saved {out_path} — shape: {input_ids.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
