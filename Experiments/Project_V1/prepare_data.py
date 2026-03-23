"""
prepare_data.py
===============
Download WikiText-103, tokenize with the model's own tokenizer, chunk into
fixed-length sequences, and save member / non-member corpora for MIA experiments.

Outputs
-------
data/members.pt          — list[LongTensor[seq_len]]  (train split)
data/nonmembers.pt       — list[LongTensor[seq_len]]  (test split)
data/probe_members.pt    — list[LongTensor[seq_len]]  (500 random from train)
data/probe_nonmembers.pt — list[LongTensor[seq_len]]  (500 random from test)

Usage
-----
python prepare_data.py
python prepare_data.py --seq_len 256 --probe_size 500 --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from shared.logger import build_logger

_log = build_logger("prepare_data")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_tokenizer(model_name: str) -> AutoTokenizer:
    _log.info("Loading tokenizer from %s …", model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        _log.info("  pad_token set to eos_token (%r)", tok.eos_token)
    _log.info(
        "  vocab_size=%d  mask_token_id=%s  pad_token_id=%s",
        tok.vocab_size,
        tok.mask_token_id,
        tok.pad_token_id,
    )
    return tok


def _split_to_chunks(
    split_texts: list[str],
    tokenizer: AutoTokenizer,
    seq_len: int,
    split_name: str,
) -> list[torch.LongTensor]:
    """
    Concatenate all non-empty text rows, tokenize as one long sequence
    (no special tokens), then slice into non-overlapping chunks of exactly
    seq_len tokens.  The final partial chunk is dropped.
    """
    _log.info("[%s] Concatenating %d rows …", split_name, len(split_texts))
    full_text = "\n".join(t for t in split_texts if t.strip())
    _log.info("[%s]  Total characters: %d", split_name, len(full_text))

    _log.info("[%s] Tokenizing (this may take a minute for large splits) …", split_name)
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    )
    ids: torch.LongTensor = enc["input_ids"][0]  # [total_tokens]
    _log.info("[%s]  Total tokens: %d", split_name, ids.shape[0])

    n_chunks = ids.shape[0] // seq_len
    chunks = [ids[i * seq_len : (i + 1) * seq_len] for i in range(n_chunks)]
    _log.info(
        "[%s]  Chunks of %d tokens: %d  (dropped %d trailing tokens)",
        split_name,
        seq_len,
        len(chunks),
        ids.shape[0] - n_chunks * seq_len,
    )
    return chunks


def _probe_sample(
    chunks: list[torch.LongTensor],
    probe_size: int,
    rng: random.Random,
) -> list[torch.LongTensor]:
    """Return a random subset of size min(probe_size, len(chunks))."""
    n = min(probe_size, len(chunks))
    return rng.sample(chunks, n)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare WikiText-103 member/non-member corpora for MIA."
    )
    parser.add_argument(
        "--model",
        default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        help="HuggingFace model ID — used only to load the tokenizer.",
    )
    parser.add_argument("--seq_len",    type=int, default=256,  help="Chunk size in tokens.")
    parser.add_argument("--probe_size", type=int, default=500,  help="Samples per probe set.")
    parser.add_argument("--data_dir",   default="data",         help="Output directory.")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # Output directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Output directory: %s", data_dir.resolve())

    # Tokenizer
    tokenizer = _load_tokenizer(args.model)

    # Dataset
    _log.info("Loading wikitext-103-raw-v1 from HuggingFace datasets …")
    from datasets import load_dataset  # import here to keep top-level deps light
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    _log.info("  Available splits: %s", list(dataset.keys()))

    # Member corpus — train split
    train_texts: list[str] = dataset["train"]["text"]
    member_chunks = _split_to_chunks(train_texts, tokenizer, args.seq_len, "train")

    # Non-member corpus — test split
    test_texts: list[str] = dataset["test"]["text"]
    nonmember_chunks = _split_to_chunks(test_texts, tokenizer, args.seq_len, "test")

    # Probe sets (random subsets)
    probe_members    = _probe_sample(member_chunks,    args.probe_size, rng)
    probe_nonmembers = _probe_sample(nonmember_chunks, args.probe_size, rng)

    # Save
    paths = {
        "members.pt":          member_chunks,
        "nonmembers.pt":       nonmember_chunks,
        "probe_members.pt":    probe_members,
        "probe_nonmembers.pt": probe_nonmembers,
    }
    for fname, data in paths.items():
        out_path = data_dir / fname
        torch.save(data, out_path)
        _log.info("Saved %s  (%d sequences)", out_path, len(data))

    # Sanity verification
    _log.info("─" * 60)
    _log.info("Verification:")
    for fname in paths:
        loaded = torch.load(data_dir / fname, weights_only=True)
        shapes = set(tuple(t.shape) for t in loaded)
        _log.info("  %-24s  n=%6d  shapes=%s", fname, len(loaded), shapes)

    _log.info("Done.")


if __name__ == "__main__":
    main()
