"""
mdlm_qwen3_test.py
==================
Production-grade test harness for Qwen3-0.6B-diffusion-mdlm-v0.1
A masked-diffusion language model (MDLM) adapted from Qwen3-0.6B.

Reference: https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1
Paper:     MDLM — arXiv:2406.07524 (Sahoo et al., NeurIPS 2024)
           dLLM — arXiv:2602.22661

Device policy
--------------
  - If CUDA is available  → use cuda  (fp32 / bf16)
  - If MPS  is available  → use mps   (fp32 only; bf16 unsupported on MPS)
  - Otherwise             → cpu       (fp32)

Usage
-----
    python mdlm_qwen3_test.py
    python mdlm_qwen3_test.py --steps 64 --max_new_tokens 128 --prompt "Explain MDLM."

Shared utilities (device, model loading, batch prep, logging) live in
mdlm_shared.py so they can be imported cleanly by other scripts.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from mdlm_shared import (
    RunLogger,
    build_logger,
    load_model_and_tokenizer,
    prepare_batch,
    select_device,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
log = build_logger("mdlm_test")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GenerationConfig:
    """
    All knobs exposed by the MDLM iterative denoising loop.

    steps          : How many denoising iterations to run.  More = better
                     quality, slower.  Must be divisible by (max_new_tokens /
                     block_size).
    max_new_tokens : How many tokens to generate per sample.  Must be a
                     multiple of block_size.
    block_size     : Semi-autoregressive block size — tokens are generated
                     left-to-right in chunks of this size, each chunk fully
                     denoised before the next begins.  64 is the model default.
    temperature    : Gumbel noise temperature injected into the logit sampling
                     step.  0.0 = deterministic argmax; >0 = stochastic.
    cfg_scale      : Classifier-free guidance scale.  0.0 = disabled.
                     >0 requires a masked (unconditional) forward pass per step.
    remasking      : Strategy for deciding which predicted tokens to re-mask at
                     each step.
                     "low_confidence" → keep only the top-k most confident
                     predictions; re-mask the rest.
                     "random"         → keep k randomly chosen predictions.
    """
    steps: int = 128
    max_new_tokens: int = 256
    block_size: int = 64
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: Literal["low_confidence", "random"] = "low_confidence"


# ──────────────────────────────────────────────────────────────────────────────
# Core MDLM generation utilities
# ──────────────────────────────────────────────────────────────────────────────
def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Inject Gumbel noise into the logits for stochastic sampling.

    The MDLM paper samples x_0 predictions by computing:
        logits_noisy = exp(logits) / (-log(U))^temperature,   U ~ Uniform(0,1)
    At temperature=0 this reduces to standard argmax (deterministic).

    Args:
        logits      : Raw model logits  [..., vocab_size], any float dtype.
        temperature : Noise strength.   0.0 = no noise.

    Returns:
        Noisy logits of the same shape (cast to float64 for numerical stability).
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    u = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(u)) ** temperature
    return logits.exp() / gumbel_noise


def compute_transfer_schedule(
    mask_index: torch.BoolTensor,
    steps: int,
) -> torch.LongTensor:
    """
    For each sample in the batch, split the masked positions evenly across
    `steps` denoising steps — "how many tokens to unmask at step i".

    Implements the uniform unmasking schedule: if a block has N masked tokens
    and S steps, each step uncovers floor(N/S) tokens, with the first (N mod S)
    steps each uncovering one extra to handle the remainder exactly.

    Args:
        mask_index : BoolTensor [batch, seq_len] — True where token is masked.
        steps      : Number of denoising iterations for this block.

    Returns:
        LongTensor [batch, steps] — number of tokens to transfer at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    schedule = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long)
        + base
    )
    for i in range(mask_num.size(0)):
        schedule[i, : remainder[i]] += 1
    return schedule


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    prompt: torch.LongTensor,
    prompt_lens: torch.LongTensor,
    pad_id: int,
    cfg: GenerationConfig,
) -> torch.LongTensor:
    """
    MDLM iterative masked-diffusion generation loop.

    High-level algorithm
    --------------------
    1.  Build input sequence:  [prompt tokens | MASK ... MASK]
        (max_new_tokens masks appended after each prompt).
    2.  Semi-autoregressive blocking: split the generation region into
        (max_new_tokens / block_size) consecutive blocks.
    3.  For each block, run `steps_per_block` denoising steps:
          a. Forward pass through the model → logits for all masked positions.
          b. (Optional) CFG: run a second unconditional forward and combine.
          c. Sample x_0 predictions via Gumbel-argmax.
          d. Score each prediction by confidence or random score.
          e. Unmask the top-k highest-confidence predictions for this step
             (k = schedule[step]); re-mask the rest.
    4.  Return the fully denoised sequence.

    Args:
        model       : Loaded AutoModelForMaskedLM in eval mode.
        tokenizer   : Corresponding tokenizer.
        prompt      : LongTensor [batch, max_prompt_len] — padded prompts.
        prompt_lens : LongTensor [batch]                 — actual prompt lengths.
        pad_id      : Token id used for left-padding.
        cfg         : GenerationConfig instance.

    Returns:
        LongTensor [batch, total_length] — full sequences (prompt + generated).
    """
    mask_id = tokenizer.mask_token_id
    device = model.device
    batch_size = prompt.size(0)

    # ── 1. Construct full sequence tensor ────────────────────────────────────
    total_length = int(prompt_lens.max().item() + cfg.max_new_tokens)
    x = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=device)
    for i, length in enumerate(prompt_lens.tolist()):
        x[i, :length] = prompt[i, :length]
        x[i, length : length + cfg.max_new_tokens] = mask_id

    prompt_index = (
        torch.arange(total_length, device=device).unsqueeze(0) < prompt_lens.unsqueeze(1)
    )
    positions = torch.arange(total_length, device=device)

    # ── 2. Validate block / step divisibility ────────────────────────────────
    assert cfg.max_new_tokens % cfg.block_size == 0, (
        f"max_new_tokens ({cfg.max_new_tokens}) must be divisible by "
        f"block_size ({cfg.block_size})"
    )
    num_blocks = cfg.max_new_tokens // cfg.block_size

    assert cfg.steps % num_blocks == 0, (
        f"steps ({cfg.steps}) must be divisible by num_blocks ({num_blocks})"
    )
    steps_per_block = cfg.steps // num_blocks

    # ── 3. Semi-autoregressive denoising (one block at a time) ───────────────
    for block_idx in range(num_blocks):
        block_start = prompt_lens + block_idx * cfg.block_size
        block_end   = block_start + cfg.block_size

        init_block_mask = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        transfer_schedule = compute_transfer_schedule(init_block_mask, steps_per_block)

        for step in range(steps_per_block):
            block_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )

            # ── Forward pass (with optional CFG) ─────────────────────────────
            if cfg.cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_in = torch.cat([x, un_x], dim=0)
                logits_both = model(x_in).logits
                logits_cond, logits_uncond = torch.chunk(logits_both, 2, dim=0)
                logits = logits_uncond + (cfg.cfg_scale + 1.0) * (
                    logits_cond - logits_uncond
                )
            else:
                logits = model(x).logits

            # ── Sample x_0 predictions ───────────────────────────────────────
            logits_noisy = add_gumbel_noise(logits, temperature=cfg.temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)

            # ── Compute per-position confidence score ─────────────────────────
            if cfg.remasking == "low_confidence":
                probs = F.softmax(logits, dim=-1)
                x0_confidence = torch.gather(
                    probs, dim=-1, index=x0.unsqueeze(-1)
                ).squeeze(-1)
            elif cfg.remasking == "random":
                x0_confidence = torch.rand_like(x0, dtype=torch.float)
            else:
                raise ValueError(f"Unknown remasking strategy: {cfg.remasking!r}")

            # ── Select top-k tokens to unmask this step ───────────────────────
            confidence = torch.full_like(x0_confidence, -float("inf"))
            confidence = torch.where(block_mask, x0_confidence, confidence)
            x0 = torch.where(block_mask, x0, x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for sample_idx in range(batch_size):
                k = int(transfer_schedule[sample_idx, step].item())
                if k == 0:
                    continue
                _, top_indices = torch.topk(confidence[sample_idx], k=k)
                transfer_index[sample_idx, top_indices] = True

            x[transfer_index] = x0[transfer_index]

    return x


# ──────────────────────────────────────────────────────────────────────────────
# Output decoding
# ──────────────────────────────────────────────────────────────────────────────
def decode_outputs(
    generated: torch.LongTensor,
    prompt_lens: torch.LongTensor,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> list[str]:
    """
    Slice out only the generated region [prompt_len : prompt_len+max_new_tokens]
    and decode each sample, preserving special tokens for inspection.
    """
    decoded = []
    for i in range(generated.size(0)):
        start = int(prompt_lens[i].item())
        end   = start + max_new_tokens
        tokens = generated[i, start:end].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        decoded.append(text)
    return decoded


# ──────────────────────────────────────────────────────────────────────────────
# Default test prompts
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MESSAGES = [
    [
        {"role": "system",  "content": "You are a helpful AI assistant."},
        {"role": "user",    "content": "What is masked diffusion language modeling in two sentences?"},
    ],
    [
        {"role": "system",  "content": "You are a helpful AI assistant."},
        {"role": "user",    "content": "Write a Python function that reverses a linked list."},
    ],
]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test Qwen3-0.6B-diffusion-mdlm-v0.1 locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",          default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1")
    p.add_argument("--steps",          type=int,   default=128,
                   help="Total denoising steps (must be divisible by max_new_tokens/block_size)")
    p.add_argument("--max_new_tokens", type=int,   default=256,
                   help="Tokens to generate per sample (must be multiple of block_size)")
    p.add_argument("--block_size",     type=int,   default=64)
    p.add_argument("--temperature",    type=float, default=0.0,
                   help="0.0 = deterministic; >0 = stochastic Gumbel sampling")
    p.add_argument("--cfg_scale",      type=float, default=0.0,
                   help="Classifier-free guidance scale (0 = disabled)")
    p.add_argument("--remasking",      choices=["low_confidence", "random"],
                   default="low_confidence")
    p.add_argument("--prompt",         type=str,   default=None,
                   help="Single user prompt string (overrides default batch)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    device = select_device()

    gen_cfg = GenerationConfig(
        steps          = args.steps,
        max_new_tokens = args.max_new_tokens,
        block_size     = args.block_size,
        temperature    = args.temperature,
        cfg_scale      = args.cfg_scale,
        remasking      = args.remasking,
    )
    log.info("Generation config: %s", gen_cfg)

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    if args.prompt:
        messages_list = [[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user",   "content": args.prompt},
        ]]
    else:
        messages_list = DEFAULT_MESSAGES

    log.info("Preparing %d prompt(s) …", len(messages_list))
    prompt_tensor, prompt_lens = prepare_batch(messages_list, tokenizer, device)
    log.info("  Prompt tensor shape: %s", tuple(prompt_tensor.shape))

    log.info(
        "Starting generation  (steps=%d, max_new_tokens=%d, block_size=%d) …",
        gen_cfg.steps, gen_cfg.max_new_tokens, gen_cfg.block_size,
    )
    t_gen = time.time()
    output = generate(
        model       = model,
        tokenizer   = tokenizer,
        prompt      = prompt_tensor,
        prompt_lens = prompt_lens,
        pad_id      = pad_id,
        cfg         = gen_cfg,
    )
    elapsed = time.time() - t_gen
    tokens_generated = len(messages_list) * gen_cfg.max_new_tokens
    log.info("Generation done in %.1fs  (%.1f tok/s)", elapsed, tokens_generated / elapsed)

    # ── Decode + display ─────────────────────────────────────────────────────
    decoded = decode_outputs(output, prompt_lens, tokenizer, gen_cfg.max_new_tokens)
    print("\n" + "═" * 72)
    for idx, (msgs, text) in enumerate(zip(messages_list, decoded)):
        user_turn = next(m["content"] for m in msgs if m["role"] == "user")
        print(f"\n[Sample {idx + 1}]  Prompt: {user_turn[:80]!r}")
        print("─" * 72)
        print(text)
    print("\n" + "═" * 72)

    # ── Persist run metadata ──────────────────────────────────────────────────
    RunLogger().log_generation(
        model_name      = args.model,
        config          = gen_cfg,
        elapsed         = elapsed,
        tokens_generated = tokens_generated,
    )


if __name__ == "__main__":
    main()
