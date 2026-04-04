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
    python mdlm_qwen3_test.py --steps 64 --max_new_tokens 128 --prompt "Explain MDLM in two sentences."
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    level=logging.INFO,
)
log = logging.getLogger(__name__)


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
# Device selection
# ──────────────────────────────────────────────────────────────────────────────
def select_device() -> torch.device:
    """
    Prefer CUDA > MPS (lite) > CPU.
    MPS does not support bfloat16, so we note that and fall back to float32.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        log.info("Device: MPS (Apple Silicon) — using float32 (bf16 unsupported)")
    else:
        dev = torch.device("cpu")
        log.info("Device: CPU")
    return dev


def model_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 on CUDA; float32 everywhere else."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32


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
    # Gumbel max trick: add -log(-log(u)) which equals dividing by (-log(u))^temp
    gumbel_noise = (-torch.log(u)) ** temperature
    return logits.exp() / gumbel_noise


def compute_transfer_schedule(
    mask_index: torch.BoolTensor,
    steps: int,
) -> torch.LongTensor:
    """
    For each sample in the batch, split the masked positions evenly across
    `steps` denoising steps — "how many tokens to unmask at step i".

    This implements the uniform unmasking schedule: if a block has N masked
    tokens and S steps, each step uncovers floor(N/S) tokens, with the first
    (N mod S) steps each uncovering one extra to handle the remainder exactly.

    Args:
        mask_index : BoolTensor [batch, seq_len] — True where token is masked.
        steps      : Number of denoising iterations for this block.

    Returns:
        LongTensor [batch, steps] — number of tokens to transfer at each step.
    """
    # Total masked count per sample: [batch, 1]
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    schedule = torch.zeros(
        mask_num.size(0), steps,
        device=mask_index.device,
        dtype=torch.long,
    ) + base
    # Distribute remainder: first `remainder` steps get +1
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
        x[i, :length] = prompt[i, :length]          # copy prompt
        x[i, length : length + cfg.max_new_tokens] = mask_id  # mask generation region

    # Boolean mask: True at prompt positions (never overwritten)
    prompt_index = (
        torch.arange(total_length, device=device).unsqueeze(0)
        < prompt_lens.unsqueeze(1)
    )
    positions = torch.arange(total_length, device=device)  # [total_length]

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
        block_start = prompt_lens + block_idx * cfg.block_size   # [batch]
        block_end   = block_start + cfg.block_size               # [batch]

        # Mask for positions inside the current block that are still [MASK]
        init_block_mask = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        # How many tokens to transfer at each step: [batch, steps_per_block]
        transfer_schedule = compute_transfer_schedule(init_block_mask, steps_per_block)

        # ── 3a. Iterative denoising for this block ───────────────────────────
        for step in range(steps_per_block):
            # Recompute live mask (tokens already unmasked in prior steps are gone)
            block_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )

            # ── Forward pass (with optional CFG) ─────────────────────────────
            if cfg.cfg_scale > 0.0:
                # Unconditional copy: replace prompt tokens with [MASK]
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                # Batch conditional + unconditional together for efficiency
                x_in = torch.cat([x, un_x], dim=0)
                logits_both = model(x_in).logits
                logits_cond, logits_uncond = torch.chunk(logits_both, 2, dim=0)
                # CFG combination: push predictions away from unconditional
                logits = logits_uncond + (cfg.cfg_scale + 1.0) * (
                    logits_cond - logits_uncond
                )
            else:
                logits = model(x).logits  # [batch, seq_len, vocab_size]

            # ── Sample x_0 predictions ───────────────────────────────────────
            logits_noisy = add_gumbel_noise(logits, temperature=cfg.temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)  # [batch, seq_len]

            # ── Compute per-position confidence score ─────────────────────────
            if cfg.remasking == "low_confidence":
                # Softmax probability assigned to the predicted token
                probs = F.softmax(logits, dim=-1)
                x0_confidence = torch.gather(
                    probs, dim=-1, index=x0.unsqueeze(-1)
                ).squeeze(-1)  # [batch, seq_len]
            elif cfg.remasking == "random":
                # Uniform random score → equivalent to random unmasking order
                x0_confidence = torch.rand_like(x0, dtype=torch.float)
            else:
                raise ValueError(f"Unknown remasking strategy: {cfg.remasking!r}")

            # ── Select top-k tokens to unmask this step ───────────────────────
            # Positions outside the current block get -inf (never selected)
            confidence = torch.full_like(x0_confidence, -float("inf"))
            confidence = torch.where(block_mask, x0_confidence, confidence)

            # Replace x values at block positions with predictions; keep others
            x0 = torch.where(block_mask, x0, x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for sample_idx in range(batch_size):
                k = int(transfer_schedule[sample_idx, step].item())
                if k == 0:
                    continue
                _, top_indices = torch.topk(confidence[sample_idx], k=k)
                transfer_index[sample_idx, top_indices] = True

            # Commit the selected predictions to x
            x[transfer_index] = x0[transfer_index]

    return x


# ──────────────────────────────────────────────────────────────────────────────
# Model + tokenizer loading
# ──────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """
    Download (or load from HF cache) the MDLM model and tokenizer.

    Notes
    -----
    - trust_remote_code=True is required because dllm-hub uses a custom
      modelling class registered via model_type in config.json.
    - We use the dtype appropriate for the target device (see model_dtype()).
    - .eval() disables dropout; @torch.no_grad() handles gradient tracking.
    """
    dtype = model_dtype(device)
    log.info("Loading tokenizer from %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure a pad token exists (Qwen tokenizers sometimes omit this)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("  pad_token set to eos_token (%r)", tokenizer.eos_token)

    log.info("Loading model in %s precision …", dtype)
    t0 = time.time()
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()
    log.info("  Model loaded in %.1fs  |  params: %.0fM",
             time.time() - t0, sum(p.numel() for p in model.parameters()) / 1e6)

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Prompt preparation
# ──────────────────────────────────────────────────────────────────────────────
def prepare_batch(
    messages_list: list[list[dict]],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Apply the chat template to each conversation and left-pad into a batch.

    enable_thinking=False is strongly recommended by the model card for
    stable, reproducible outputs; thinking mode is an autoregressive feature
    that doesn't integrate cleanly with the diffusion generation loop.

    Returns
    -------
    prompt_tensor : LongTensor [batch, max_prompt_len] — right-aligned prompts.
    prompt_lens   : LongTensor [batch]                 — actual length of each.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    token_ids_list = [
        tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,   # keep diffusion loop stable
        )
        for msgs in messages_list
    ]

    prompt_lens = torch.tensor([len(ids) for ids in token_ids_list], dtype=torch.long)
    max_len = int(prompt_lens.max().item())

    # Left-pad (pad on the left so the model sees prompt at end)
    prompt_tensor = torch.full((len(token_ids_list), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(token_ids_list):
        prompt_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    return prompt_tensor.to(device), prompt_lens.to(device)


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

    # ── Device ───────────────────────────────────────────────────────────────
    device = select_device()

    # ── Generation config ────────────────────────────────────────────────────
    gen_cfg = GenerationConfig(
        steps          = args.steps,
        max_new_tokens = args.max_new_tokens,
        block_size     = args.block_size,
        temperature    = args.temperature,
        cfg_scale      = args.cfg_scale,
        remasking      = args.remasking,
    )
    log.info("Generation config: %s", gen_cfg)

    # ── Load model + tokenizer ───────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ── Prepare prompts ──────────────────────────────────────────────────────
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

    # ── Generate ─────────────────────────────────────────────────────────────
    log.info("Starting generation  (steps=%d, max_new_tokens=%d, block_size=%d) …",
             gen_cfg.steps, gen_cfg.max_new_tokens, gen_cfg.block_size)
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


if __name__ == "__main__":
    main()
