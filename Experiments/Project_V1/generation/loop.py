"""
generation/loop.py
==================
MDLM iterative masked-diffusion generation loop.
Imports from: generation.config, generation.noise
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from generation.config import GenerationConfig
from generation.noise import add_gumbel_noise, compute_transfer_schedule


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
    MDLM iterative masked-diffusion generation.

    Algorithm
    ---------
    1. Build [prompt | MASK … MASK] sequence.
    2. Split the generation region into (max_new_tokens / block_size) blocks.
    3. For each block, run `steps_per_block` denoising steps:
         a. Forward pass → logits.
         b. Optional CFG: second unconditional pass, combine linearly.
         c. Gumbel-argmax sample → x_0 predictions.
         d. Score each prediction by confidence (or random).
         e. Unmask top-k confident predictions; re-mask the rest.
    4. Return fully denoised sequence.

    Args:
        model       : Loaded model in eval() mode.
        tokenizer   : Corresponding tokenizer.
        prompt      : LongTensor [batch, max_prompt_len].
        prompt_lens : LongTensor [batch] — actual length of each prompt.
        pad_id      : Token id used for left-padding.
        cfg         : GenerationConfig.

    Returns:
        LongTensor [batch, total_length] — prompt + generated tokens.
    """
    mask_id    = tokenizer.mask_token_id
    device     = model.device
    batch_size = prompt.size(0)

    # ── 1. Build full sequence ────────────────────────────────────────────────
    total_length = int(prompt_lens.max().item() + cfg.max_new_tokens)
    x = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=device)
    for i, length in enumerate(prompt_lens.tolist()):
        x[i, :length] = prompt[i, :length]
        x[i, length : length + cfg.max_new_tokens] = mask_id

    prompt_index = (
        torch.arange(total_length, device=device).unsqueeze(0) < prompt_lens.unsqueeze(1)
    )
    positions = torch.arange(total_length, device=device)

    # ── 2. Validate divisibility ──────────────────────────────────────────────
    assert cfg.max_new_tokens % cfg.block_size == 0, (
        f"max_new_tokens ({cfg.max_new_tokens}) must be divisible by "
        f"block_size ({cfg.block_size})"
    )
    num_blocks      = cfg.max_new_tokens // cfg.block_size
    assert cfg.steps % num_blocks == 0, (
        f"steps ({cfg.steps}) must be divisible by num_blocks ({num_blocks})"
    )
    steps_per_block = cfg.steps // num_blocks

    # ── 3. Semi-autoregressive denoising ─────────────────────────────────────
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

            # Forward (with optional CFG)
            if cfg.cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_in = torch.cat([x, un_x], dim=0)
                logits_both = model(x_in).logits
                logits_cond, logits_uncond = torch.chunk(logits_both, 2, dim=0)
                logits = logits_uncond + (cfg.cfg_scale + 1.0) * (logits_cond - logits_uncond)
            else:
                logits = model(x).logits

            # Gumbel-argmax → x_0 predictions
            x0 = torch.argmax(add_gumbel_noise(logits, cfg.temperature), dim=-1)

            # Confidence scores
            if cfg.remasking == "low_confidence":
                probs = F.softmax(logits, dim=-1)
                x0_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            elif cfg.remasking == "random":
                x0_confidence = torch.rand_like(x0, dtype=torch.float)
            else:
                raise ValueError(f"Unknown remasking strategy: {cfg.remasking!r}")

            # Select top-k to unmask this step
            confidence = torch.where(block_mask, x0_confidence, torch.full_like(x0_confidence, -float("inf")))
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
