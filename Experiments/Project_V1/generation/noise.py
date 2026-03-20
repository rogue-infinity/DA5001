"""
generation/noise.py
===================
Gumbel noise injection and token-transfer schedule computation.
No project-level imports.
"""

import torch


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Inject Gumbel-max noise into logits for stochastic sampling.

    The MDLM paper samples x_0 predictions via:
        logits_noisy = exp(logits) / (-log(U))^temperature,   U ~ Uniform(0,1)
    At temperature=0 this reduces to deterministic argmax.

    Args:
        logits      : Raw model logits [..., vocab_size].
        temperature : Noise strength. 0.0 = no noise.

    Returns:
        Noisy logits, cast to float64 for numerical stability.
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
    For each sample in the batch, split masked positions evenly across
    `steps` denoising iterations.

    Implements the uniform unmasking schedule: if a block has N masked
    tokens and S steps, each step uncovers floor(N/S) tokens, with the
    first (N mod S) steps each uncovering one extra.

    Args:
        mask_index : BoolTensor [batch, seq_len] — True at masked positions.
        steps      : Number of denoising iterations for this block.

    Returns:
        LongTensor [batch, steps] — tokens to unmask at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)   # [batch, 1]
    base = mask_num // steps
    remainder = mask_num % steps
    schedule = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long)
        + base
    )
    for i in range(mask_num.size(0)):
        schedule[i, : remainder[i]] += 1
    return schedule
