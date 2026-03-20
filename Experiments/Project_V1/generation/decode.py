"""
generation/decode.py
====================
Output decoding and default test prompts.
No project-level imports.
"""

import torch
from transformers import AutoTokenizer


DEFAULT_MESSAGES: list[list[dict]] = [
    [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user",   "content": "What is masked diffusion language modeling in two sentences?"},
    ],
    [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user",   "content": "Write a Python function that reverses a linked list."},
    ],
]


def decode_outputs(
    generated: torch.LongTensor,
    prompt_lens: torch.LongTensor,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> list[str]:
    """
    Slice the generated region [prompt_len : prompt_len + max_new_tokens]
    from each sample and decode to text.  Special tokens are preserved for
    inspection (skip_special_tokens=False).
    """
    decoded = []
    for i in range(generated.size(0)):
        start  = int(prompt_lens[i].item())
        end    = start + max_new_tokens
        tokens = generated[i, start:end].tolist()
        decoded.append(tokenizer.decode(tokens, skip_special_tokens=False))
    return decoded
