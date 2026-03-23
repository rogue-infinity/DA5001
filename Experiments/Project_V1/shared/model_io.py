"""
shared/model_io.py
==================
Model + tokenizer loading and prompt batch preparation.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from shared.device import model_dtype
from shared.logger import build_logger

_log = build_logger("shared.model_io")


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    attn_implementation: Optional[str] = None,
) -> tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """
    Load (or pull from HF cache) the MDLM model and tokenizer.

    Args:
        model_name          : HuggingFace model identifier.
        device              : Target torch.device.
        attn_implementation : Optional attention backend override.
                              Pass "eager" to expose attention weights
                              via forward hooks (needed by metrics extractor).

    Notes
    -----
    trust_remote_code=True is required for custom modelling classes registered
    via model_type in config.json (dllm-hub models).
    """
    dtype = model_dtype(device)
    _log.info("Loading tokenizer from %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        _log.info("  pad_token set to eos_token (%r)", tokenizer.eos_token)

    kwargs: dict = dict(dtype=dtype, trust_remote_code=True)
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    suffix = f" (attn={attn_implementation})" if attn_implementation else ""
    _log.info("Loading model in %s precision%s …", dtype, suffix)
    t0 = time.time()
    model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    _log.info("  Loaded in %.1fs  |  params: %.0fM", time.time() - t0, n_params)

    return model, tokenizer


def prepare_batch(
    messages_list: list[list[dict]],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Apply the chat template to each conversation and left-pad into a batch.

    enable_thinking=False keeps the diffusion loop stable; the thinking mode
    is an autoregressive feature that does not integrate cleanly with MDLM.

    Returns
    -------
    prompt_tensor : LongTensor [batch, max_prompt_len]
    prompt_lens   : LongTensor [batch]
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    token_ids_list = [
        tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        for msgs in messages_list
    ]

    prompt_lens = torch.tensor([len(ids) for ids in token_ids_list], dtype=torch.long)
    max_len = int(prompt_lens.max().item())

    prompt_tensor = torch.full((len(token_ids_list), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(token_ids_list):
        prompt_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    return prompt_tensor.to(device), prompt_lens.to(device)
