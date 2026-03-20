"""
generation/config.py
====================
GenerationConfig — all knobs for the MDLM iterative denoising loop.
No project-level imports.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class GenerationConfig:
    """
    steps          : Total denoising iterations. Must be divisible by
                     (max_new_tokens / block_size).
    max_new_tokens : Tokens to generate per sample. Must be a multiple
                     of block_size.
    block_size     : Semi-autoregressive block size. Tokens are generated
                     left-to-right in chunks; each chunk is fully denoised
                     before the next begins. 64 is the model default.
    temperature    : Gumbel noise temperature. 0.0 = deterministic argmax;
                     >0 = stochastic.
    cfg_scale      : Classifier-free guidance scale. 0.0 = disabled.
                     >0 doubles the forward pass cost per step.
    remasking      : "low_confidence" keeps the top-k most confident
                     predictions; "random" keeps k chosen uniformly.
    """
    steps: int = 128
    max_new_tokens: int = 256
    block_size: int = 64
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: Literal["low_confidence", "random"] = "low_confidence"
