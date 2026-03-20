"""
shared/device.py
================
Device selection and dtype resolution.
No project-level imports.
"""

import torch


def select_device() -> torch.device:
    """
    Prefer CUDA > MPS > CPU.
    MPS does not support bfloat16, so float32 is used there.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 on CUDA; float32 everywhere else."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32
