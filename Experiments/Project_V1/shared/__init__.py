"""shared — zero-dependency utilities shared across all scripts."""

from shared.device import model_dtype, select_device
from shared.logger import RunLogger, build_logger
from shared.model_io import load_model_and_tokenizer, prepare_batch

__all__ = [
    "select_device",
    "model_dtype",
    "build_logger",
    "RunLogger",
    "load_model_and_tokenizer",
    "prepare_batch",
]
