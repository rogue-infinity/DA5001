"""metrics — one file per MIA signal family."""

from metrics.attention import attention_mean, stack_attention_maps
from metrics.bundle import MetricsBundle, print_summary
from metrics.consistency import mask_consistency
from metrics.elbo import compute_elbo_derivatives, compute_elbo_loss
from metrics.entropy import aggregate_entropy, token_entropy
from metrics.forward import forward_with_hooks, patch_model_for_outputs
from metrics.gradients import gradient_norm
from metrics.hidden import compute_baseline_dirs, cosine_sim_to_baseline, layer_norms
from metrics.masking import apply_masking

__all__ = [
    # bundle
    "MetricsBundle",
    "print_summary",
    # forward infrastructure
    "forward_with_hooks",
    "patch_model_for_outputs",
    "apply_masking",
    # per-signal
    "compute_elbo_loss",
    "compute_elbo_derivatives",
    "token_entropy",
    "aggregate_entropy",
    "mask_consistency",
    "attention_mean",
    "stack_attention_maps",
    "layer_norms",
    "compute_baseline_dirs",
    "cosine_sim_to_baseline",
    "gradient_norm",
]
