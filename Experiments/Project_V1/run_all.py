"""
run_all.py
==========
Single entry-point that runs both generation and metrics extraction.

Workflow
--------
1. Generate text from a prompt using the MDLM denoising loop.
2. For each generated sample, extract all MIA metrics (ELBO, entropy,
   consistency, attention, hidden states, gradient norms).
3. Print generated text and metrics summary to stdout.
4. Log everything to logs/runs.jsonl.

Usage
-----
    python run_all.py
    python run_all.py --prompt "Explain MDLM in two sentences." \\
                      --steps 64 --max_new_tokens 128 \\
                      --timesteps 10 --mask_configs 4 --save bundles/
"""

import argparse
import time
from pathlib import Path

import torch

from generation import GenerationConfig, DEFAULT_MESSAGES, generate, decode_outputs
from metrics import print_summary
from run_extraction import extract_metrics
from shared import (
    RunLogger,
    build_logger,
    load_model_and_tokenizer,
    prepare_batch,
    select_device,
)

log = build_logger("run_all")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MDLM: generate text then extract MIA metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model",          default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1")

    # ── Generation ────────────────────────────────────────────────────────────
    p.add_argument("--prompt",         type=str,   default=None,
                   help="Single user prompt. Uses default batch if not set.")
    p.add_argument("--steps",          type=int,   default=128)
    p.add_argument("--max_new_tokens", type=int,   default=256)
    p.add_argument("--block_size",     type=int,   default=64)
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--cfg_scale",      type=float, default=0.0)
    p.add_argument("--remasking",      choices=["low_confidence", "random"],
                   default="low_confidence")

    # ── Extraction ────────────────────────────────────────────────────────────
    p.add_argument("--timesteps",      type=int,   default=20)
    p.add_argument("--mask_configs",   type=int,   default=8)
    p.add_argument("--grad_timesteps", type=int,   default=4)
    p.add_argument("--no_attentions",  action="store_true",
                   help="Skip attention capture during extraction.")
    p.add_argument("--seed",           type=int,   default=42)

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--save",           type=str,   default=None,
                   help="Directory to save MetricsBundle .pt files. "
                        "Created if it does not exist.")
    p.add_argument("--debug",          action="store_true")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import logging
    args = parse_args()
    if args.debug:
        log.setLevel(logging.DEBUG)

    device = select_device()
    run_logger = RunLogger()

    gen_cfg = GenerationConfig(
        steps          = args.steps,
        max_new_tokens = args.max_new_tokens,
        block_size     = args.block_size,
        temperature    = args.temperature,
        cfg_scale      = args.cfg_scale,
        remasking      = args.remasking,
    )

    # Load model once — use eager attention if we need attention maps
    attn_impl = "eager" if not args.no_attentions else None
    model, tokenizer = load_model_and_tokenizer(args.model, device, attn_impl)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ── Step 1: Generation ────────────────────────────────────────────────────
    messages_list = (
        [[{"role": "system", "content": "You are a helpful AI assistant."},
          {"role": "user",   "content": args.prompt}]]
        if args.prompt else DEFAULT_MESSAGES
    )

    log.info("═" * 60)
    log.info("STEP 1 — Generation")
    log.info("═" * 60)
    log.info("Config: %s", gen_cfg)

    prompt_tensor, prompt_lens = prepare_batch(messages_list, tokenizer, device)
    t_gen = time.time()
    output = generate(model, tokenizer, prompt_tensor, prompt_lens, pad_id, gen_cfg)
    elapsed_gen = time.time() - t_gen
    tokens_generated = len(messages_list) * gen_cfg.max_new_tokens

    log.info("Generation done in %.1fs  (%.1f tok/s)",
             elapsed_gen, tokens_generated / elapsed_gen)

    decoded = decode_outputs(output, prompt_lens, tokenizer, gen_cfg.max_new_tokens)

    print("\n" + "═" * 72)
    for idx, (msgs, text) in enumerate(zip(messages_list, decoded)):
        user_turn = next(m["content"] for m in msgs if m["role"] == "user")
        print(f"\n[Sample {idx + 1}]  Prompt: {user_turn[:80]!r}")
        print("─" * 72)
        print(text)
    print("\n" + "═" * 72)

    run_logger.log_generation(
        model_name      = args.model,
        config          = gen_cfg,
        elapsed         = elapsed_gen,
        tokens_generated = tokens_generated,
    )

    # ── Step 2: Extract metrics for each generated sample ─────────────────────
    log.info("═" * 60)
    log.info("STEP 2 — Metrics Extraction (%d sample(s))", len(decoded))
    log.info("═" * 60)

    save_dir = Path(args.save) if args.save else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for idx, text in enumerate(decoded):
        log.info("── Sample %d/%d ──", idx + 1, len(decoded))

        bundle = extract_metrics(
            model              = model,
            tokenizer          = tokenizer,
            text               = text,
            n_timesteps        = args.timesteps,
            n_mask_configs     = args.mask_configs,
            grad_timesteps     = args.grad_timesteps,
            capture_attentions = not args.no_attentions,
            seed               = args.seed,
        )

        print_summary(bundle)

        save_path = None
        if save_dir is not None:
            save_path = save_dir / f"bundle_sample{idx + 1}.pt"
            torch.save(bundle, save_path)
            log.info("Bundle saved → %s", save_path)

        run_logger.log_extraction(
            model_name     = args.model,
            text           = text,
            n_timesteps    = args.timesteps,
            n_mask_configs = args.mask_configs,
            grad_timesteps = args.grad_timesteps,
            bundle         = bundle,
            save_path      = save_path,
        )

    log.info("All done. Logs → logs/runs.jsonl")


if __name__ == "__main__":
    main()
