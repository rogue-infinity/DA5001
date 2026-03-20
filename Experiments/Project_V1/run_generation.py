"""
run_generation.py
=================
CLI for MDLM text generation.

Usage
-----
    python run_generation.py
    python run_generation.py --steps 64 --max_new_tokens 128 --prompt "Explain MDLM."
"""

import argparse
import time

from generation import GenerationConfig, DEFAULT_MESSAGES, generate, decode_outputs
from shared import select_device, load_model_and_tokenizer, prepare_batch, build_logger, RunLogger

log = build_logger("run_generation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MDLM generation harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",          default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1")
    p.add_argument("--steps",          type=int,   default=128,
                   help="Denoising steps (must divide max_new_tokens/block_size)")
    p.add_argument("--max_new_tokens", type=int,   default=256,
                   help="Tokens to generate (must be multiple of block_size)")
    p.add_argument("--block_size",     type=int,   default=64)
    p.add_argument("--temperature",    type=float, default=0.0,
                   help="0.0 = deterministic; >0 = stochastic Gumbel")
    p.add_argument("--cfg_scale",      type=float, default=0.0,
                   help="Classifier-free guidance scale (0 = disabled)")
    p.add_argument("--remasking",      choices=["low_confidence", "random"],
                   default="low_confidence")
    p.add_argument("--prompt",         type=str,   default=None,
                   help="Single user prompt (overrides default batch)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()

    gen_cfg = GenerationConfig(
        steps          = args.steps,
        max_new_tokens = args.max_new_tokens,
        block_size     = args.block_size,
        temperature    = args.temperature,
        cfg_scale      = args.cfg_scale,
        remasking      = args.remasking,
    )
    log.info("Generation config: %s", gen_cfg)

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    messages_list = (
        [[{"role": "system", "content": "You are a helpful AI assistant."},
          {"role": "user",   "content": args.prompt}]]
        if args.prompt else DEFAULT_MESSAGES
    )

    log.info("Preparing %d prompt(s) …", len(messages_list))
    prompt_tensor, prompt_lens = prepare_batch(messages_list, tokenizer, device)

    log.info("Generating (steps=%d, max_new_tokens=%d, block_size=%d) …",
             gen_cfg.steps, gen_cfg.max_new_tokens, gen_cfg.block_size)
    t0 = time.time()
    output = generate(model, tokenizer, prompt_tensor, prompt_lens, pad_id, gen_cfg)
    elapsed = time.time() - t0
    tokens_generated = len(messages_list) * gen_cfg.max_new_tokens

    log.info("Done in %.1fs  (%.1f tok/s)", elapsed, tokens_generated / elapsed)

    decoded = decode_outputs(output, prompt_lens, tokenizer, gen_cfg.max_new_tokens)

    print("\n" + "═" * 72)
    for idx, (msgs, text) in enumerate(zip(messages_list, decoded)):
        user_turn = next(m["content"] for m in msgs if m["role"] == "user")
        print(f"\n[Sample {idx + 1}]  Prompt: {user_turn[:80]!r}")
        print("─" * 72)
        print(text)
    print("\n" + "═" * 72)

    RunLogger().log_generation(
        model_name      = args.model,
        config          = gen_cfg,
        elapsed         = elapsed,
        tokens_generated = tokens_generated,
    )


if __name__ == "__main__":
    main()
