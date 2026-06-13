"""
Generate text from a trained GPT checkpoint.

Supports top-k and nucleus (top-p) sampling, optional KV-cache for fast
inference, and automatic tokenizer detection from the checkpoint metadata.

Usage
-----
::

    python generate.py --prompt "ROMEO:" --max_new_tokens 300

    # Use nucleus sampling
    python generate.py --prompt "ROMEO:" --top_p 0.9

    # Benchmark KV-cache speed
    python generate.py --max_new_tokens 200
    python generate.py --max_new_tokens 200 --no_kv_cache

    # Use BPE tokenizer checkpoint
    python generate.py --ckpt_dir checkpoints --tokenizer bpe
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from config import ModelConfig
from model import GPT
from tokenizer import get_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate text from a trained GPT checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt_dir", type=str, default="checkpoints",
                   help="Directory containing model.pt and tokenizer.json")
    p.add_argument("--ckpt_file", type=str, default="model.pt",
                   help="Checkpoint filename within ckpt_dir")
    p.add_argument("--prompt", type=str, default="\n",
                   help="Seed text for generation")
    p.add_argument("--max_new_tokens", type=int, default=300,
                   help="Number of new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Softmax temperature (higher = more random)")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-k sampling (0 = disabled)")
    p.add_argument("--top_p", type=float, default=None,
                   help="Nucleus (top-p) sampling threshold (None = disabled)")
    p.add_argument("--no_kv_cache", action="store_true",
                   help="Disable KV cache (for benchmarking)")
    p.add_argument("--tokenizer", type=str, choices=["char", "bpe"], default=None,
                   help="Override tokenizer kind (auto-detected from checkpoint by default)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt_path = f"{args.ckpt_dir}/{args.ckpt_file}"
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Reconstruct model config ───────────────────────────────────────────
    raw_cfg = checkpoint.get("model_config", checkpoint.get("config", {}))
    model_cfg = ModelConfig(**{k: v for k, v in raw_cfg.items() if k in ModelConfig.__dataclass_fields__})

    # ── Load tokenizer ─────────────────────────────────────────────────────
    tok_kind = args.tokenizer or checkpoint.get("tokenizer_kind", "char")
    tokenizer = get_tokenizer(
        kind=tok_kind,
        tokenizer_path=f"{args.ckpt_dir}/tokenizer.json",
    )
    print(f"Tokenizer: {tok_kind}, vocab_size={tokenizer.vocab_size}")

    # ── Build and load model ───────────────────────────────────────────────
    model = GPT(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model: {model.num_params() / 1e6:.3f}M parameters")

    # ── Encode prompt ──────────────────────────────────────────────────────
    prompt_ids = tokenizer.encode(args.prompt)
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    top_k = args.top_k if args.top_k > 0 else None
    use_kv_cache = not args.no_kv_cache
    cache_label = "KV-cache ON" if use_kv_cache else "KV-cache OFF"
    print(f"Generating {args.max_new_tokens} tokens ... ({cache_label})\n")

    # ── Generate ───────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            top_p=args.top_p,
            use_kv_cache=use_kv_cache,
        )
    elapsed = time.perf_counter() - t_start

    # ── Decode and print ───────────────────────────────────────────────────
    generated_ids = output[0].tolist()
    text = tokenizer.decode(generated_ids)
    print(text)

    new_tokens = len(generated_ids) - len(prompt_ids)
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else float("inf")
    print(f"\n-- {new_tokens} tokens in {elapsed:.2f}s  ({tok_per_sec:.1f} tok/s)  [{cache_label}] --")


if __name__ == "__main__":
    main()
