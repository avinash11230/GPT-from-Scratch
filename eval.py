"""
Evaluate a trained GPT checkpoint on the full train and validation splits.

Unlike the quick estimate done during training (which averages over 50 random
batches), this script processes the full dataset in a sliding-window fashion
to give a more accurate final loss and perplexity number.

Usage
-----
::

    python eval.py --ckpt_dir checkpoints
    python eval.py --ckpt_dir checkpoints --ckpt_file ckpt_step2000.pt
"""

from __future__ import annotations

import argparse
import math

import torch

from config import ModelConfig
from model import GPT
from tokenizer import get_tokenizer


def evaluate_split(
    model: GPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Compute exact cross-entropy loss over an entire data split.

    Slides a context window across the full token sequence without overlap,
    accumulating the average token-level loss.

    Args:
        model:      Model in eval mode.
        data:       1-D token tensor for the split.
        block_size: Context length (matches the model's block_size).
        batch_size: Number of windows to process in parallel.
        device:     Compute device.

    Returns:
        Mean cross-entropy loss across all complete windows.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for start in range(0, len(data) - block_size - 1, block_size * batch_size):
            batch_starts = range(start, min(start + block_size * batch_size, len(data) - block_size - 1), block_size)
            if not batch_starts:
                break
            x = torch.stack([data[i : i + block_size] for i in batch_starts]).to(device)
            y = torch.stack([data[i + 1 : i + 1 + block_size] for i in batch_starts]).to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on train and val splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--ckpt_file", type=str, default="model.pt")
    p.add_argument("--data", type=str, default="data/input.txt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--tokenizer", type=str, choices=["char", "bpe"], default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{args.ckpt_dir}/{args.ckpt_file}"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    raw_cfg = checkpoint.get("model_config", checkpoint.get("config", {}))
    model_cfg = ModelConfig(**{k: v for k, v in raw_cfg.items() if k in ModelConfig.__dataclass_fields__})

    tok_kind = args.tokenizer or checkpoint.get("tokenizer_kind", "char")
    tokenizer = get_tokenizer(kind=tok_kind, tokenizer_path=f"{args.ckpt_dir}/tokenizer.json")

    model = GPT(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded: {ckpt_path}  |  {model.num_params() / 1e6:.3f}M params")

    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    print(f"\nEvaluating over {len(train_data):,} train tokens and {len(val_data):,} val tokens ...\n")

    train_loss = evaluate_split(model, train_data, model_cfg.block_size, args.batch_size, device)
    val_loss = evaluate_split(model, val_data, model_cfg.block_size, args.batch_size, device)

    print(f"{'Split':<8} {'Loss':>8}  {'Perplexity':>12}")
    print("-" * 32)
    print(f"{'train':<8} {train_loss:>8.4f}  {math.exp(train_loss):>12.2f}")
    print(f"{'val':<8} {val_loss:>8.4f}  {math.exp(val_loss):>12.2f}")


if __name__ == "__main__":
    main()
