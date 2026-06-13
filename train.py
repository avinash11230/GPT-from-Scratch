"""
Training loop for the from-scratch GPT model on a character-level
dataset (Tiny Shakespeare by default). Saves a checkpoint and the
tokenizer vocabulary when finished.

Usage:
    python train.py --steps 2000
"""

import argparse
import os
import torch

from model import GPT
from tokenizer import CharTokenizer


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/input.txt")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(args.data, "r") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(args.steps):
        if step % args.eval_interval == 0 or step == args.steps - 1:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, device)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "block_size": args.block_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
        },
    }, os.path.join(args.out_dir, "model.pt"))
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))
    print(f"Saved checkpoint to {args.out_dir}/")


if __name__ == "__main__":
    main()
