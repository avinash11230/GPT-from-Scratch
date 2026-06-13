"""
Training loop for the GPT from-scratch model.

Supports:
* YAML-based config system (:mod:`config`) with CLI overrides
* Character-level or BPE tokenizer selected via ``--tokenizer``
* Cosine LR schedule with linear warmup
* Gradient clipping and gradient accumulation
* Mixed-precision training (autocast + GradScaler) with CPU fallback
* Checkpoint saving every N steps + resume from checkpoint
* TensorBoard logging of loss, perplexity, learning rate, and text samples
* Periodic text-sample generation to monitor qualitative progress

Usage
-----
Quick start (defaults to 'small' config, character tokenizer)::

    python train.py

Use a named config preset::

    python train.py --config configs/small.yaml

Override individual fields::

    python train.py --config configs/small.yaml --steps 2000 --tokenizer bpe

Resume from a checkpoint::

    python train.py --config configs/small.yaml --resume checkpoints/ckpt_step2000.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import ModelConfig, TrainConfig, load_config, get_lr, config_to_dict
from model import GPT
from tokenizer import get_tokenizer, CharTokenizer


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences.

    Args:
        data:       1-D token-id tensor of the full split.
        block_size: Context length.
        batch_size: Number of sequences.
        device:     Target device.

    Returns:
        ``(x, y)`` tensors of shape ``(batch_size, block_size)``.
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_iters: int = 50,
) -> dict[str, float]:
    """Estimate mean train and val loss over *eval_iters* random batches.

    Args:
        model:      The GPT model (in eval mode during the call).
        train_data: Training split tensor.
        val_data:   Validation split tensor.
        block_size: Context length.
        batch_size: Batch size for eval.
        device:     Compute device.
        eval_iters: Number of batches to average.

    Returns:
        Dict with keys ``"train"`` and ``"val"`` mapping to mean cross-entropy loss.
    """
    model.eval()
    results: dict[str, float] = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        results[name] = losses.mean().item()
    model.train()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.  CLI values override YAML config fields."""
    p = argparse.ArgumentParser(
        description="Train GPT from scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="", help="Path to YAML config file")

    # Train overrides
    p.add_argument("--data", type=str, default=None, help="Path to training text file")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--tokenizer", type=str, choices=["char", "bpe"], default=None)
    p.add_argument("--bpe_vocab_size", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--accum_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")

    # Model overrides
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)

    return p.parse_args()


def apply_overrides(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    args: argparse.Namespace,
) -> None:
    """Apply non-None CLI args on top of loaded config in-place."""
    override_map = {
        # train
        "data": ("data_path", train_cfg),
        "out_dir": ("out_dir", train_cfg),
        "tokenizer": ("tokenizer", train_cfg),
        "bpe_vocab_size": ("bpe_vocab_size", train_cfg),
        "steps": ("steps", train_cfg),
        "batch_size": ("batch_size", train_cfg),
        "accum_steps": ("accum_steps", train_cfg),
        "lr": ("lr", train_cfg),
        "warmup_steps": ("warmup_steps", train_cfg),
        "grad_clip": ("grad_clip", train_cfg),
        "eval_interval": ("eval_interval", train_cfg),
        "resume": ("resume", train_cfg),
        # model
        "block_size": ("block_size", model_cfg),
        "n_embd": ("n_embd", model_cfg),
        "n_head": ("n_head", model_cfg),
        "n_layer": ("n_layer", model_cfg),
        "dropout": ("dropout", model_cfg),
    }
    for arg_name, (field, cfg_obj) in override_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(cfg_obj, field, val)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    """Run the full training loop.

    Args:
        model_cfg: Model architecture hyperparameters.
        train_cfg: Training loop hyperparameters.
    """
    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────
    with open(train_cfg.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = get_tokenizer(
        kind=train_cfg.tokenizer,
        text=text,
        vocab_size=train_cfg.bpe_vocab_size,
    )
    model_cfg.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: {train_cfg.tokenizer}, vocab_size={tokenizer.vocab_size}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Dataset: {len(data):,} tokens  |  train {len(train_data):,}  |  val {len(val_data):,}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = GPT(model_cfg).to(device)
    n_params = model.num_params()
    print(f"Model: {n_params / 1e6:.3f}M parameters")

    # ── Optimizer & AMP ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=(0.9, 0.95))

    use_amp = train_cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    if train_cfg.use_amp and not use_amp:
        print("Note: AMP requested but running on CPU - skipping (not supported).")

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_step = 0
    if train_cfg.resume:
        print(f"Resuming from {train_cfg.resume} ...")
        ckpt = torch.load(train_cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt.get("step", 0) + 1
        print(f"  -> resuming from step {start_step}")

    # ── TensorBoard ────────────────────────────────────────────────────────
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=train_cfg.log_dir)

    # ── Directories ────────────────────────────────────────────────────────
    os.makedirs(train_cfg.out_dir, exist_ok=True)

    # ── Encode sample prompt for periodic generation ───────────────────────
    sample_ids = torch.tensor(
        [tokenizer.encode(train_cfg.sample_prompt)], dtype=torch.long, device=device
    )

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for step in range(start_step, train_cfg.steps):
        # ── LR update ─────────────────────────────────────────────────────
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation loop ────────────────────────────────────
        loss_accum = 0.0
        for micro_step in range(train_cfg.accum_steps):
            x, y = get_batch(train_data, model_cfg.block_size, train_cfg.batch_size, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / train_cfg.accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # ── Gradient clipping ─────────────────────────────────────────────
        if train_cfg.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ── Logging (loss only, every step) ───────────────────────────────
        writer.add_scalar("train/loss_step", loss_accum * train_cfg.accum_steps, step)
        writer.add_scalar("train/lr", lr, step)

        # ── Evaluation ────────────────────────────────────────────────────
        if step % train_cfg.eval_interval == 0 or step == train_cfg.steps - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                model_cfg.block_size, train_cfg.batch_size, device,
                train_cfg.eval_iters,
            )
            val_ppl = math.exp(min(losses["val"], 20))
            train_ppl = math.exp(min(losses["train"], 20))
            elapsed = time.time() - t0

            print(
                f"step {step:5d} | train loss {losses['train']:.4f} (ppl {train_ppl:.1f})"
                f" | val loss {losses['val']:.4f} (ppl {val_ppl:.1f})"
                f" | lr {lr:.2e} | {elapsed:.1f}s"
            )

            writer.add_scalar("loss/train", losses["train"], step)
            writer.add_scalar("loss/val", losses["val"], step)
            writer.add_scalar("perplexity/train", train_ppl, step)
            writer.add_scalar("perplexity/val", val_ppl, step)

        # ── Periodic text sample ──────────────────────────────────────────
        if step % train_cfg.sample_interval == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                gen_ids = model.generate(
                    sample_ids,
                    max_new_tokens=train_cfg.sample_tokens,
                    temperature=0.8,
                    top_k=50,
                )
            sample_text = tokenizer.decode(gen_ids[0].tolist())
            writer.add_text("samples/generated", sample_text, step)
            print(f"\n-- Sample at step {step} --\n{sample_text}\n" + "-" * 40)
            model.train()

        # ── Intermediate checkpoint ───────────────────────────────────────
        if train_cfg.save_interval > 0 and step > 0 and step % train_cfg.save_interval == 0:
            _save_checkpoint(model, optimizer, tokenizer, model_cfg, train_cfg, step)

    # ── Final checkpoint ───────────────────────────────────────────────────
    _save_checkpoint(model, optimizer, tokenizer, model_cfg, train_cfg, step=train_cfg.steps - 1, final=True)
    writer.close()
    print("Training complete.")


def _save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    step: int,
    final: bool = False,
) -> None:
    """Serialise model, optimizer, and metadata to a checkpoint file.

    Args:
        model:      Trained model.
        optimizer:  Optimizer (state saved for resuming).
        tokenizer:  Fitted tokenizer (saved alongside the checkpoint).
        model_cfg:  Model hyperparameters.
        train_cfg:  Training hyperparameters.
        step:       Current training step.
        final:      If True, also saves as ``model.pt`` (the canonical checkpoint).
    """
    fname = f"ckpt_step{step}.pt" if not final else "model.pt"
    path = os.path.join(train_cfg.out_dir, fname)
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "tokenizer_kind": train_cfg.tokenizer,
    }
    torch.save(payload, path)
    tok_path = os.path.join(train_cfg.out_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"  * Checkpoint saved -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.config:
        model_cfg, train_cfg = load_config(args.config)
    else:
        model_cfg, train_cfg = ModelConfig(), TrainConfig()

    apply_overrides(model_cfg, train_cfg, args)

    print("=" * 60)
    print(f"  GPT From Scratch - training run")
    print(f"  Config:    {args.config or 'defaults'}")
    print(f"  Model:     {model_cfg.n_layer}L / {model_cfg.n_embd}d / {model_cfg.n_head}H")
    print(f"  Steps:     {train_cfg.steps}")
    print(f"  Tokenizer: {train_cfg.tokenizer}")
    print("=" * 60)

    train(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
