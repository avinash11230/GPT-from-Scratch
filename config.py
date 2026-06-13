"""
Configuration system for the GPT from-scratch project.

All hyperparameters live in two dataclasses (:class:`ModelConfig` and
:class:`TrainConfig`).  A YAML file can be loaded with :func:`load_config`
and individual fields can be overridden from the command line afterwards.

This makes experiments reproducible — every run can be fully described by
a single YAML file plus optional CLI overrides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional
import os

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Hyperparameters that define the model architecture.

    Attributes:
        vocab_size:  Number of tokens in the vocabulary.  Set automatically
                     after tokenizer training — you rarely need to set this
                     manually in a config file.
        block_size:  Maximum context length (sequence length).
        n_embd:      Embedding / hidden dimension.
        n_head:      Number of attention heads.  Must divide *n_embd* evenly.
        n_layer:     Number of transformer blocks.
        dropout:     Dropout probability applied to attention weights and
                     residual connections.
        weight_tying: If True, tie the token embedding and LM-head weights,
                      which reduces parameter count and often improves quality.
    """

    vocab_size: int = 65          # filled in at runtime after tokenizer build
    block_size: int = 256
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    weight_tying: bool = True


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Hyperparameters for the training loop.

    Attributes:
        data_path:       Path to the raw text dataset.
        out_dir:         Directory where checkpoints and the tokenizer are saved.
        tokenizer:       ``"char"`` or ``"bpe"``.
        bpe_vocab_size:  Target vocabulary size when using the BPE tokenizer.
        steps:           Total number of optimisation steps.
        batch_size:      Number of sequences per micro-batch.
        accum_steps:     Gradient accumulation steps.  The effective batch size
                         is ``batch_size × accum_steps``.
        lr:              Peak learning rate for the cosine schedule.
        min_lr:          Minimum learning rate at the end of the cosine decay.
        warmup_steps:    Number of steps for the linear LR warmup.
        grad_clip:       Max gradient norm (0.0 = disabled).
        eval_interval:   How often (in steps) to run validation.
        eval_iters:      Number of batches to average for each loss estimate.
        save_interval:   How often (in steps) to save an intermediate checkpoint.
        sample_interval: How often (in steps) to log a generated text sample.
        sample_prompt:   Seed text for the periodic sample.
        sample_tokens:   Number of tokens to generate for each logged sample.
        use_amp:         Use automatic mixed precision (autocast + GradScaler).
        log_dir:         TensorBoard log directory.
        resume:          Path to a checkpoint to resume from.  Empty = start fresh.
        seed:            Random seed for reproducibility.
    """

    data_path: str = "data/input.txt"
    out_dir: str = "checkpoints"
    tokenizer: str = "char"
    bpe_vocab_size: int = 2048
    steps: int = 5000
    batch_size: int = 64
    accum_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 50
    save_interval: int = 1000
    sample_interval: int = 500
    sample_prompt: str = "\n"
    sample_tokens: int = 200
    use_amp: bool = True
    log_dir: str = "runs"
    resume: str = ""
    seed: int = 42


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> tuple[ModelConfig, TrainConfig]:
    """Load a YAML config file and return ``(ModelConfig, TrainConfig)``.

    The YAML file can contain keys from either dataclass under top-level keys
    ``model`` and ``train``.  Missing keys fall back to dataclass defaults.

    Args:
        path: Path to the YAML config file.

    Returns:
        A ``(ModelConfig, TrainConfig)`` tuple.

    Example YAML::

        model:
          n_layer: 6
          n_embd:  256
          n_head:  8
          block_size: 512
        train:
          steps: 10000
          tokenizer: bpe
    """
    if not _YAML_AVAILABLE:
        raise ImportError(
            "pyyaml is required to load YAML configs.  Run: pip install pyyaml"
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model_data = data.get("model", {})
    train_data = data.get("train", {})

    model_cfg = ModelConfig(**{k: v for k, v in model_data.items() if k in ModelConfig.__dataclass_fields__})
    train_cfg = TrainConfig(**{k: v for k, v in train_data.items() if k in TrainConfig.__dataclass_fields__})
    return model_cfg, train_cfg


def config_to_dict(model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:
    """Serialise both configs to a plain dict for checkpoint storage."""
    return {"model": asdict(model_cfg), "train": asdict(train_cfg)}


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine learning rate schedule with linear warmup.

    - Steps 0 … ``warmup_steps``:  linear ramp from 0 → ``lr``.
    - Steps ``warmup_steps`` … ``steps``:  cosine decay from ``lr`` → ``min_lr``.

    Args:
        step: Current training step (0-indexed).
        cfg:  :class:`TrainConfig` carrying schedule parameters.

    Returns:
        The learning rate for this step.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)

    progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine_decay


# ---------------------------------------------------------------------------
# Named preset configs
# ---------------------------------------------------------------------------

PRESETS: dict[str, tuple[ModelConfig, TrainConfig]] = {
    "tiny": (
        ModelConfig(block_size=128, n_embd=64, n_head=2, n_layer=2, dropout=0.0),
        TrainConfig(steps=3000, batch_size=32, lr=1e-3, warmup_steps=50),
    ),
    "small": (
        ModelConfig(block_size=256, n_embd=128, n_head=4, n_layer=4, dropout=0.1),
        TrainConfig(steps=5000, batch_size=64, lr=3e-4, warmup_steps=100),
    ),
    "medium": (
        ModelConfig(block_size=512, n_embd=256, n_head=8, n_layer=6, dropout=0.1),
        TrainConfig(steps=10000, batch_size=32, accum_steps=4, lr=3e-4, warmup_steps=200),
    ),
}
