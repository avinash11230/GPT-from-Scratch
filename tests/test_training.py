"""
End-to-end training integration test.

Runs a handful of gradient steps on a tiny model and tiny dataset and
asserts that the loss decreases — proving that all components (data loading,
forward pass, backward pass, optimizer) are wired up correctly.
"""

from __future__ import annotations

import torch
import pytest

from config import ModelConfig, TrainConfig
from model import GPT
from train import get_batch, estimate_loss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MICRO_TEXT = (
    "To be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune "
    "or to take arms against a sea of troubles. " * 20
)


@pytest.fixture()
def micro_cfg() -> tuple[ModelConfig, TrainConfig]:
    """Configurations for a micro model that trains in seconds."""
    model_cfg = ModelConfig(
        vocab_size=1,        # filled in after tokenizer
        block_size=32,
        n_embd=32,
        n_head=2,
        n_layer=1,
        dropout=0.0,
        weight_tying=False,  # easier to test param count
    )
    train_cfg = TrainConfig(
        steps=20,
        batch_size=8,
        lr=1e-3,
        warmup_steps=0,
        grad_clip=0.0,
        eval_iters=10,
    )
    return model_cfg, train_cfg


# ---------------------------------------------------------------------------
# Training integration test
# ---------------------------------------------------------------------------

class TestTrainingLossDecreases:
    def test_loss_decreases(self, micro_cfg: tuple[ModelConfig, TrainConfig]) -> None:
        """After 20 gradient steps the loss must be lower than at step 0."""
        model_cfg, train_cfg = micro_cfg
        device = torch.device("cpu")

        # Build a tiny dataset from the micro text
        from tokenizer import CharTokenizer
        tokenizer = CharTokenizer(MICRO_TEXT)
        model_cfg.vocab_size = tokenizer.vocab_size

        data = torch.tensor(tokenizer.encode(MICRO_TEXT), dtype=torch.long)
        split = int(0.9 * len(data))
        train_data, val_data = data[:split], data[split:]

        model = GPT(model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

        # Measure initial loss
        initial_losses = estimate_loss(
            model, train_data, val_data,
            model_cfg.block_size, train_cfg.batch_size, device,
            eval_iters=train_cfg.eval_iters,
        )
        initial_loss = initial_losses["train"]

        # Run training steps
        model.train()
        for _ in range(train_cfg.steps):
            x, y = get_batch(train_data, model_cfg.block_size, train_cfg.batch_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Measure final loss
        final_losses = estimate_loss(
            model, train_data, val_data,
            model_cfg.block_size, train_cfg.batch_size, device,
            eval_iters=train_cfg.eval_iters,
        )
        final_loss = final_losses["train"]

        assert final_loss < initial_loss, (
            f"Training loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_gradient_flows_to_all_params(self, micro_cfg: tuple[ModelConfig, TrainConfig]) -> None:
        """Every parameter must receive a non-zero gradient after one step."""
        model_cfg, train_cfg = micro_cfg
        from tokenizer import CharTokenizer
        tokenizer = CharTokenizer(MICRO_TEXT)
        model_cfg.vocab_size = tokenizer.vocab_size
        model_cfg.weight_tying = False  # ensure lm_head has independent grad

        data = torch.tensor(tokenizer.encode(MICRO_TEXT), dtype=torch.long)
        train_data = data[:int(0.9 * len(data))]

        device = torch.device("cpu")
        model = GPT(model_cfg).to(device)
        x, y = get_batch(train_data, model_cfg.block_size, train_cfg.batch_size, device)
        _, loss = model(x, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"

    def test_num_params_correct(self, micro_cfg: tuple[ModelConfig, TrainConfig]) -> None:
        """Parameter count must be positive and match manual calculation."""
        model_cfg, _ = micro_cfg
        model_cfg.vocab_size = 50
        model_cfg.weight_tying = False
        model = GPT(model_cfg)
        n = model.num_params()
        assert n > 0
        manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n == manual
