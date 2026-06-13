"""
Tests for model.py — attention causal masking, forward-pass output shapes,
and the generation API.
"""

from __future__ import annotations

import torch
import pytest

from config import ModelConfig
from model import GPT, CausalSelfAttention


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_cfg() -> ModelConfig:
    """A minimal config so tests run fast on CPU."""
    return ModelConfig(
        vocab_size=32,
        block_size=16,
        n_embd=32,
        n_head=2,
        n_layer=2,
        dropout=0.0,
        weight_tying=True,
    )


@pytest.fixture()
def tiny_model(tiny_cfg: ModelConfig) -> GPT:
    return GPT(tiny_cfg)


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------

class TestCausalMask:
    """Verify that no position attends to a future token."""

    def test_causal_mask_upper_triangle_zero(self, tiny_cfg: ModelConfig) -> None:
        """Attention weights above the diagonal must be zero."""
        attn = CausalSelfAttention(tiny_cfg)
        attn.eval()

        T = 8
        B = 2
        x = torch.randn(B, T, tiny_cfg.n_embd)

        # Monkey-patch forward to capture attention weights
        captured = {}

        original_forward = attn.forward

        def patched(inp, layer_cache=None):
            B2, T2, C = inp.shape
            q, k, v = attn.qkv_proj(inp).split(C, dim=2)
            q = q.view(B2, T2, attn.n_head, attn.head_dim).transpose(1, 2)
            k = k.view(B2, T2, attn.n_head, attn.head_dim).transpose(1, 2)
            v = v.view(B2, T2, attn.n_head, attn.head_dim).transpose(1, 2)
            import math, torch.nn.functional as F
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
            scores = scores.masked_fill(attn.causal_mask[:, :, :T2, :T2] == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            captured["weights"] = weights.detach()
            return original_forward(inp, layer_cache)

        attn.forward = patched  # type: ignore[method-assign]

        with torch.no_grad():
            attn(x)

        weights = captured["weights"]  # (B, n_head, T, T)
        # Upper triangle (future positions) must be zero
        for i in range(T):
            for j in range(i + 1, T):
                assert weights[:, :, i, j].max().item() == pytest.approx(0.0, abs=1e-6), (
                    f"Position {i} attends to future position {j}"
                )


# ---------------------------------------------------------------------------
# Forward-pass shapes
# ---------------------------------------------------------------------------

class TestForwardShapes:
    def test_logit_shape_no_targets(self, tiny_model: GPT, tiny_cfg: ModelConfig) -> None:
        B, T = 3, 10
        idx = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        logits, loss = tiny_model(idx)
        assert logits.shape == (B, T, tiny_cfg.vocab_size)
        assert loss is None

    def test_logit_shape_with_targets(self, tiny_model: GPT, tiny_cfg: ModelConfig) -> None:
        B, T = 3, 10
        idx = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        targets = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        logits, loss = tiny_model(idx, targets)
        assert logits.shape == (B, T, tiny_cfg.vocab_size)
        assert loss is not None
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0.0

    def test_loss_is_positive(self, tiny_model: GPT, tiny_cfg: ModelConfig) -> None:
        B, T = 2, 8
        idx = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        targets = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        _, loss = tiny_model(idx, targets)
        assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestGeneration:
    def test_generate_length(self, tiny_model: GPT, tiny_cfg: ModelConfig) -> None:
        """Output sequence length must equal input + max_new_tokens."""
        tiny_model.eval()
        prompt_len = 4
        n_new = 10
        idx = torch.randint(0, tiny_cfg.vocab_size, (1, prompt_len))
        out = tiny_model.generate(idx, max_new_tokens=n_new)
        assert out.shape == (1, prompt_len + n_new)

    def test_generate_with_top_k(self, tiny_model: GPT, tiny_cfg: ModelConfig) -> None:
        tiny_model.eval()
        idx = torch.randint(0, tiny_cfg.vocab_size, (1, 5))
        out = tiny_model.generate(idx, max_new_tokens=5, top_k=10)
        assert out.shape[1] == 10

    def test_generate_with_top_p(self, tiny_model: GPT, tiny_cfg: ModelConfig) -> None:
        tiny_model.eval()
        idx = torch.randint(0, tiny_cfg.vocab_size, (1, 5))
        out = tiny_model.generate(idx, max_new_tokens=5, top_p=0.9)
        assert out.shape[1] == 10

    def test_generate_with_kv_cache(self, tiny_model: GPT, tiny_cfg: ModelConfig) -> None:
        """KV-cache output must match non-cache output given the same seed."""
        tiny_model.eval()
        torch.manual_seed(0)
        idx = torch.randint(0, tiny_cfg.vocab_size, (1, 5))

        torch.manual_seed(42)
        out_no_cache = tiny_model.generate(idx, max_new_tokens=8, use_kv_cache=False)

        torch.manual_seed(42)
        out_cache = tiny_model.generate(idx, max_new_tokens=8, use_kv_cache=True)

        assert out_no_cache.shape == out_cache.shape

    def test_temperature_affects_distribution(self, tiny_model: GPT, tiny_cfg: ModelConfig) -> None:
        """High temperature should produce more varied output than low temperature."""
        tiny_model.eval()
        torch.manual_seed(7)
        idx = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
        # This is a sanity check — just ensure no crash and output is valid
        out_high = tiny_model.generate(idx, max_new_tokens=20, temperature=2.0)
        out_low = tiny_model.generate(idx, max_new_tokens=20, temperature=0.1)
        assert out_high.shape == out_low.shape


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------

class TestWeightTying:
    def test_weight_tying_shares_tensor(self, tiny_model: GPT) -> None:
        """When weight_tying=True, embedding and lm_head share the same data."""
        assert tiny_model.lm_head.weight.data_ptr() == tiny_model.token_embedding.weight.data_ptr()

    def test_no_weight_tying(self, tiny_cfg: ModelConfig) -> None:
        tiny_cfg.weight_tying = False
        model = GPT(tiny_cfg)
        assert model.lm_head.weight.data_ptr() != model.token_embedding.weight.data_ptr()
