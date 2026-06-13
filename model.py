"""
GPT-style decoder-only transformer, implemented from scratch using only
basic PyTorch building blocks (nn.Linear, nn.LayerNorm, nn.Embedding).
No pre-built transformer layers are used.

Features
--------
* Multi-head causal self-attention with optional KV-cache for fast inference
* Pre-norm residual connections (LayerNorm applied before each sub-layer)
* GELU activation in the feed-forward blocks
* Weight tying between token embedding and LM head (reduces params, often
  improves perplexity — see Press & Wolf 2017)
* Top-k and nucleus (top-p) sampling in the generation loop
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------------------
# KV-Cache type alias
# ---------------------------------------------------------------------------

# Each layer stores a dict {"k": Tensor(B, n_head, T_cached, head_dim),
#                           "v": Tensor(B, n_head, T_cached, head_dim)}
KVCache = list[dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (look-back-only) mask.

    During *training* this runs the standard O(T²) attention over the full
    sequence.  During *inference* a KV-cache can be passed so that only the
    new token's Q needs to be computed while K/V are reused from previous
    steps, reducing inference cost from O(T²) to O(T).

    Args:
        cfg: :class:`~config.ModelConfig` carrying n_embd, n_head, etc.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Causal mask registered as a buffer so it moves with the model
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(
        self,
        x: torch.Tensor,
        layer_cache: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """Forward pass with optional KV-cache.

        Args:
            x:           Input tensor of shape ``(B, T, C)``.
            layer_cache: Optional dict with keys ``"k"`` and ``"v"`` holding
                         cached keys/values from previous generation steps.
                         If provided, only the new token's Q is used for
                         the query; K and V are concatenated with the cache.

        Returns:
            ``(output, updated_cache)`` where *output* has the same shape as
            *x* and *updated_cache* is ``None`` when no cache was supplied.
        """
        B, T, C = x.shape

        q, k, v = self.qkv_proj(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        new_cache = None
        if layer_cache is not None:
            # Append new K/V to the cache
            if "k" in layer_cache:
                k = torch.cat([layer_cache["k"], k], dim=2)
                v = torch.cat([layer_cache["v"], v], dim=2)
            new_cache = {"k": k, "v": v}

        T_k = k.size(2)  # key length (possibly > T when using cache)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if layer_cache is None:
            # Standard training path: apply causal mask to the full sequence
            attn_scores = attn_scores.masked_fill(
                self.causal_mask[:, :, :T, :T_k] == 0, float("-inf")
            )
        # When using KV cache the new tokens can attend to all cached positions
        # (they are all in the past), so no masking is needed.

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out)), new_cache


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise two-layer MLP with GELU activation.

    The hidden dimension is 4 × the embedding dimension, matching the
    original GPT-2 design.  GELU is used instead of ReLU because it
    has a smoother gradient near zero and performs better on language tasks.

    Args:
        cfg: :class:`~config.ModelConfig` carrying n_embd and dropout.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """One decoder block: pre-norm → attention → residual + pre-norm → MLP → residual.

    Pre-norm (LayerNorm before each sub-layer) is more stable to train than
    post-norm and is the standard for modern transformer language models.

    Args:
        cfg: :class:`~config.ModelConfig`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        layer_cache: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x:           Input tensor ``(B, T, C)``.
            layer_cache: Optional per-layer KV cache dict.

        Returns:
            ``(output, updated_layer_cache)``.
        """
        attn_out, new_cache = self.attn(self.ln1(x), layer_cache)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """GPT decoder-only transformer.

    Token + positional embeddings → N transformer blocks → LayerNorm → LM head.
    The LM head is optionally weight-tied to the token embedding table (see
    ``cfg.weight_tying``), which cuts parameter count and typically improves
    perplexity at no training cost.

    Args:
        cfg: :class:`~config.ModelConfig` with all hyperparameters.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding ↔ LM head weights.
        # This reduces parameters and is standard in modern LMs.
        if cfg.weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Xavier-style initialisation for linear and embedding layers."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self, non_embedding: bool = False) -> int:
        """Count trainable parameters.

        Args:
            non_embedding: If True, exclude the embedding table from the count
                           (useful to see how much of the model is "compute").

        Returns:
            Integer parameter count.
        """
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n -= self.position_embedding.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            idx:      Integer token id tensor of shape ``(B, T)``.
            targets:  Optional target ids ``(B, T)`` for loss computation.
            kv_cache: Optional list of per-layer KV cache dicts for inference.
                      If supplied, ``T`` is expected to be 1 (single new token).

        Returns:
            ``(logits, loss)`` where *loss* is ``None`` when *targets* is not given.
        """
        B, T = idx.shape
        device = idx.device

        # Position ids start after the cached tokens when using KV cache
        cache_len = kv_cache[0]["k"].size(2) if (kv_cache and "k" in kv_cache[0]) else 0
        positions = torch.arange(cache_len, cache_len + T, device=device)

        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)

        new_kv_cache: Optional[KVCache] = [] if kv_cache is not None else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, updated_layer_cache = block(x, layer_cache)
            if new_kv_cache is not None:
                new_kv_cache.append(updated_layer_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Return updated cache in-place by mutating the input list
        if kv_cache is not None and new_kv_cache is not None:
            for i, lc in enumerate(new_kv_cache):
                if lc is not None:
                    kv_cache[i] = lc

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens.

        Sampling strategies (applied in order if both are set):
        1. Temperature scaling
        2. Top-k filtering
        3. Nucleus (top-p) filtering

        Args:
            idx:            Seed context ``(1, T)`` of token ids.
            max_new_tokens: Number of tokens to generate.
            temperature:    Softmax temperature (higher = more random).
            top_k:          Keep only the top-k logits before sampling.
            top_p:          Nucleus sampling — keep the smallest set of tokens
                            whose cumulative probability ≥ top_p.
            use_kv_cache:   If True, maintain a KV cache across steps for O(T)
                            per-step attention cost instead of O(T²).

        Returns:
            Token id tensor of shape ``(1, T + max_new_tokens)``.
        """
        # Initialise KV cache: one empty dict per transformer layer
        kv_cache: Optional[KVCache] = None
        if use_kv_cache:
            kv_cache = [{} for _ in range(len(self.blocks))]

        for _ in range(max_new_tokens):
            if use_kv_cache and kv_cache is not None:
                # Only feed the last token when the cache holds the rest
                idx_cond = idx[:, -1:] if kv_cache[0] else idx[:, -self.cfg.block_size:]
            else:
                idx_cond = idx[:, -self.cfg.block_size:]

            logits, _ = self(idx_cond, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            # Nucleus (top-p) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens whose cumulative prob exceeds top_p
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                # Scatter back to original indexing
                logits = torch.zeros_like(logits).scatter(
                    1, sorted_indices, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx
