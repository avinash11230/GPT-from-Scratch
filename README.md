# LLM From Scratch

A small GPT-style language model built from scratch in PyTorch, no
transformer libraries used. Every component, embeddings, multi-head
causal self-attention, the feed-forward block, and the autoregressive
sampling loop, is implemented directly so the full architecture fits
in about 150 lines of readable code.

The model is trained on the Tiny Shakespeare dataset at the character
level and learns to generate Shakespeare-like text after a few
thousand training steps.

## Architecture

- Token + learned positional embeddings
- Stack of decoder blocks, each with:
  - Multi-head causal self-attention (masked so tokens can't see the future)
  - Position-wise feed-forward network with GELU
  - Residual connections and LayerNorm (pre-norm)
- Final LayerNorm and linear projection to vocabulary logits

Default config: 4 layers, 4 attention heads, 128-dim embeddings,
128-token context window, roughly 0.8M parameters. All of these are
adjustable via command-line flags.

## Project structure

```
llm-from-scratch/
├── model.py        # GPT architecture: attention, MLP, transformer blocks
├── tokenizer.py     # character-level tokenizer
├── train.py         # training loop with train/val loss tracking
├── generate.py       # load a checkpoint and sample text
├── data/input.txt    # Tiny Shakespeare dataset
└── requirements.txt
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py --steps 2000
```

Generate text from the trained checkpoint:

```bash
python generate.py --prompt "ROMEO:" --max_new_tokens 300
```

## Notes

This is intentionally a small-scale, educational implementation. It is
built to demonstrate understanding of transformer internals (attention
math, causal masking, residual streams, autoregressive decoding) rather
than to compete with production LLMs. The architecture scales directly:
increasing `--n_layer`, `--n_embd`, and `--block_size` produces a larger
model with the same code.
