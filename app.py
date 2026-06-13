"""
Gradio demo for the GPT from-scratch model.

Loads a trained checkpoint and provides an interactive text-generation
interface with controls for temperature, nucleus sampling (top-p), top-k,
and maximum new tokens.

Usage
-----
::

    python app.py
    python app.py --ckpt_dir checkpoints
    python app.py --ckpt_dir checkpoints --share   # public Gradio link

The app auto-discovers all `*.pt` checkpoint files in the checkpoint directory
so you can switch between training snapshots in the UI.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Iterator

import torch
import gradio as gr

from config import ModelConfig
from model import GPT
from tokenizer import get_tokenizer


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

_loaded_models: dict[str, tuple[GPT, object, ModelConfig]] = {}


def load_model(ckpt_path: str) -> tuple[GPT, object, ModelConfig]:
    """Load (and cache) a checkpoint.

    Returns:
        ``(model, tokenizer, model_cfg)`` triple.
    """
    if ckpt_path in _loaded_models:
        return _loaded_models[ckpt_path]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    raw_cfg = checkpoint.get("model_config", checkpoint.get("config", {}))
    model_cfg = ModelConfig(
        **{k: v for k, v in raw_cfg.items() if k in ModelConfig.__dataclass_fields__}
    )

    ckpt_dir = os.path.dirname(ckpt_path)
    tok_kind = checkpoint.get("tokenizer_kind", "char")
    tokenizer = get_tokenizer(kind=tok_kind, tokenizer_path=os.path.join(ckpt_dir, "tokenizer.json"))

    model = GPT(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _loaded_models[ckpt_path] = (model, tokenizer, model_cfg)
    return model, tokenizer, model_cfg


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_text(
    ckpt_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    use_kv_cache: bool,
) -> Iterator[str]:
    """Generator that yields the output text token-by-token (streaming effect).

    Args:
        ckpt_path:      Full path to the ``.pt`` checkpoint file.
        prompt:         Seed text.
        max_new_tokens: Number of tokens to generate.
        temperature:    Softmax temperature.
        top_p:          Nucleus sampling threshold (1.0 = disabled).
        top_k:          Top-k filter (0 = disabled).
        use_kv_cache:   Use KV cache for faster inference.

    Yields:
        Progressively longer output strings (for Gradio streaming).
    """
    if not ckpt_path:
        yield "⚠️ No checkpoint selected."
        return

    try:
        model, tokenizer, model_cfg = load_model(ckpt_path)
    except Exception as e:
        yield f"❌ Failed to load checkpoint:\n{e}"
        return

    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt or "\n")
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    top_k_val = top_k if top_k > 0 else None
    top_p_val = top_p if top_p < 1.0 else None

    t_start = time.perf_counter()
    generated = list(prompt_ids)

    # Generate token by token so we can stream output
    kv_cache = [{} for _ in range(len(model.blocks))] if use_kv_cache else None
    idx = context

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if use_kv_cache and kv_cache is not None:
                idx_cond = idx[:, -1:] if kv_cache[0] else idx[:, -model_cfg.block_size:]
            else:
                idx_cond = idx[:, -model_cfg.block_size:]

            logits, _ = model(idx_cond, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature

            import torch.nn.functional as F
            if top_k_val is not None:
                values, _ = torch.topk(logits, min(top_k_val, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            if top_p_val is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p_val
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
            generated.append(next_id.item())

            # Yield current decoded output for streaming
            yield tokenizer.decode(generated)

    elapsed = time.perf_counter() - t_start
    n_new = len(generated) - len(prompt_ids)
    tok_per_sec = n_new / elapsed if elapsed > 0 else 0
    final = tokenizer.decode(generated)
    yield final + f"\n\n---\n⏱ {n_new} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)"


def get_model_info(ckpt_path: str) -> str:
    """Return a formatted string with model metadata for the info panel."""
    if not ckpt_path:
        return "No checkpoint loaded."
    try:
        model, _, model_cfg = load_model(ckpt_path)
        return (
            f"**Checkpoint:** `{os.path.basename(ckpt_path)}`\n\n"
            f"**Parameters:** {model.num_params() / 1e6:.3f}M\n\n"
            f"**Architecture:** {model_cfg.n_layer}L / {model_cfg.n_embd}d / "
            f"{model_cfg.n_head} heads / block_size={model_cfg.block_size}\n\n"
            f"**Vocab size:** {model_cfg.vocab_size}\n\n"
            f"**Weight tying:** {'Yes' if model_cfg.weight_tying else 'No'}"
        )
    except Exception as e:
        return f"Error loading checkpoint: {e}"


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui(ckpt_dir: str) -> gr.Blocks:
    """Construct the Gradio Blocks interface.

    Args:
        ckpt_dir: Directory to scan for ``.pt`` checkpoint files.

    Returns:
        The :class:`gr.Blocks` app object.
    """
    checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    default_ckpt = checkpoints[0] if checkpoints else ""

    with gr.Blocks(
        title="GPT From Scratch",
        theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
        css="""
            #header { text-align: center; padding: 1rem 0 0.5rem; }
            #header h1 { font-size: 2rem; font-weight: 800; }
            #header p  { color: #888; font-size: 0.95rem; }
            .generate-btn { font-size: 1.1rem !important; }
        """,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
            <div id="header">
              <h1>🧠 GPT From Scratch</h1>
              <p>Decoder-only transformer trained on Tiny Shakespeare — built from scratch in PyTorch</p>
            </div>
        """)

        with gr.Row():
            # ── Left column: controls ────────────────────────────────────────
            with gr.Column(scale=1):
                ckpt_dropdown = gr.Dropdown(
                    choices=checkpoints,
                    value=default_ckpt,
                    label="Checkpoint",
                    info="Select a trained checkpoint file",
                )
                model_info = gr.Markdown(get_model_info(default_ckpt), label="Model info")
                ckpt_dropdown.change(get_model_info, inputs=ckpt_dropdown, outputs=model_info)

                prompt_box = gr.Textbox(
                    value="ROMEO:",
                    label="Prompt",
                    lines=3,
                    placeholder="Type your seed text here…",
                )
                max_tokens_slider = gr.Slider(
                    minimum=50, maximum=1000, value=300, step=10,
                    label="Max new tokens",
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.05,
                    label="Temperature",
                    info="Higher → more random",
                )
                top_p_slider = gr.Slider(
                    minimum=0.5, maximum=1.0, value=1.0, step=0.01,
                    label="Top-p (nucleus)",
                    info="1.0 = disabled",
                )
                top_k_slider = gr.Slider(
                    minimum=0, maximum=200, value=50, step=1,
                    label="Top-k",
                    info="0 = disabled",
                )
                kv_cache_toggle = gr.Checkbox(value=True, label="Use KV cache (faster)")

                generate_btn = gr.Button(
                    "✨ Generate", variant="primary", elem_classes=["generate-btn"]
                )

            # ── Right column: output ─────────────────────────────────────────
            with gr.Column(scale=2):
                output_box = gr.Textbox(
                    label="Generated text",
                    lines=25,
                    show_copy_button=True,
                    placeholder="Generated text will appear here…",
                )

        # ── Examples ────────────────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["ROMEO:", 300, 0.8, 1.0, 50],
                ["HAMLET:\nTo be, or not", 400, 0.9, 0.95, 40],
                ["KING HENRY:", 250, 0.7, 1.0, 30],
                ["ACT I\nSCENE I", 500, 1.0, 0.9, 60],
            ],
            inputs=[prompt_box, max_tokens_slider, temperature_slider, top_p_slider, top_k_slider],
        )

        # ── Wire up generation ──────────────────────────────────────────────
        generate_btn.click(
            fn=generate_text,
            inputs=[
                ckpt_dropdown,
                prompt_box,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                kv_cache_toggle,
            ],
            outputs=output_box,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Gradio demo for GPT From Scratch")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints",
                   help="Directory containing .pt checkpoint files")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = p.parse_args()

    demo = build_ui(args.ckpt_dir)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
