"""
Load a trained checkpoint and generate text from it.

Usage:
    python generate.py --prompt "ROMEO:" --max_new_tokens 300
"""

import argparse
import torch

from model import GPT
from tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(f"{args.ckpt_dir}/model.pt", map_location=device)
    tokenizer = CharTokenizer.load(f"{args.ckpt_dir}/tokenizer.json")

    model = GPT(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = model.generate(context, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
