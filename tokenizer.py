"""
A simple character-level tokenizer. Each unique character in the
training text becomes one token. This keeps the project dependency-free
and easy to explain end to end (no external tokenizer libraries).
"""

import json


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, indices):
        return "".join(self.itos[i] for i in indices)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"stoi": self.stoi}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.stoi = data["stoi"]
        tok.itos = {int(i): ch for ch, i in tok.stoi.items()}
        tok.vocab_size = len(tok.stoi)
        return tok
