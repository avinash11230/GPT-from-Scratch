"""
Tokenizer module.

Provides two tokenizer implementations that share the same encode/decode/save/load
interface so the rest of the codebase is tokenizer-agnostic:

* :class:`CharTokenizer` — original character-level tokenizer (one token per
  character).  Fast, simple, large vocabulary for small texts.

* See :mod:`bpe_tokenizer` for :class:`BPETokenizer` — sub-word BPE tokenizer
  that yields shorter sequences at the cost of a training pass.

Use :func:`get_tokenizer` to select between them via a string flag.
"""

from __future__ import annotations

import json
from typing import List, Union


class CharTokenizer:
    """Character-level tokenizer.

    Every unique character in the training corpus gets its own token id.
    Vocabulary size equals the number of distinct characters, which is
    typically 60–100 for English text.

    Args:
        text: Full training corpus.  Used to build the character vocabulary.
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        """Map each character in *text* to its integer token id."""
        return [self.stoi[ch] for ch in text]

    def decode(self, indices: List[int]) -> str:
        """Convert a list of integer ids back to a string."""
        return "".join(self.itos[i] for i in indices)

    def save(self, path: str) -> None:
        """Serialise vocabulary to a JSON file.

        Args:
            path: Destination file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "char", "stoi": self.stoi}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Load a :class:`CharTokenizer` from a JSON file.

        Args:
            path: Path to the JSON vocabulary file.

        Returns:
            Loaded tokenizer instance.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.stoi = data["stoi"]
        tok.itos = {int(i): ch for ch, i in tok.stoi.items()}
        tok.vocab_size = len(tok.stoi)
        return tok


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_tokenizer(
    kind: str,
    text: str = "",
    vocab_size: int = 2048,
    tokenizer_path: str = "",
) -> Union[CharTokenizer, "BPETokenizer"]:
    """Return the tokenizer selected by *kind*.

    Args:
        kind:            ``"char"`` or ``"bpe"``.
        text:            Training corpus (required when training a new tokenizer).
        vocab_size:      Target BPE vocabulary size (ignored for ``"char"``).
        tokenizer_path:  If non-empty, load an existing tokenizer from this
                         JSON file instead of training a new one.

    Returns:
        A tokenizer with ``encode``, ``decode``, ``save``, and ``load`` methods
        and a ``vocab_size`` attribute.

    Raises:
        ValueError: If *kind* is not ``"char"`` or ``"bpe"``.
    """
    if kind == "char":
        if tokenizer_path:
            return CharTokenizer.load(tokenizer_path)
        if not text:
            raise ValueError("text must be provided to train a CharTokenizer")
        return CharTokenizer(text)

    if kind == "bpe":
        from bpe_tokenizer import BPETokenizer, BPETrainer

        if tokenizer_path:
            return BPETokenizer.load(tokenizer_path)
        if not text:
            raise ValueError("text must be provided to train a BPETokenizer")
        print(f"Training BPE tokenizer (target vocab_size={vocab_size}) ...")
        trainer = BPETrainer(vocab_size=vocab_size, verbose=200)
        tok = trainer.train(text)
        print(f"BPE training complete - actual vocab_size={tok.vocab_size}")
        return tok

    raise ValueError(f"Unknown tokenizer kind: {kind!r}. Choose 'char' or 'bpe'.")


# Re-export BPETokenizer so callers can do `from tokenizer import BPETokenizer`
try:
    from bpe_tokenizer import BPETokenizer  # noqa: F401
except ImportError:
    pass
