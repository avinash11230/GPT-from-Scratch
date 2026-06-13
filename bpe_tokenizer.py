"""
Byte-Pair Encoding (BPE) tokenizer implemented from scratch.

BPE works by starting with individual bytes/characters as the base vocabulary
and iteratively merging the most frequent adjacent pair into a new token until
the desired vocabulary size is reached.  The resulting tokenizer produces
shorter token sequences than character-level tokenisation while remaining
fully transparent — no external tokenizer library is required.

References:
    Sennrich et al. 2016 "Neural Machine Translation of Rare Words with Subword Units"
    (the original BPE tokenizer paper)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pairs(word: Tuple[str, ...]) -> Dict[Tuple[str, str], int]:
    """Return a frequency map of all adjacent symbol pairs in *word*."""
    pairs: Dict[Tuple[str, str], int] = defaultdict(int)
    for i in range(len(word) - 1):
        pairs[(word[i], word[i + 1])] += 1
    return pairs


def _merge_vocab(
    vocab: Dict[Tuple[str, ...], int],
    pair: Tuple[str, str],
    merged: str,
) -> Dict[Tuple[str, ...], int]:
    """Return a new vocab dict where every occurrence of *pair* is replaced
    by the single symbol *merged*.

    Args:
        vocab:  Mapping from symbol-tuple → word frequency.
        pair:   The two adjacent symbols to merge.
        merged: The new symbol that replaces the pair.

    Returns:
        Updated vocabulary dict.
    """
    new_vocab: Dict[Tuple[str, ...], int] = {}
    a, b = pair
    for word_tuple, freq in vocab.items():
        new_word: List[str] = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == a and word_tuple[i + 1] == b:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BPETrainer:
    """Trains a BPE vocabulary on a corpus of text.

    The training algorithm:
    1. Build an initial word-frequency map.  Each word is split into its
       individual characters plus a special end-of-word marker ``</w>``.
    2. Repeat until *vocab_size* is reached:
       a. Count all adjacent symbol-pair frequencies across the corpus.
       b. Merge the most frequent pair into a single new symbol.
       c. Record the merge rule.

    Args:
        vocab_size:   Target vocabulary size (including base characters).
        special_tokens: Extra tokens like ``<pad>``, ``<unk>`` to pre-add.
        verbose:      Print progress every *verbose* merges (0 = silent).
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        special_tokens: Optional[List[str]] = None,
        verbose: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.special_tokens: List[str] = special_tokens or ["<unk>"]
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, text: str) -> "BPETokenizer":
        """Run BPE training on *text* and return a ready-to-use tokenizer.

        Args:
            text: Raw training corpus.

        Returns:
            A fitted :class:`BPETokenizer`.
        """
        # Step 1: build word-frequency dict
        word_freq = self._build_word_freq(text)

        # Convert each word into a tuple of symbols (chars + </w>)
        vocab: Dict[Tuple[str, ...], int] = {
            tuple(list(word) + ["</w>"]): freq
            for word, freq in word_freq.items()
        }

        # Collect the base character set (all characters in training text + </w>)
        base_symbols = set(text) | {"</w>"}

        # Build initial token → id mapping
        all_tokens: List[str] = list(self.special_tokens)
        for sym in sorted(base_symbols):
            if sym not in all_tokens:
                all_tokens.append(sym)

        merges: List[Tuple[str, str]] = []
        num_merges = self.vocab_size - len(all_tokens)

        # Step 2: iterative merging
        for i in range(num_merges):
            # Count pair frequencies
            pair_freq: Dict[Tuple[str, str], int] = defaultdict(int)
            for word_tuple, freq in vocab.items():
                for pair, cnt in _get_pairs(word_tuple).items():
                    pair_freq[pair] += freq * cnt

            if not pair_freq:
                break

            best_pair = max(pair_freq, key=lambda p: pair_freq[p])
            merged_sym = "".join(best_pair)

            merges.append(best_pair)
            all_tokens.append(merged_sym)
            vocab = _merge_vocab(vocab, best_pair, merged_sym)

            if self.verbose and (i + 1) % self.verbose == 0:
                print(
                    f"  merge {i + 1}/{num_merges}: {best_pair!r} -> {merged_sym!r}"
                    f"  (freq={pair_freq[best_pair]})"
                )

        stoi = {tok: i for i, tok in enumerate(all_tokens)}
        itos = {i: tok for tok, i in stoi.items()}
        return BPETokenizer(stoi=stoi, itos=itos, merges=merges)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_word_freq(text: str) -> Dict[str, int]:
        """Count word frequencies in *text*.  Whitespace is split but
        leading spaces are preserved as part of the word (GPT-style)."""
        freq: Dict[str, int] = defaultdict(int)
        # Split on whitespace, preserving the original words
        for word in re.findall(r"\S+", text):
            freq[word] += 1
        return freq


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """BPE tokenizer.  Encode text to token ids and decode ids back to text.

    Typically created via :class:`BPETrainer`.  Can also be loaded from a
    JSON file saved by :meth:`save`.

    Args:
        stoi:   Symbol → token-id mapping.
        itos:   Token-id → symbol mapping.
        merges: Ordered list of merge rules as ``(a, b)`` pairs.
    """

    UNK = "<unk>"
    EOW = "</w>"

    def __init__(
        self,
        stoi: Dict[str, int],
        itos: Dict[int, str],
        merges: List[Tuple[str, str]],
    ) -> None:
        self.stoi = stoi
        self.itos = itos
        # Build a fast lookup dict: pair → merged symbol
        self._merge_rank: Dict[Tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }
        self.merges = merges
        self.vocab_size = len(stoi)
        self._unk_id = stoi.get(self.UNK, 0)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode *text* into a list of integer token ids.

        Args:
            text: Input string.

        Returns:
            List of integer token ids.
        """
        token_ids: List[int] = []
        for word in re.findall(r"\S+|\s+", text):
            if word.strip() == "":
                # Encode whitespace character by character
                for ch in word:
                    token_ids.append(self.stoi.get(ch, self._unk_id))
            else:
                word_tokens = self._encode_word(word)
                token_ids.extend(word_tokens)
        return token_ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of integer token ids back to a string.

        Args:
            ids: Token id sequence.

        Returns:
            Decoded text.
        """
        tokens = [self.itos.get(i, self.UNK) for i in ids]
        text = "".join(tokens)
        # Remove end-of-word markers, restoring spaces
        text = text.replace(self.EOW, " ")
        return text

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the tokenizer to a JSON file at *path*.

        Args:
            path: Destination file path.
        """
        payload = {
            "kind": "bpe",
            "stoi": self.stoi,
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a tokenizer from a JSON file saved by :meth:`save`.

        Args:
            path: Path to the JSON file.

        Returns:
            A ready-to-use :class:`BPETokenizer`.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        stoi: Dict[str, int] = data["stoi"]
        itos: Dict[int, str] = {int(i): sym for sym, i in stoi.items()}
        merges: List[Tuple[str, str]] = [tuple(pair) for pair in data["merges"]]  # type: ignore[misc]
        return cls(stoi=stoi, itos=itos, merges=merges)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_word(self, word: str) -> List[int]:
        """Apply BPE merges to a single word and return token ids."""
        # Start: each character is its own symbol + end-of-word marker
        symbols = list(word) + [self.EOW]

        # Iteratively apply merges in priority order
        while len(symbols) > 1:
            best_rank = len(self._merge_rank)
            best_idx = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_rank.get(pair, len(self._merge_rank))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break  # no more applicable merges

            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2:]

        return [self.stoi.get(sym, self._unk_id) for sym in symbols]
