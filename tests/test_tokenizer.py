"""
Tests for the tokenizer implementations.

Covers:
* CharTokenizer round-trip (encode → decode returns original string)
* BPETokenizer round-trip
* BPE vocabulary size is within expected range
* Factory function dispatches correctly
"""

from __future__ import annotations

import pytest

from tokenizer import CharTokenizer, get_tokenizer
from bpe_tokenizer import BPETrainer, BPETokenizer


SAMPLE_TEXT = (
    "Hello world! This is a test of the tokenizer. "
    "It should handle punctuation, numbers like 42, and UPPERCASE too."
)

SHAKESPEARE_SNIPPET = (
    "ROMEO:\nBut, soft! what light through yonder window breaks?\n"
    "It is the east, and Juliet is the sun.\n"
    "Arise, fair sun, and kill the envious moon,\n"
    "Who is already sick and pale with grief,\n"
    "That thou her maid art far more fair than she.\n"
)


# ---------------------------------------------------------------------------
# CharTokenizer
# ---------------------------------------------------------------------------

class TestCharTokenizer:
    def test_roundtrip_simple(self) -> None:
        tok = CharTokenizer(SAMPLE_TEXT)
        ids = tok.encode(SAMPLE_TEXT)
        decoded = tok.decode(ids)
        assert decoded == SAMPLE_TEXT

    def test_roundtrip_shakespeare(self) -> None:
        tok = CharTokenizer(SHAKESPEARE_SNIPPET)
        ids = tok.encode(SHAKESPEARE_SNIPPET)
        assert tok.decode(ids) == SHAKESPEARE_SNIPPET

    def test_vocab_size(self) -> None:
        tok = CharTokenizer(SAMPLE_TEXT)
        assert tok.vocab_size == len(set(SAMPLE_TEXT))

    def test_encode_returns_list_of_ints(self) -> None:
        tok = CharTokenizer("abc")
        ids = tok.encode("abc")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_save_load_roundtrip(self, tmp_path) -> None:
        tok = CharTokenizer(SAMPLE_TEXT)
        path = str(tmp_path / "char_tok.json")
        tok.save(path)
        tok2 = CharTokenizer.load(path)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.decode(tok2.encode(SAMPLE_TEXT)) == SAMPLE_TEXT


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------

class TestBPETokenizer:
    @pytest.fixture(scope="class")
    def trained_tokenizer(self) -> BPETokenizer:
        """Train a small BPE tokenizer on the Shakespeare snippet."""
        trainer = BPETrainer(vocab_size=150, verbose=0)
        return trainer.train(SHAKESPEARE_SNIPPET * 5)  # repeat for more signal

    def test_roundtrip(self, trained_tokenizer: BPETokenizer) -> None:
        """Encode → decode should recover the original text (modulo EOW markers)."""
        ids = trained_tokenizer.encode(SHAKESPEARE_SNIPPET)
        decoded = trained_tokenizer.decode(ids)
        # BPE replaces </w> with a space, so we normalise whitespace for comparison
        original_normalised = " ".join(SHAKESPEARE_SNIPPET.split())
        decoded_normalised = " ".join(decoded.split())
        assert decoded_normalised == original_normalised

    def test_vocab_size_in_range(self, trained_tokenizer: BPETokenizer) -> None:
        """Actual vocab size should be ≤ the requested size."""
        assert trained_tokenizer.vocab_size <= 150
        assert trained_tokenizer.vocab_size > 10  # at minimum the base characters

    def test_encode_returns_valid_ids(self, trained_tokenizer: BPETokenizer) -> None:
        ids = trained_tokenizer.encode("Hello world")
        assert all(0 <= i < trained_tokenizer.vocab_size for i in ids)

    def test_save_load_roundtrip(self, trained_tokenizer: BPETokenizer, tmp_path) -> None:
        path = str(tmp_path / "bpe_tok.json")
        trained_tokenizer.save(path)
        tok2 = BPETokenizer.load(path)
        assert tok2.vocab_size == trained_tokenizer.vocab_size
        ids1 = trained_tokenizer.encode(SHAKESPEARE_SNIPPET)
        ids2 = tok2.encode(SHAKESPEARE_SNIPPET)
        assert ids1 == ids2

    def test_empty_string(self, trained_tokenizer: BPETokenizer) -> None:
        ids = trained_tokenizer.encode("")
        assert ids == []
        assert trained_tokenizer.decode([]) == ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestGetTokenizer:
    def test_get_char_tokenizer(self) -> None:
        tok = get_tokenizer("char", text=SAMPLE_TEXT)
        assert isinstance(tok, CharTokenizer)

    def test_get_bpe_tokenizer(self) -> None:
        tok = get_tokenizer("bpe", text=SHAKESPEARE_SNIPPET * 3, vocab_size=100)
        assert isinstance(tok, BPETokenizer)

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tokenizer kind"):
            get_tokenizer("sentencepiece", text="hello")

    def test_char_from_file(self, tmp_path) -> None:
        tok = CharTokenizer(SAMPLE_TEXT)
        path = str(tmp_path / "tok.json")
        tok.save(path)
        tok2 = get_tokenizer("char", tokenizer_path=path)
        assert isinstance(tok2, CharTokenizer)
        assert tok2.vocab_size == tok.vocab_size
