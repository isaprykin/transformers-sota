"""Tests for Tokenizer to help with parity with tensor2tensor."""
import random
import string
import unittest

import tokenizer

class TokenizerTest(unittest.TestCase):
  """Test that verifies the basic functions of Tokenizer.

  They are mostly taken from tensor2tensor repo for an easy way to ensure
  parity.
  """

  def test_encode_decode(self):
    """Test that encoding and decoding are consisent."""
    corpus = (
        "This is a corpus of text that provides a bunch of tokens from which "
        "to build a vocabulary. It will be used when strings are encoded "
        "with a TextEncoder subclass. The encoder was coded by a coder")
    alphabet = set(corpus) - {" "}

    original = "This is a coded sentence encoded by the SubwordTextEncoder"

    encoder = tokenizer.Builder(
        100, minimum_threshold=2, maximum_threshold=10
    ).from_corpus([original, corpus])

    encoded = encoder.encode(original)
    decoded = encoder.decode(encoded)
    self.assertEqual(original, decoded)

    subtoken_strings = {encoder.token_strings[i] for i in encoded}
    self.assertIn("encoded_", subtoken_strings)
    self.assertIn("coded_", subtoken_strings)
    self.assertIn("TextEncoder_", encoder.token_strings)
    self.assertIn("coder_", encoder.token_strings)

    self.assertTrue(alphabet.issubset(encoder.alphabet))
    for symbol in alphabet:
      self.assertIn(symbol, encoder.token_strings)

  def test_small_vocab(self):
    """Test whether a small vocabulary is okay."""
    corpus = "The quick brown fox jumps over the lazy dog"
    alphabet = set(corpus) - {" "}

    encoder = tokenizer.Builder(
        10, minimum_threshold=2, maximum_threshold=10
    ).from_corpus([corpus])

    self.assertTrue(alphabet.issubset(encoder.alphabet))
    for symbol in alphabet:
      self.assertIn(symbol, encoder.token_strings)

  def test_long_tokens(self):
    """See whether generating a vocabulary with very long words is slow."""
    token_length = 4000
    num_tokens = 50
    target_vocab_size = 600
    max_subtoken_length = 10
    max_count = 500

    random.seed(0)
    long_tokens = []
    for _ in range(num_tokens):
      long_token = "".join([random.choice(string.ascii_uppercase)
                            for _ in range(token_length)])
      long_tokens.append(long_token)

    corpus = " ".join(long_tokens)
    alphabet = set(corpus) - {" "}

    builder = tokenizer.Builder(
        target_vocab_size, minimum_threshold=1, maximum_threshold=max_count)
    builder.set_maximum_subtoken_length(max_subtoken_length)
    encoder = builder.from_corpus([corpus])

    self.assertTrue(alphabet.issubset(encoder.alphabet))

    for symbol in alphabet:
      self.assertIn(symbol, encoder.token_strings)

  def test_custom_reserved_tokens(self):
    """Test that we can pass custom reserved tokens to SubwordTextEncoder."""
    corpus = "The quick brown fox jumps over the lazy dog"

    start_symbol = "<S>"
    end_symbol = "<E>"
    reserved_tokens = (tokenizer.Tokenizer.RESERVED_TOKENS
                       + [start_symbol, end_symbol])

    builder = tokenizer.Builder(10,
                                minimum_threshold=2, maximum_threshold=10)
    builder.set_reserved_tokens(reserved_tokens)
    encoder = builder.from_corpus([corpus])

    self.assertEqual(encoder.decode([2]), start_symbol)
    self.assertEqual(encoder.decode([3]), end_symbol)

    reconstructed_corpus = encoder.decode(encoder.encode(corpus))
    self.assertEqual(corpus, reconstructed_corpus)

  def test_encodable_when_not_in_alphabet(self):
    """Test that it encodes unknown symbols via the unicode translation."""
    corpus = "the quick brown fox jumps over the lazy dog"

    encoder = tokenizer.Builder(
        100, minimum_threshold=2, maximum_threshold=10
    ).from_corpus([corpus])
    original = "This has UPPER CASE letters that are out of alphabet"

    encoded = encoder.encode(original)
    decoded = encoder.decode(encoded)
    self.assertEqual(original, decoded)

    encoded_str = "".join(encoder.token_strings[i] for i in encoded)
    self.assertIn("\\84;", encoded_str)


if __name__ == '__main__':
  unittest.main()
