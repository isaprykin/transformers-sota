"""Vocabulary of tokens for the text corpus."""

import collections
import logging
import itertools

import six

import unicode_tokens


_LOGGER = logging.getLogger(__name__)


class Tokenizer:
  """Encodes text in token ids and decodes it back."""

  def __init__(self, alphabet, token_strings):
    """Initializes a Tokenizer.

    Use Builder for learning a vocabulary for Tokenizer.

    Args:
        alphabet: the known universe of chatacters.
        token_strings: list of tokens in the vocabulary.
    """
    self.alphabet = alphabet
    self.token_strings = token_strings
    self._token_to_id = {
        t: i
        for i, t in enumerate(token_strings) if t
    }
    self._max_token_length = max([len(t) for t in token_strings])

  def encode(self, text):
    """Encode the text into token ids."""
    tokens = unicode_tokens.tokenize(unicode_tokens.native_to_unicode(text))
    token_ids = []
    for token in tokens:
      token_ids.extend(
          self._escaped_token_to_token_ids(
              unicode_tokens.escape_token(token, self.alphabet)))
    return token_ids

  def _escaped_token_to_token_ids(self, escaped_token):
    return [
        self._token_to_id[token]
        for token in _token_to_subtoken_strings(
            escaped_token, self._token_to_id, self._max_token_length)
    ]

  def decode(self, token_ids):
    """Decode the text from token ids."""
    return unicode_tokens.unicode_to_native(unicode_tokens.untokenize(
        _token_ids_to_tokens(token_ids, self.token_strings)))

  def store_to_file(self, filename, add_single_quotes=True):
    """Write the token vocabulary into a file."""
    with open(filename, 'w') as opened_file:
      for token in self.token_strings:
        if add_single_quotes:
          opened_file.write(
              '\'' + unicode_tokens.unicode_to_native(token) + '\'\n')
        else:
          opened_file.write(
              unicode_tokens.unicode_to_native(token) + '\n')

  PAD = '<pad>'
  EOS = '<EOS>'
  RESERVED_TOKENS = [PAD, EOS]
  VOCAB_FILENAME = 'vocab.subwords'


def _token_ids_to_tokens(token_ids, token_strings):
  concatenated = ''.join(
      [_token_id_to_token_string(token_id, token_strings)
       for token_id in token_ids])
  split = concatenated.split('_')
  tokens = []
  for part in split:
    if part:
      unescaped = unicode_tokens.unescape_token(part + '_')
      if unescaped:
        tokens.append(unescaped)
  return tokens


def _token_id_to_token_string(token_id, token_strings):
  if 0 <= token_id < len(token_strings):
    return token_strings[token_id]
  return u''


class Builder:
  """Learns a vocabulary for Tokenizer."""

  def __init__(self, target_size,
               minimum_threshold=1, maximum_threshold=1e3):
    """Initializes Builder with vocabulary parameters.

    Args:
        target_size:  the target size of the vocabulary.
          See OKAY_PERCENT_THRESHOLD.
        minimum_threshold: the minimum bound for how frequent a word have
          to be to become a token.
        maximum_threshold: the maximum bound for how frequent a word have
          to be to become a token.  The right threshold is going to be found
          between minimum_threshold and maximum_threshold.
    """
    self._target_size = target_size
    self._minimum_threshold = minimum_threshold
    self._maximum_threshold = maximum_threshold
    self._max_subtoken_length = None
    self._reserved_tokens = Tokenizer.RESERVED_TOKENS

  @staticmethod
  def from_file(filepath):
    """Load the Tokenizer from a vocabulary stored in a file."""
    tokens = []
    with open(filepath, 'r') as opened_file:
      for token in opened_file:
        token = token.rstrip()
        if token.startswith('\''):
          assert token.endswith('\'')
          token = token[1:-1]
        tokens.append(unicode_tokens.native_to_unicode(token))
    alphabet = _generate_alphabet(tokens)
    return Tokenizer(alphabet, tokens)

  def from_corpus(self, corpus):  # pylint: disable=too-many-locals
    """Learn vocabulary from the corpus.

    The algorithm for creating a vocabulary is as follows:
    Let the current vocabulary consist of the alphabet symbols.
    Repeatedly, 1) how frequent tokens are in the corpus given the
    current vocabulary 2) come up with new subtokens by extending
    and shrinking the current ones until the end of the word.  Take a
    note how frequent these new subtokens are across the corpus. 3) Accept
    tokens that are more frequent than X into the current vocabulary. 4)
    Repeat 1-3 with the new incremental vocabulary NUM_ITERATIONS times.
    5) Binary search such an X from the
    minimum_threshold<=X<=maximum_threshold range, such that the vocabulary is
    within OKAY_PERCENT_THRESHOLD of the target size.

    Args:
        corpus: Corpus for creating a vocabulary.
    """
    token_counts = _count_tokens(corpus)

    alphabet = _generate_alphabet(token_counts.keys(), self._reserved_tokens)
    escaped_reserved_tokens = [
        unicode_tokens.escape_token(unicode_tokens.native_to_unicode(t),
                                    alphabet)
        for t in self._reserved_tokens
    ]

    token_to_id = []
    def is_within_threshold():
      difference = abs(len(token_to_id) - self._target_size)
      allowed_difference = (
          self._target_size * (Builder.OKAY_PERCENT_THRESHOLD / 100))
      return difference < allowed_difference

    min_threshold = self._minimum_threshold
    max_threshold = self._maximum_threshold
    current_threshold = max_threshold + 1
    while (not is_within_threshold() and min_threshold < max_threshold
           and current_threshold > 2):
      if len(token_to_id) < self._target_size:
        max_threshold = current_threshold - 1
      elif len(token_to_id) > self._target_size:
        min_threshold = current_threshold + 1
      current_threshold = (min_threshold + max_threshold) // 2

      token_to_id = {
          s: i + len(self._reserved_tokens)
          for i, s in enumerate(list(alphabet)) if s
      }

      for _ in range(Builder.NUM_ITERATIONS):
        new_subtoken_counts = collections.Counter()
        max_token_length = max([len(s) for s in token_to_id.keys()])

        for token, count in six.iteritems(token_counts):
          escaped_token = unicode_tokens.escape_token(token, alphabet)
          current_subtokens = _token_to_subtoken_strings(
              escaped_token, token_to_id, max_token_length)

          self._break_up_into_new_subtokens(
              escaped_token, current_subtokens, token_count=count,
              # `new_subtoken_counts` is the output:
              new_subtoken_counts=new_subtoken_counts
          )

        token_strings_by_length = _bucket_by_length(
            new_subtoken_counts, current_threshold)

        new_token_strings = _pick_longest_new_tokens(
            token_strings_by_length, new_subtoken_counts, current_threshold,
            alphabet, escaped_reserved_tokens)

        # Accept new tokens to use in the next iteration.
        token_to_id = {
            s: i
            for i, s in enumerate(new_token_strings) if s
        }

      _LOGGER.info('%d potential vocabulary tokens at the %d cut-off.',
                   len(new_token_strings), current_threshold)

    return Tokenizer(alphabet, new_token_strings)

  NUM_ITERATIONS = 4
  OKAY_PERCENT_THRESHOLD = 1

  def _break_up_into_new_subtokens(self, token, current_subtokens, token_count,
                                   new_subtoken_counts):
    start = 0
    for subtoken in current_subtokens:
      how_far_to_look = self._max_subtoken_length or len(token) + 1
      last_position = min(len(token) + 1, start + how_far_to_look)

      for end in range(start + 1, last_position):
        new_subtoken = token[start:end]
        new_subtoken_counts[new_subtoken] += token_count
      start += len(subtoken)

  def set_reserved_tokens(self, reserved_tokens):
    self._reserved_tokens = reserved_tokens

  def set_maximum_subtoken_length(self, max_subtoken_length):
    self._max_subtoken_length = max_subtoken_length


def _count_tokens(corpus):
  token_counts = collections.Counter()
  for item in corpus:
    tokens = unicode_tokens.tokenize(unicode_tokens.native_to_unicode(item))
    for token in tokens:
      token_counts[token] += 1
  return token_counts


def _generate_alphabet(tokens, reserved_tokens=None):
  reserved_tokens = reserved_tokens or []
  universe_of_tokens = itertools.chain(
      tokens,
      [unicode_tokens.native_to_unicode(t) for t in reserved_tokens],
      unicode_tokens.UNICODE_ESCAPE_CHARACTERS)
  alphabet = set({char for token in universe_of_tokens for char in token})
  return alphabet


def _token_to_subtoken_strings(token, token_to_id, max_subtoken_length):
  token_strings = []
  start = 0
  while start < len(token):
    for end in range(min(len(token), start + max_subtoken_length),
                     start, -1):
      subtoken = token[start:end]
      if subtoken in token_to_id:
        token_strings.append(subtoken)
        start = end
        break
  return token_strings


def _bucket_by_length(subtoken_counts, threshold):
  subtokens_by_length = []
  for subtoken, count in six.iteritems(subtoken_counts):
    length = len(subtoken)
    if count >= threshold:
      while len(subtokens_by_length) <= length:
        subtokens_by_length.append(set())
      subtokens_by_length[length].add(subtoken)
  return subtokens_by_length


def _pick_longest_new_tokens(token_strings_by_length, new_token_counts,
                             current_threshold, alphabet,
                             escaped_reserved_tokens):
  new_tokens = []
  for length in range(len(token_strings_by_length) - 1, 0, -1):
    tokens = token_strings_by_length[length]
    for token in tokens:
      count = new_token_counts[token]
      if count >= current_threshold:
        if token not in alphabet:
          new_tokens.append((count, token))

        for i in range(1, length):
          new_token_counts[token[:i]] -= count

  new_tokens.extend((new_token_counts.get(a, 0), a)
                    for a in alphabet)

  new_tokens.sort(reverse=True)
  new_tokens = [token for _, token in new_tokens]
  new_tokens = escaped_reserved_tokens + new_tokens
  return new_tokens
