"""Functions that deal with Unicode, tokenization and escaping."""
import logging
import re
import sys
import unicodedata

import six


_LOGGER = logging.getLogger(__name__)


def escape_token(token, alphabet):
  """Escape the token and replace unknown chatacters with uni-code."""
  token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
  escaped = [c if c in alphabet and c != u"\n" else r"\%d;" %
             ord(c) for c in token]
  return u"".join(escaped) + "_"


def unescape_token(escaped_token):
  """Unescape the token replacing unicode with charcters."""
  def sub(match):
    if match.group(1) is None:
      return u"_" if match.group(0) == u"\\u" else u"\\"

    try:
      return six.unichr(int(match.group(1)))
    except (ValueError, OverflowError) as _:
      return u"\u3013"  # Unicode for undefined character.

  # Some tokens get an underscore that stands for the end of the word. Remove it
  trimmed = escaped_token[:-1] if escaped_token.endswith("_") else escaped_token
  return _UNESCAPE_REGEX.sub(sub, trimmed)


_UNESCAPE_REGEX = re.compile(r"\\u|\\\\|\\([0-9]+);")


def tokenize(text):
  """Splits based on spaces and whether the subsequence is alphanumeric."""
  tokens = []
  start = 0
  is_alphanumeric = [c in _ALPHANUMERIC_CHAR_SET for c in text]
  for position in range(1, len(text)):
    if is_alphanumeric[position] != is_alphanumeric[position - 1]:
      token = text[start:position]
      if token != u" " or start == 0:
        tokens.append(token)
      start = position
  final_token = text[start:]
  tokens.append(final_token)
  return tokens


def untokenize(tokens):
  token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
  text = []
  for i, token in enumerate(tokens):
    if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
      text.append(u" ")
    text.append(token)
  return "".join(text)


_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))


def unicode_to_native(text):
  if six.PY2:
    return text.encode("utf-8") if is_unicode(text) else text
  return text


def native_to_unicode(text):
  if is_unicode(text):
    return text
  try:
    return to_unicode(text)
  except UnicodeDecodeError:
    assert False, 'UnicodeDecodeError on {}'.format(text)
    res = to_unicode(text, ignore_errors=True)
    return res


def is_unicode(text):
  return isinstance(text, six.text_type)


def to_unicode(text, ignore_errors=False):
  if is_unicode(text):
    return text
  error_mode = "ignore" if ignore_errors else "strict"
  return text.decode("utf-8", errors=error_mode)


UNICODE_ESCAPE_CHARACTERS = set(u"\\_u;0123456789")
