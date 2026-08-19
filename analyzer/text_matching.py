"""
text_matching.py — The One Place Headlines Get Tidied Up
=========================================================

Both sides of the fact-check lookup MUST tidy headlines the same way:

  * load_mega_data.py tidies each headline before saving it
  * analysis_engine.py tidies the user's headline before searching

If those two ever disagree, the lookup silently stops finding things — no
crash, no warning, it just quietly starts missing matches. So the rule lives
here, once, and both sides import it. Do not reimplement it anywhere else.

Why this is needed: a saved headline like

    Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing

uses a curly apostrophe (’). Someone typing the straight one (') on their
keyboard produced a completely different piece of text, so an exact match
failed and the article looked unknown. Same for a missing semicolon, a double
space, or a trailing full stop.
"""

import re
import unicodedata

# Curly quotes, dashes and other lookalike characters that word processors and
# news sites produce, mapped to the plain keyboard equivalent.
_LOOKALIKES = {
    '‘': "'",  # left single quote
    '’': "'",  # right single quote / curly apostrophe
    '‚': "'",
    '‛': "'",
    '′': "'",  # prime
    'ʼ': "'",  # modifier letter apostrophe
    '“': '"',  # left double quote
    '”': '"',  # right double quote
    '„': '"',
    '″': '"',
    '–': '-',  # en dash
    '—': '-',  # em dash
    '―': '-',
    '−': '-',  # minus sign
    ' ': ' ',  # non-breaking space
    '…': '...',  # ellipsis
}

# Anything that is not a letter, a digit or a space. Punctuation carries no
# meaning for matching purposes and is the most common source of near-misses.
_NOT_WORD_OR_SPACE = re.compile(r'[^a-z0-9 ]+')

# Apostrophes are deleted rather than turned into a space, so "Trump's" becomes
# "trumps" instead of "trump s" — which also makes "Trump's" and "Trumps" match.
_APOSTROPHE = re.compile(r"'")

# Runs of whitespace (including tabs and newlines) collapse to a single space.
_WHITESPACE_RUN = re.compile(r'\s+')


def normalize_headline(headline):
    """
    Reduce a headline to a plain, comparable form.

    Applies the same steps every time, in the same order:
      1. Replace curly quotes/dashes with their plain keyboard equivalents
      2. Split accented characters apart and drop the accent marks
      3. Lowercase
      4. Delete apostrophes, then turn remaining punctuation into spaces
      5. Collapse runs of whitespace and trim the ends

    Returns '' for empty or None input.

    >>> normalize_headline("Donald Trump's  New Year’s Eve Message; Disturbing!")
    'donald trumps new years eve message disturbing'
    >>> a = normalize_headline("It’s Fake")     # curly apostrophe
    >>> b = normalize_headline("It's fake.")    # straight, plus a full stop
    >>> a == b
    True
    >>> normalize_headline(None)
    ''
    """
    if not headline:
        return ''

    text = str(headline)

    # 1. Fold lookalike characters down to plain ASCII equivalents
    for fancy, plain in _LOOKALIKES.items():
        text = text.replace(fancy, plain)

    # 2. Strip accents (café -> cafe) so spelling variants line up
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))

    # 3. Case no longer matters
    text = text.lower()

    # 4. Drop punctuation — this is what makes a missing semicolon, a stray
    #    quote or a trailing full stop stop mattering. Apostrophes vanish
    #    entirely; everything else becomes a space so words stay separated.
    text = _APOSTROPHE.sub('', text)
    text = _NOT_WORD_OR_SPACE.sub(' ', text)

    # 5. One space between words, nothing at the ends
    text = _WHITESPACE_RUN.sub(' ', text).strip()

    return text
