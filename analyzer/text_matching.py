"""
Tidy text into a comparable form, so the fact-check lookup can match on it.

A saved headline may carry a curly apostrophe (’) where the person pasting it
typed the straight one ('), which is enough for an exact match to fail and the
article to look unknown. Same for a double space or a trailing full stop.

Both sides of the lookup must tidy identically — load_mega_data.py when it saves,
analysis_engine.py when it searches — or matches are silently missed, with no
crash to notice. So the rules live here once and both sides import them.
"""

import re
import unicodedata

# Curly quotes, dashes and other lookalikes that word processors and news sites
# produce, mapped to the plain keyboard equivalent.
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
    ' ': ' ',  # non-breaking space
    '…': '...',  # ellipsis
}

# Punctuation carries no meaning for matching and is the commonest near-miss.
_NOT_WORD_OR_SPACE = re.compile(r'[^a-z0-9 ]+')

# Deleted rather than turned into a space, so "Trump's" becomes "trumps" instead
# of "trump s" — which also makes "Trump's" and "Trumps" match.
_APOSTROPHE = re.compile(r"'")

_WHITESPACE_RUN = re.compile(r'\s+')


def normalize_headline(headline):
    """Folds lookalike characters, strips accents, lowercases, drops punctuation
    and collapses whitespace, in that order. Returns '' for empty or None."""
    if not headline:
        return ''

    text = str(headline)

    for fancy, plain in _LOOKALIKES.items():
        text = text.replace(fancy, plain)

    # café -> cafe, so spelling variants line up
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))

    text = text.lower()

    # Apostrophes vanish entirely; other punctuation becomes a space so words
    # stay separated.
    text = _APOSTROPHE.sub('', text)
    text = _NOT_WORD_OR_SPACE.sub(' ', text)

    text = _WHITESPACE_RUN.sub(' ', text).strip()

    return text


# Measured on the real corpus: colliding rows numbered 2,235 at a 10-word key,
# 683 at 20 and 521 at 30, so 20 is where the curve flattens. Must stay well
# inside the ~100-word stored extract, since the key is built from the OPENING
# words so a whole-article paste still keys the same as our shorter copy.
BODY_KEY_WORDS = 20


def body_key(article_text, words=BODY_KEY_WORDS):
    """The opening of an article body as a key: tidied by normalize_headline, then
    cut to the first `words` words. Returns '' when there is not enough text, and
    the lookup skips blank keys rather than letting short articles all collide."""
    found = normalize_headline(article_text).split()
    if len(found) < words:
        return ''
    return ' '.join(found[:words])
