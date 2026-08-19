"""
text_cleaning.py — Strip the giveaways out of article text
===========================================================

The problem this solves
-----------------------
Run `python inspect_model.py` on the old model and the strongest signal for
"real" is the word `reuters`, at a weight of -26. Nothing else comes close.
That is not the model spotting a lie — it is the model noticing that every
single article in True.csv came off the Reuters wire, and every article in
Fake.csv came off a blog that pastes `Featured image via ...` under its photos.

So the model never had to read anything. The answer was written in the margin.

Removing those markers forces it back onto what is actually in the writing:
whether claims are attributed to someone, whether the language hedges or
asserts, whether there are specifics. That is a real signal and it is not tied
to one publisher.

What this deliberately does NOT remove
--------------------------------------
Shouty capitals, exclamation marks, insults, "you won't believe", "sources
say" — all of that stays. It is not a publisher fingerprint, it is the
difference between reporting and ranting, and it is exactly what we want the
model to learn from.

Used by BOTH sides, and it has to stay that way
-----------------------------------------------
  * train_honest_model.py cleans every article before learning from it
  * analysis_engine.py cleans the user's text before asking for a prediction

If those two ever disagree the model is fed a kind of text it has never seen,
and its answers quietly turn to noise — no crash, no warning. Same rule as
text_matching.py: it lives here, once, and both sides import it.
"""

import re

# Runs of whitespace collapse to a single space once the cutting is done.
_WHITESPACE_RUN = re.compile(r'\s+')

# Each rule is (name, pattern, replacement). The name is only used for
# reporting — train_honest_model.py counts how often each one fires so you can
# see how much fingerprint was in the corpus to begin with.
#
# Order matters in one place: the dateline rule has to run before the bare
# agency-name rule, or "WASHINGTON (Reuters) -" loses the word "Reuters" and
# leaves "WASHINGTON () -" behind.
_RULES = [

    # --- Who published it ---------------------------------------------------

    # A newswire dateline: "WASHINGTON (Reuters) - ", "LONDON/PARIS (Reuters) -",
    # "NEW YORK (AP) —". The place name is optional because the bracket alone
    # still gives the game away.
    ('agency dateline', re.compile(
        r"\b[A-Z][A-Za-z.'\-]*(?:[ /][A-Z][A-Za-z.'\-]*){0,4}\s*"
        r"\(\s*(?:Reuters|AP|AFP|UPI|Reuters Breakingviews)\s*\)"
        r"\s*[-–—:]*\s*"
    ), ' '),

    # The agency naming itself anywhere else in the body.
    ('agency name', re.compile(
        r'\b(?:reuters|associated\s+press|agence\s+france[-\s]?presse)\b',
        re.IGNORECASE,
    ), ' '),

    # Wire copy signs off with its own staff list: "Reporting by Jeff Mason;
    # Editing by Peter Cooney". No blog does this, so it is a perfect tell.
    ('wire sign-off', re.compile(
        r'\b(?:additional\s+)?(?:reporting|writing|editing)\s+by\s+[^.;\n]{0,90}[.;]?',
        re.IGNORECASE,
    ), ' '),

    # --- Where the pictures came from --------------------------------------

    # The blog boilerplate: "Featured image via Getty Images", "Photo by ...",
    # "Screengrab via YouTube".
    ('image credit', re.compile(
        r'\b(?:featured\s+image|photo|image|screen\s?grab|screenshot)s?\s*'
        r'(?:credit\s*)?(?:via|by|from|courtesy\s+of|:)\s*[^.\n]{0,80}',
        re.IGNORECASE,
    ), ' '),

    # Stock photo agencies, which appear in captions rather than prose.
    ('photo agency', re.compile(
        r'\b(?:getty\s+images|getty|shutterstock|istock|ap\s+photo|'
        r'afp\s*/?\s*getty|pool\s+photo)\b',
        re.IGNORECASE,
    ), ' '),

    # --- Web plumbing -------------------------------------------------------

    ('web address', re.compile(r'https?://\S+|\bwww\.\S+', re.IGNORECASE), ' '),

    # Embedded tweet leftovers. The quoted words stay — only the plumbing goes.
    ('social embed', re.compile(
        r'\bpic\.twitter\.com/\S+|\bt\.co/\S+', re.IGNORECASE,
    ), ' '),

    # "Follow us on Twitter", "Sign up for our newsletter", "Read more:".
    ('site furniture', re.compile(
        r'\b(?:follow\s+us\s+on|like\s+us\s+on|sign\s+up\s+for\s+our|'
        r'subscribe\s+to\s+our|read\s+more)\b[^.\n]{0,60}',
        re.IGNORECASE,
    ), ' '),

    # Formatting tags bolted onto headlines by one particular kind of site:
    # "[VIDEO]", "(WATCH)", "WATCH: ". This is the most debatable rule here —
    # arguably "WATCH:" is a style choice and therefore fair signal — but the
    # bracketed form is a platform artefact, so both go.
    ('media tag', re.compile(
        r'[\[\(]\s*(?:video|watch|photos?|images?|breaking)\s*[\]\)]|'
        r'\b(?:watch|video|photos)\s*:\s*',
        re.IGNORECASE,
    ), ' '),
]


def clean_article_text(text):
    """
    Remove publisher fingerprints from one piece of text.

    Returns '' for empty or None input.

    >>> clean_article_text("WASHINGTON (Reuters) - The Senate voted on Tuesday.")
    'The Senate voted on Tuesday.'
    >>> clean_article_text("He is a TOTAL disaster!!! Featured image via Getty Images")
    'He is a TOTAL disaster!!!'
    >>> clean_article_text(None)
    ''
    """
    if not text:
        return ''

    out = str(text)
    for _name, pattern, replacement in _RULES:
        out = pattern.sub(replacement, out)

    return _WHITESPACE_RUN.sub(' ', out).strip()


def count_fingerprints(text):
    """
    Report which rules would fire on this text, and how many times, without
    changing anything. train_honest_model.py adds these up across the whole
    corpus so you can see what the model was really keying off.

    Returns a dict of {rule name: number of hits}, omitting rules that found
    nothing.
    """
    if not text:
        return {}

    subject = str(text)
    hits = {}
    for name, pattern, _replacement in _RULES:
        found = len(pattern.findall(subject))
        if found:
            hits[name] = hits.get(name, 0) + found

    return hits


def rule_names():
    """The rule names, in the order they are applied."""
    return [name for name, _pattern, _replacement in _RULES]
