"""
Strip publisher fingerprints out of article text.

The old model's strongest signal for "real" was the word `reuters`, at a weight of
-26, with nothing else close. It had noticed that every real article came off the
Reuters wire and every fake one off a blog that writes "Featured image via ..."
under its photos, so it never had to read anything. Removing those markers forces
it back onto the writing. Shouty capitals, insults and "sources say" all stay:
that is reporting versus ranting, which is what we want it to learn.

Used by BOTH sides — train_honest_model.py before learning, analysis_engine.py
before predicting. If they disagree the model sees text it never trained on.
"""

import re

_WHITESPACE_RUN = re.compile(r'\s+')

# (name, pattern, replacement). The name is for reporting only; count_fingerprints
# tallies it. Order matters in one place: the dateline rule must run before the
# bare agency-name rule, or "WASHINGTON (Reuters) -" leaves "WASHINGTON () -".
_RULES = [

    # --- Who published it ---------------------------------------------------

    # A newswire dateline: "WASHINGTON (Reuters) - ", "NEW YORK (AP) —". The place
    # name is optional because the bracket alone gives the game away.
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

    # Blog boilerplate: "Featured image via Getty Images", "Screengrab via YouTube".
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

    # Embedded tweet leftovers. The quoted words stay, only the plumbing goes.
    ('social embed', re.compile(
        r'\bpic\.twitter\.com/\S+|\bt\.co/\S+', re.IGNORECASE,
    ), ' '),

    # "Follow us on Twitter", "Sign up for our newsletter", "Read more:".
    ('site furniture', re.compile(
        r'\b(?:follow\s+us\s+on|like\s+us\s+on|sign\s+up\s+for\s+our|'
        r'subscribe\s+to\s+our|read\s+more)\b[^.\n]{0,60}',
        re.IGNORECASE,
    ), ' '),

    # "[VIDEO]", "(WATCH)", "WATCH: ". The most debatable rule here — arguably
    # "WATCH:" is a style choice and so fair signal — but the bracketed form is a
    # platform artefact, so both go.
    ('media tag', re.compile(
        r'[\[\(]\s*(?:video|watch|photos?|images?|breaking)\s*[\]\)]|'
        r'\b(?:watch|video|photos)\s*:\s*',
        re.IGNORECASE,
    ), ' '),
]


def clean_article_text(text):
    """"WASHINGTON (Reuters) - The Senate voted." becomes "The Senate voted."
    Returns '' for empty or None input."""
    if not text:
        return ''

    out = str(text)
    for _name, pattern, replacement in _RULES:
        out = pattern.sub(replacement, out)

    return _WHITESPACE_RUN.sub(' ', out).strip()


def count_fingerprints(text):
    """Which rules would fire, and how often, without changing anything. Returns
    {rule name: hits}, omitting rules that found nothing. train_honest_model.py
    adds these up across the corpus."""
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
