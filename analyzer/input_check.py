"""
input_check.py — Is this actually an English news article?
=========================================================

Why this exists
---------------
The reading model was only ever taught one question: does this look like the
English news it was trained on? It has no way to say "I cannot read this", so
anything unfamiliar lands on the fake side of the line. Measured on 2 Sep 2026:

    English real news        Likely Real   score 0.0003
    the same story in Urdu   Likely Fake   score 0.9866
    Hindi                    Likely Fake   score 0.9933
    Arabic                   Likely Fake   score 0.9911
    Spanish                  Likely Fake   score 0.3498
    random symbols @#$%^&*   Likely Fake   score 0.9950
    keyboard mash asdkjfh    Likely Fake   score 0.9998
    just numbers             Likely Fake   score 0.9996
    emoji only               Likely Fake   score 0.9930
    the single word "news"   Likely Fake   score 0.9980

Every one of those would have been shown to the reader as "Likely Fake — when it
says this it is right 86.9% of the time". That 86.9% was measured on English news
articles. Quoting it over a line of emoji is inventing a statistic, which is the
one thing this project has refused to do everywhere else. Better to decline the
question than to answer it with a fabricated number.

So: refuse clearly, and never let unreadable input reach the model.

How the checks were chosen
--------------------------
Both thresholds were measured against the 6,147 McIntire articles of 25+ words,
not picked by feel.

  LATIN_FLOOR 0.85
    Share of letters that are ordinary A-Z. Real English news never dropped
    below 0.973. Urdu, Hindi, Arabic, Chinese and Russian all score 0.000, so
    this separates non-Latin scripts with a wide margin.

  ENGLISH_FLOOR 0.10
    Share of words that are common English function words -- the, of, and, to.
    English prose is full of them: median 0.359, and 99.93% of real articles are
    above 0.10. Latin-script languages have almost none of their own, because
    Spanish and French function words are different words: Spanish 0.015, French
    0.033, German 0.041, romanised Urdu 0.019. So this catches what the script
    check cannot, and rejects 4 real articles in 6,147 doing it.

    It also happens to catch keyboard mash, digits, symbols and emoji, all of
    which score 0.000 -- they contain no English function words either.

MIN_WORDS is 25 because the model reads the whole input as one piece and a
handful of words is not an article; "news" alone scored 0.998 fake. MAX_WORDS is
400 as a product decision. Note that the model only ever reads the first ~200
words (256 word-pieces, set by MAX_LENGTH when it was trained), so 400 is
already generous relative to what actually gets judged.

Usage
-----
    from analyzer.input_check import check_input

    problem = check_input(text)
    if problem:
        ...show problem.message to the user, do not run the model...
"""

import re
from collections import namedtuple

# Word count bounds for a thing we are willing to call an article.
MIN_WORDS = 25
MAX_WORDS = 400

# See the module docstring for how both of these were measured.
LATIN_FLOOR = 0.85
ENGLISH_FLOOR = 0.10

# The most common English function words. Deliberately function words rather
# than topic words: every piece of English prose is full of these regardless of
# what it is about, which is exactly what makes them a language signal rather
# than a subject signal.
ENGLISH_MARKERS = frozenset("""
the of and to in a is that for it on was with as at by from be this
have or an not are but had has were they which you we their all been more will
would there can who said one about when what so its if out up says
""".split())

# What a failed check returns. `code` is for tests and logs, `message` is shown
# to the reader, `detail` carries the measured numbers so a template or a log can
# say how far off the input was.
Problem = namedtuple('Problem', 'code message detail')


def word_count(text):
    """Words, counted the way a person would count them."""
    return len(re.findall(r"[\w'-]+", text or ''))


def latin_share(text):
    """
    Fraction of the letters that are ordinary A-Z.

    Only letters are counted. Punctuation, digits and spaces are ignored, since
    a sentence with a lot of commas is not less English for it.
    """
    letters = [c for c in (text or '') if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isascii() for c in letters) / len(letters)


def english_share(text):
    """Fraction of the words that are common English function words."""
    words = re.findall(r"[a-zA-Z']+", (text or '').lower())
    if not words:
        return 0.0
    return sum(word in ENGLISH_MARKERS for word in words) / len(words)


def check_input(text):
    """
    Decide whether this text can be accepted at all.

    Returns None when it can, or a Problem explaining why not.

    Note what is NOT checked here: a minimum length. A four-word headline is a
    perfectly good thing to paste, because Step 1 looks headlines up in the
    fact-check table of 44,931 articles and answers from the record — no model,
    no guessing, and the most reliable answer this app gives. Refusing short
    input at the form would shut that path down.

    The minimum only matters once we are about to ask the MODEL, which is a
    separate question and lives in check_long_enough() below.
    """
    text = (text or '').strip()

    if not text:
        return Problem('empty', "Please paste a news headline or article.", {})

    words = word_count(text)

    if words > MAX_WORDS:
        return Problem(
            'too_long',
            f"That is {words:,} words, and the limit is {MAX_WORDS}. Please paste "
            f"the headline and the first few paragraphs.",
            {'words': words})

    # Non-Latin scripts: Urdu, Hindi, Arabic, Chinese, Russian and so on. Also
    # catches input made entirely of emoji or symbols, which has no letters at
    # all and so scores 0.
    latin = latin_share(text)
    if latin < LATIN_FLOOR:
        return Problem(
            'not_english',
            "This tool can only read English. It has only ever been trained on "
            "English news, so on anything else it does not fail politely — it "
            "confidently calls it fake. Please paste an English article.",
            {'latin_share': round(latin, 3)})

    # Latin script but not English: Spanish, French, German, romanised Urdu. Also
    # keyboard mash and strings of digits, none of which contain English function
    # words either.
    #
    # Skipped on very short input. The measurement behind ENGLISH_FLOOR was taken
    # on articles of 25+ words, and a four-word headline can easily contain no
    # function words at all — "Trump Wins Iowa Caucus" scores 0.0 and is
    # perfectly good English. Applying a threshold outside the range it was
    # measured on would reject real headlines.
    if words >= MIN_WORDS:
        english = english_share(text)
        if english < ENGLISH_FLOOR:
            return Problem(
                'not_english_prose',
                "That does not look like English prose. This tool can only read "
                "English news articles — in any other language, or on random "
                "text, it will confidently call the article fake. Please paste "
                "an English article.",
                {'english_share': round(english, 3)})

    return None


def check_long_enough(text):
    """
    Whether there is enough here for the reading model to judge.

    Asked only after Step 1 has failed to find the headline, because this is a
    question about the MODEL, not about the input being acceptable. The model
    reads whatever it is given as one piece and has no way to say "that is not
    enough to go on" — the single word "news" came back as 99.8% fake. So on
    anything shorter than MIN_WORDS the honest answer is that we do not know.
    """
    words = word_count(text)
    if words < MIN_WORDS:
        return Problem(
            'too_short_for_model',
            f"This headline is not in our fact-check records, and at {words} "
            f"word{'s' if words != 1 else ''} there is not enough text for our "
            f"reading model to judge it. Paste at least {MIN_WORDS} words — the "
            f"headline plus the first paragraph or two.",
            {'words': words})
    return None


def describe(text):
    """The measurements for one piece of text. For debugging and for tests."""
    return {
        'words': word_count(text),
        'latin_share': round(latin_share(text), 3),
        'english_share': round(english_share(text), 3),
        'problem': (check_input(text) or Problem(None, None, {})).code,
    }


if __name__ == '__main__':
    CASES = [
        ("English news", "Central bank holds interest rates steady amid inflation "
         "concerns. The central bank left its benchmark interest rate unchanged on "
         "Wednesday, citing persistent inflation and a cooling labour market. "
         "Policymakers voted seven to two to maintain the current range."),
        ("Urdu", "مرکزی بینک نے شرح سود برقرار رکھی۔ مرکزی بینک نے بدھ کو اپنی بنیادی "
         "شرح سود میں کوئی تبدیلی نہیں کی، مسلسل مہنگائی اور سست ہوتی لیبر مارکیٹ کا "
         "حوالہ دیتے ہوئے۔ پالیسی سازوں نے سات کے مقابلے دو ووٹوں سے موجودہ حد برقرار "
         "رکھنے کا فیصلہ کیا جو ماہرین کی توقعات کے مطابق تھا۔"),
        ("Spanish", "El banco central mantuvo sin cambios su tasa de referencia el "
         "miercoles citando la inflacion persistente y un mercado laboral en proceso "
         "de enfriamiento. Los responsables de politica monetaria votaron siete a dos "
         "para mantener el rango actual de tipos de interes segun fuentes."),
        ("keyboard mash", "asdkjfh alskdjfh alskdjfh qwoieuryt zxcmvnb asldkfj "
         "qpwoeiru zmxncbv alsdkjfh qweruiop mnbvcxz lkjhgfds poiuytre wqasdfgh "
         "zxcvbnml plokijuh ygtfrdes wsxedcrf vgybhunj mikolp qazwsxed"),
        ("too short", "Trump says something outrageous again today"),
        ("emoji", " ".join(["😀🎉🚀💀🔥👍🌟⚡"] * 8)),
    ]
    print(f"  {'input':16} {'words':>6} {'latin':>7} {'english':>8}  verdict")
    print("  " + "-" * 62)
    for name, text in CASES:
        d = describe(text)
        print(f"  {name:16} {d['words']:>6} {d['latin_share']:>7.3f} "
              f"{d['english_share']:>8.3f}  {d['problem'] or 'ACCEPTED'}")
