"""
Is this actually an English news article?

The model has no way to say "I cannot read this", so anything unfamiliar lands on
the fake side. Measured 2 Sep 2026: English real news scored 0.0003, the same
story in Urdu 0.9866, Hindi/Arabic/mash 0.99+, Spanish 0.3498, the single word
"news" 0.9980. All of those would have been shown as "Likely Fake" beside a
reliability figure measured on English news, which is inventing a statistic.

Both thresholds were measured against the 6,147 McIntire articles of 25+ words:

    LATIN_FLOOR 0.85     Letters that are ordinary A-Z. English news never fell
                         below 0.973; Urdu, Hindi, Arabic, Chinese, Russian all
                         score 0.000.
    ENGLISH_FLOOR 0.10   Words that are common English function words. English
                         median 0.359, Spanish 0.015, French 0.033, German 0.041,
                         romanised Urdu 0.019 — so it catches Latin-script
                         non-English, plus mash and emoji, which have none
                         either. Cost 4 real articles out of 6,147.

MIN_WORDS 25 because a handful of words is not an article. MAX_WORDS 400 is a
product decision; the model reads only the first ~200 words regardless.
"""

import re
from collections import namedtuple

MIN_WORDS = 25
MAX_WORDS = 400

# See the module docstring for how both of these were measured.
LATIN_FLOOR = 0.85
ENGLISH_FLOOR = 0.10

# Function words, not topic words: every piece of English prose is full of these
# regardless of subject, so they signal language rather than subject.
ENGLISH_MARKERS = frozenset("""
the of and to in a is that for it on was with as at by from be this
have or an not are but had has were they which you we their all been more will
would there can who said one about when what so its if out up says
""".split())

# `code` for tests and logs, `message` for the reader, `detail` for the measured
# numbers, so a template can say how far off the input was.
Problem = namedtuple('Problem', 'code message detail')


def word_count(text):
    """Words, counted the way a person would count them."""
    return len(re.findall(r"[\w'-]+", text or ''))


def latin_share(text):
    """Fraction of the letters that are ordinary A-Z. Only letters count: a
    sentence with a lot of commas is not less English for it."""
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
    None when the text can be accepted, or a Problem explaining why not.

    Deliberately does NOT check a minimum length. A four-word headline is worth
    pasting: Step 1 looks it up and answers from the record, which is the most
    reliable answer this app gives. The minimum only matters once we are about to
    ask the MODEL — see check_long_enough().
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

    # Non-Latin scripts, and pure emoji or symbols, which have no letters and so
    # score 0.
    latin = latin_share(text)
    if latin < LATIN_FLOOR:
        return Problem(
            'not_english',
            "This tool can only read English. It has only ever been trained on "
            "English news, so on anything else it does not fail politely — it "
            "confidently calls it fake. Please paste an English article.",
            {'latin_share': round(latin, 3)})

    # Latin script but not English, plus keyboard mash and digits. Skipped on short
    # input because ENGLISH_FLOOR was measured on 25+ words: "Trump Wins Iowa
    # Caucus" scores 0.0 and is perfectly good English.
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
    Whether there is enough here for the reading model to judge. Asked only after
    Step 1 fails, because it is a question about the MODEL, not about the input
    being acceptable. The model cannot say "not enough to go on" — the single word
    "news" came back 99.8% fake — so below MIN_WORDS we decline.
    """
    words = word_count(text)
    if words < MIN_WORDS:
        return Problem(
            'too_short_for_model',
            f"This headline is not in our fact-check records, and at {words} "
            f"word{'s' if words != 1 else ''} there is not enough text for our "
            f"reading model to judge it. Paste at least {MIN_WORDS} words.",
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
