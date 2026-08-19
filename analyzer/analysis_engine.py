"""
analysis_engine.py — Hybrid Fake News Analyzer
=================================================

This is the MOST IMPORTANT file in the project.
It uses a TWO-STEP hybrid approach to classify news:

  Step 1: Database Fact-Check
    Search our database of known articles.
    If the headline matches one exactly, return that dataset's label.

  Step 2: The Reading Model (Fallback)
    If the headline is not in our database, a fine-tuned DistilBERT model reads
    the text and scores how fake it looks. Above the measured cutoff it says
    "Likely Fake", below it "Likely Real" — every article gets an answer.

THE PERCENTAGE SHOWN IS NOT THE MODEL'S CONFIDENCE
---------------------------------------------------
This matters more than anything else in this file. The model's own certainty is
worthless: it routinely reports 99% while being wrong, and on the honest test it
called 42% of fake articles definitely-real with over 95% confidence. Quoting
that number would tell users the opposite of the truth.

So what the app reports instead is the measured reliability of that KIND of
verdict, from tune_cutoff.py:

  when it says "Likely Fake"   it is right about 74% of the time
  when it says "Likely Real"   it is right about 65% of the time

Those come from 3,160 articles in a collection the model never learned from.
About half of them were fake, so a coin toss scores 50% — the "real" verdict is
therefore a weak signal and the "fake" verdict a moderate one. Both are shown
with their own figure attached so nobody mistakes one for the other.

The two are quoted separately on purpose. A single overall accuracy (68%) would
hide the fact that one direction is much more trustworthy than the other.

NOTE ON NUMBERS: nothing here hardcodes how big the database is or how accurate
the model is. The article count is counted live from the database, the model's
training scores are read from reading_model/model_meta.json, and how reliable
each verdict is comes from reading_model/cutoff.json, which tune_cutoff.py
writes at measurement time. Reloading the data, retraining the model or
re-measuring it therefore updates what the app tells users automatically,
instead of leaving a stale figure baked into the text.

If the reading model cannot be loaded — torch not installed, files missing, or
never measured — Step 2 falls back to the old TF-IDF word-counter so the app
keeps working.
"""

import json
import os
import pickle

from .text_cleaning import clean_article_text
from .text_matching import normalize_headline

# ============================================================================
# LOAD THE ML MODEL (loaded ONCE when Django starts, kept in memory)
# ============================================================================

# Get the path to the model files inside the analyzer/ folder
_current_dir = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_current_dir, 'model.pkl')
_vectorizer_path = os.path.join(_current_dir, 'vectorizer.pkl')
_meta_path = os.path.join(_current_dir, 'model_meta.json')

# Load them into memory once
try:
    with open(_model_path, 'rb') as f:
        _model = pickle.load(f)
    with open(_vectorizer_path, 'rb') as f:
        _vectorizer = pickle.load(f)
    _ml_ready = True
except FileNotFoundError:
    _model = None
    _vectorizer = None
    _ml_ready = False
    print("WARNING: model.pkl or vectorizer.pkl not found. ML predictions disabled.")

# The model's measured performance, recorded by train_mega_model.py.
# If it is missing we quote no accuracy figure at all rather than guessing one.
try:
    with open(_meta_path, 'r', encoding='utf-8') as f:
        _model_meta = json.load(f)
except (FileNotFoundError, ValueError):
    _model_meta = None
    print("NOTE: model_meta.json not found. The app will not quote a model accuracy.")

# Whether to strip publisher fingerprints out of the user's text before asking
# for a prediction. This MUST match how the model was trained: feed cleaned text
# to a model that learned from raw text (or the reverse) and its answers quietly
# turn to noise. train_honest_model.py sets 'text_cleaned' in the metadata;
# train_mega_model.py does not, so an older model is fed the raw text it expects.
_clean_before_predict = bool(_model_meta and _model_meta.get('text_cleaned'))


def _measured_performance():
    """
    Return a sentence describing the model's recorded accuracy, or an empty
    string if we have no recorded measurement. Never invents a number.

    Which sentence depends on how the model was measured, because the two are
    not comparable and must not be worded as if they were:

      cross_source  train_honest_model.py held an entire collection of articles
                    back — different outlets, never studied — and examined the
                    model on that. This is the figure worth quoting.

      anything else train_mega_model.py shuffled everything together and hid
                    one article in five. Those hidden articles came from the
                    same piles as the studied ones, so the figure flatters the
                    model. Reported plainly, without the word "accuracy".
    """
    if not _model_meta:
        return ''

    if _model_meta.get('evaluation') == 'cross_source':
        return (
            f" To measure this, the model was tested on "
            f"{_model_meta['test_samples']:,} articles from a separate collection "
            f"it never learned from, and labelled {_model_meta['test_accuracy']}% "
            f"of them correctly (measured {_model_meta['trained_at']})."
        )

    return (
        f" This model was trained on {_model_meta['training_samples']:,} articles and "
        f"correctly labelled {_model_meta['test_accuracy']}% of the "
        f"{_model_meta['test_samples']:,} articles that were held back from training. "
        f"Those held-back articles came from the same collections as the training "
        f"ones, so treat that figure as optimistic (measured "
        f"{_model_meta['trained_at']})."
    )


# ============================================================================
# THE READING MODEL (Step 2's first choice)
# ============================================================================

_reading_dir = os.path.join(_current_dir, 'reading_model')
_reading_meta_path = os.path.join(_reading_dir, 'model_meta.json')
_reading_cutoff_path = os.path.join(_reading_dir, 'cutoff.json')

# None  = not attempted yet
# False = attempted and unavailable, do not try again
# dict  = loaded and ready
_reading_cache = None


def _load_reading_model():
    """
    Load the reading model on first use and keep it in memory afterwards.

    Loaded lazily rather than at import time on purpose. The model is ~270MB and
    takes a few seconds to wake up, and importing this module happens for every
    management command — migrations, shell, the test suite. Paying that cost to
    run a migration would be silly. The trade is that the first analysis after a
    restart is slow; every one after it is not.

    Returns the loaded state, or None if the model cannot be used. Never raises:
    a missing model must degrade to the old word-counter, not break the site.
    """
    global _reading_cache

    if _reading_cache is not None:
        return _reading_cache or None

    _reading_cache = False  # assume failure; overwritten on success

    if not os.path.isdir(_reading_dir):
        print("NOTE: analyzer/reading_model/ not found. "
              "Step 2 will use the old word-counter.")
        return None

    # How reliable each verdict is, measured by tune_cutoff.py. Without this we
    # have no honest figure to show alongside a verdict, so we decline to use the
    # model at all rather than quote its own worthless confidence.
    try:
        with open(_reading_cutoff_path, encoding='utf-8') as f:
            cutoff_data = json.load(f)
    except (FileNotFoundError, ValueError):
        print("NOTE: reading_model/cutoff.json not found. Run "
              "'python tune_cutoff.py' to measure the model before using it. "
              "Step 2 will use the old word-counter meanwhile.")
        return None

    verdict = cutoff_data.get('verdict')
    if not verdict:
        print("NOTE: cutoff.json has no 'verdict' measurements. Re-run "
              "'python tune_cutoff.py'. Step 2 will use the old word-counter.")
        return None

    try:
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
    except ImportError:
        print("NOTE: torch/transformers not installed, so the reading model "
              "cannot run. Step 2 will use the old word-counter. "
              "Install them with: python -m pip install -r requirements.txt")
        return None

    try:
        with open(_reading_meta_path, encoding='utf-8') as f:
            reading_meta = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(_reading_dir)
        model = AutoModelForSequenceClassification.from_pretrained(_reading_dir)
        model.eval()
    except Exception as problem:  # noqa: BLE001 - any failure means fall back
        print(f"NOTE: the reading model failed to load ({problem}). "
              f"Step 2 will use the old word-counter.")
        return None

    _reading_cache = {
        'torch': torch,
        'tokenizer': tokenizer,
        'model': model,
        # Which output column means FAKE, read from the model rather than
        # assumed. If it is ever retrained with the labels the other way round,
        # this keeps working instead of silently inverting every answer.
        'fake_column': model.config.label2id['FAKE'],
        'max_length': reading_meta.get('max_length', 256),
        'meta': reading_meta,
        'verdict': verdict,
    }
    return _reading_cache


def _fake_score(state, text):
    """
    How fake this text looks, from 0.0 to 1.0, according to the reading model.

    no_grad() because we only want an answer — nothing here is learning.
    """
    torch = state['torch']

    encoded = state['tokenizer'](
        [text], truncation=True, padding=True,
        max_length=state['max_length'], return_tensors='pt',
    )

    with torch.no_grad():
        logits = state['model'](**encoded).logits

    return float(torch.softmax(logits, dim=1)[0, state['fake_column']])


def _reading_model_evidence(state):
    """
    One sentence on where the reliability figures came from, built from the
    saved measurements rather than typed in here.
    """
    verdict = state['verdict']
    meta = state['meta']

    return (
        f" That figure comes from testing on {verdict['measured_on']:,} articles "
        f"in a separate collection the model never learned from — the only test "
        f"that describes text like yours, which by definition is not in our "
        f"dataset (model trained {meta.get('trained_at', 'date unrecorded')})."
    )


# ============================================================================
# STEP 1 HELPER: FINDING A KNOWN ARTICLE
# ============================================================================

def best_match(headline):
    """
    Find the best saved article for this headline, or None.

    Both sides of the comparison go through normalize_headline(), so a curly
    versus straight apostrophe, a missing semicolon, odd capitalisation or a
    trailing full stop no longer cause a miss.

    When several saved rows share the same tidied headline they can disagree —
    one source calling it REAL while another calls it FAKE. Picking .first()
    there returns whichever row the database happens to reach first, which is
    not a decision. Instead we order explicitly:

      1. fact-checked rows before publisher-inferred ones (BASIS_RANK)
      2. then oldest row id, purely so repeat lookups stay stable

    so the same input always produces the same verdict, and real fact-checking
    always outranks a guess based on the publisher.
    """
    from .models import KnownArticle

    key = normalize_headline(headline)
    if not key:
        return None

    candidates = KnownArticle.objects.filter(headline_key=key)

    # Sort in Python rather than SQL: the ranking lives in BASIS_RANK on the
    # model, and the candidate list for one exact key is tiny (usually 1 row).
    return min(
        candidates,
        key=lambda a: (KnownArticle.BASIS_RANK.get(a.label_basis, 99), a.id),
        default=None,
    )


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_text(headline, article_text=""):
    """
    Analyze a news headline and optional article text for fake news.

    Uses a two-step hybrid approach:
      1. Database lookup for known articles (tidied headline match)
      2. The reading model for unknown articles, which always returns a verdict

    Parameters:
        headline (str): The news headline
        article_text (str): Optional article body text

    Returns:
        dict with: score, label, confidence, method, explanation

        label is one of 'Likely Fake', 'Likely Real' or 'Unknown'.
        confidence is how often that KIND of verdict has been measured correct —
        NOT how sure the model is. See this module's docstring; the difference
        is the point of the whole design.
    """
    # Import models here to avoid circular imports
    from .models import KnownArticle

    # Count the corpus live, so the figure we report is always the real one
    known_total = KnownArticle.objects.count()

    # ------------------------------------------------------------------
    # STEP 1: Database Fact-Check
    # ------------------------------------------------------------------
    match = best_match(headline)

    if match:
        # We found this exact headline in the loaded corpus
        if match.label == 'FAKE':
            label = 'Likely Fake'
        else:
            label = 'Likely Real'

        return {
            'score': 100 if match.label == 'FAKE' else 0,
            'label': label,
            'confidence': 100.0,
            'method': 'database',
            'matched_article': match,
            'explanation': (
                f"This headline matched an article in our fact-check dataset "
                f"({known_total:,} articles). {match.describe_basis()} "
                f"The 100% figure reflects the certainty of that headline match, "
                f"not an independent check of the article's claims."
            ),
        }

    # ------------------------------------------------------------------
    # STEP 2: The reading model (headline not found in database)
    # ------------------------------------------------------------------
    # Combine headline and article text — the model was trained on the two
    # joined, so it must be asked the same way.
    full_text = f"{headline} {article_text}".strip()

    reading = _load_reading_model()

    if reading:
        # Strip the publisher fingerprints — the newswire dateline, the agency
        # name, image credits, links — so the model has to judge the writing
        # rather than recognise where it was published. The reading model was
        # trained on cleaned text (model_meta records text_cleaned), so this is
        # not optional for it.
        #
        # If cleaning removes everything (the input was nothing but a link, say)
        # keep the original, because a blank string tells the model nothing at
        # all and it would still return a confident-looking percentage.
        cleaned = clean_article_text(full_text)
        model_input = cleaned if cleaned else full_text

        fake_score = _fake_score(reading, model_input)
        verdict = reading['verdict']
        score = round(fake_score * 100)

        # The cutoff is not 0.5. tune_cutoff.py measured that this model leans
        # towards "real" on text it has not seen, so the line that maximises
        # accuracy sits lower — see cutoff.json.
        says_fake = fake_score > verdict['score_above']

        # 'confidence' carries how often THIS KIND of verdict turns out correct,
        # never the model's own certainty. See this module's docstring for why
        # that distinction is the whole point.
        if says_fake:
            label = 'Likely Fake'
            reliability = verdict['fake_precision']
            wrong_share = round(100 - reliability, 1)
            caveat = (
                f"So roughly {wrong_share}% of the articles it calls fake are "
                f"genuinely real news — treat this as a reason to check further, "
                f"not a conclusion."
            )
        else:
            label = 'Likely Real'
            reliability = verdict['real_precision']
            wrong_share = round(100 - reliability, 1)
            caveat = (
                f"This is the weaker of its two verdicts: about {wrong_share}% of "
                f"the articles it calls real are actually fake, so this is not "
                f"clearance to trust the story. Verify it yourself."
            )

        return {
            'score': score,
            'label': label,
            'confidence': reliability,
            'method': 'reading_model',
            'explanation': (
                f"This headline was not in our fact-check dataset "
                f"({known_total:,} articles), so our reading model examined the "
                f"writing itself and scored it {score}% on its fake scale. "
                f"When it reaches this verdict it is correct {reliability}% of "
                f"the time. {caveat}"
                f"{_reading_model_evidence(reading)}"
            ),
        }

    # ------------------------------------------------------------------
    # STEP 2 FALLBACK: the old word-counter, if the reading model is unusable
    # ------------------------------------------------------------------
    if _ml_ready:
        # Only clean when this model was trained that way; see
        # _clean_before_predict above.
        if _clean_before_predict:
            cleaned = clean_article_text(full_text)
            if cleaned:
                full_text = cleaned

        # Convert text to TF-IDF vector (same format the model was trained on)
        text_vector = _vectorizer.transform([full_text])

        # Get prediction (0 = REAL, 1 = FAKE)
        prediction = _model.predict(text_vector)[0]

        # Get confidence percentage from predict_proba
        probabilities = _model.predict_proba(text_vector)[0]
        # probabilities[0] = probability of REAL, probabilities[1] = probability of FAKE
        confidence = round(max(probabilities) * 100, 2)

        if prediction == 1:
            label = 'Likely Fake'
            verdict_word = 'Fake'
            score = int(confidence)
        else:
            label = 'Likely Real'
            verdict_word = 'Real'
            score = int(100 - confidence)

        return {
            'score': score,
            'label': label,
            'confidence': confidence,
            'method': 'ml_prediction',
            'explanation': (
                f"This headline was not in our fact-check dataset ({known_total:,} articles), "
                f"so our machine-learning model analyzed the text instead. It is "
                f"{confidence}% confident the article is {verdict_word}."
                f"{_measured_performance()}"
            ),
        }

    # ------------------------------------------------------------------
    # FALLBACK: Neither database nor ML available
    # ------------------------------------------------------------------
    return {
        'score': 0,
        'label': 'Unknown',
        'confidence': 0.0,
        'method': 'unavailable',
        'explanation': 'Analysis system is not available. Please ensure the model files are present.',
    }

