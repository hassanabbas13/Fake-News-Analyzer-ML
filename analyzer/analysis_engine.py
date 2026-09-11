"""
The hybrid analyzer. Three steps, in order, first answer wins.

  1. Database fact-check. Two exact lookups against the corpus — tidied headline,
     then the article's tidied opening words. Two keys because headlines are
     fragile: drop "(VIDEO)" off the end and a story we hold a fact-check for
     stops being recognisable.
  2. The reading model, a fine-tuned DistilBERT. Text too short to read is turned
     away first: asked about the single word "news" it answered 99.8% fake.
  3. The web search, where Step 2 is unreliable or the reader asked. Shown BESIDE
     the verdict, never merged into it. See web_check.py.

THE PERCENTAGE SHOWN IS NOT THE MODEL'S CONFIDENCE. The model's own certainty is
worthless — it routinely reports 99% while being wrong — so what the app reports
is the measured reliability of that KIND of verdict, read live from cutoff.json.
The two directions are quoted separately, because one overall figure would hide
it whenever one verdict became much weaker than the other.

Do not write any of those figures into this file. The article count is counted
live, training scores come from model_meta.json, per-verdict reliability from
cutoff.json. An earlier version of this docstring quoted 74% and 65% and went on
quoting them through two retrainings after they stopped being true.

If the reading model cannot be loaded, Step 2 falls back to the old TF-IDF
word-counter and the result page says which one answered.
"""

import json
import os
import pickle

from .text_cleaning import clean_article_text
from .text_matching import body_key, normalize_headline
from .input_check import check_long_enough
from .web_check import check_online, should_check

# --- The old TF-IDF word-counter: Step 2's fallback -------------------------
# Loaded once at import and kept in memory.

_current_dir = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_current_dir, 'model.pkl')
_vectorizer_path = os.path.join(_current_dir, 'vectorizer.pkl')
_meta_path = os.path.join(_current_dir, 'model_meta.json')

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

# Missing metadata means we quote no accuracy figure at all rather than guess one.
try:
    with open(_meta_path, 'r', encoding='utf-8') as f:
        _model_meta = json.load(f)
except (FileNotFoundError, ValueError):
    _model_meta = None
    print("NOTE: model_meta.json not found. The app will not quote a model accuracy.")

# Must match how the model was trained, or its answers quietly turn to noise.
# train_honest_model.py sets 'text_cleaned'; train_mega_model.py does not.
_clean_before_predict = bool(_model_meta and _model_meta.get('text_cleaned'))


def _measured_performance():
    """
    A sentence describing the word-counter's recorded accuracy, or '' if nothing
    was measured. Never invents a number.

    The wording depends on how it was measured, because the two are not comparable.
    A cross_source figure held back a whole collection of unseen outlets and is the
    one worth quoting; anything else shuffled everything together, so the held-back
    articles came from the same piles and the figure flatters the model.
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


# --- The reading model: Step 2's first choice -------------------------------

_reading_dir = os.path.join(_current_dir, 'reading_model')
_reading_meta_path = os.path.join(_reading_dir, 'model_meta.json')
_reading_cutoff_path = os.path.join(_reading_dir, 'cutoff.json')

# None = not attempted yet, False = unavailable so stop trying, dict = ready
_reading_cache = None


def _load_reading_model():
    """
    Load the reading model on first use and keep it in memory afterwards. Lazy on
    purpose: it is ~270MB and this module is imported by every management command,
    so the cost falls on the first analysis instead of on migrations and tests.

    Returns the loaded state, or None if it cannot be used. Never raises: a missing
    model must degrade to the word-counter, not break the site.
    """
    global _reading_cache

    if _reading_cache is not None:
        return _reading_cache or None

    _reading_cache = False  # assume failure; overwritten on success

    if not os.path.isdir(_reading_dir):
        print("NOTE: analyzer/reading_model/ not found. "
              "Step 2 will use the old word-counter.")
        return None

    # Without cutoff.json there is no honest figure to show beside a verdict, so
    # decline the model rather than quote its own worthless confidence.
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
        # Read from the model, so a retrain with the labels the other way round
        # keeps working instead of inverting every answer.
        'fake_column': model.config.label2id['FAKE'],
        'max_length': reading_meta.get('max_length', 256),
        'meta': reading_meta,
        'verdict': verdict,
    }
    return _reading_cache


def _fake_score(state, text):
    """How fake this text looks, 0.0 to 1.0, according to the reading model."""
    torch = state['torch']

    encoded = state['tokenizer'](
        [text], truncation=True, padding=True,
        max_length=state['max_length'], return_tensors='pt',
    )

    # no_grad because we only want an answer — nothing here is learning.
    with torch.no_grad():
        logits = state['model'](**encoded).logits

    return float(torch.softmax(logits, dim=1)[0, state['fake_column']])


# --- Step 1: finding a known article ----------------------------------------

def _pick_best(candidates):
    """
    Choose between saved rows that share a lookup key. They can disagree — one
    source calling a story REAL, another FAKE — and .first() would return whichever
    row the database reached first, which is not a decision. Instead: fact-checked
    rows before publisher-inferred ones, then oldest id so lookups stay stable.

    Sorted in Python because the ranking lives in BASIS_RANK and the candidate list
    for one exact key is tiny.
    """
    from .models import KnownArticle

    return min(
        candidates,
        key=lambda a: (KnownArticle.BASIS_RANK.get(a.label_basis, 99), a.id),
        default=None,
    )


def best_match(headline):
    """The best saved article for this headline, or None. Both sides go through
    normalize_headline(), so a curly apostrophe or a trailing full stop no longer
    causes a miss."""
    from .models import KnownArticle

    key = normalize_headline(headline)
    if not key:
        return None

    return _pick_best(KnownArticle.objects.filter(headline_key=key))


def best_match_by_body(article_text):
    """
    The second door into the corpus, for when a headline has been edited but the
    body pasted underneath is still a perfect copy.

    Still an EXACT match on tidied text, not a similarity score, which is what lets
    the caller keep reporting 100% honestly. None when the body is too short to key.
    """
    from .models import KnownArticle

    key = body_key(article_text)
    if not key:
        return None

    return _pick_best(KnownArticle.objects.filter(body_key=key))


def analyze_text(headline, article_text="", search_web=False):
    """
    Analyze a headline and optional article body.

    search_web adds an online check on a confident score; the unsure band still
    triggers one on its own when it is False. Ignored when Step 1 answers.

    Returns score, label, confidence, method and explanation. label is 'Likely
    Fake', 'Likely Real', 'Not Sure' or 'Unknown'. confidence is how often that KIND
    of verdict is measured correct, NOT how sure the model is.
    """
    # Imported here to avoid a circular import.
    from .models import KnownArticle

    # Counted live, so the figure reported is always the real one.
    known_total = KnownArticle.objects.count()

    # --- Step 1: database fact-check ---
    # Headline first, then the body, because headlines get edited and bodies mostly
    # do not. BOTH are exact matches, so the 100% below stays honest either way.
    match = best_match(headline)
    matched_on = 'headline'

    if not match:
        # Two texts to try, because views.analyze() already guessed where the
        # headline ends by splitting on the first newline. Paste a body alone and
        # that guess strands the real opening words in `headline`, so try the body,
        # then the whole paste back together. This widens WHERE we look, not how
        # loosely we compare.
        for candidate in (article_text, f"{headline}\n{article_text}"):
            match = best_match_by_body(candidate)
            if match:
                matched_on = 'body'
                break

    if match:
        if match.label == 'FAKE':
            label = 'Likely Fake'
        else:
            label = 'Likely Real'

        # Only on a body match, where the headline shown further down the page is
        # not the one the reader pasted.
        if matched_on == 'body':
            note = (
                " Matched on the article text — the headline you pasted is not "
                "the one we have on file."
            )
        else:
            note = ''

        return {
            'score': 100 if match.label == 'FAKE' else 0,
            'label': label,
            'confidence': 100.0,
            'method': 'database',
            'matched_article': match,
            'matched_on': matched_on,
            # The three things the rest of the page does not already say: dataset
            # size, who labelled it and on what grounds, what the 100% measures.
            'explanation': (
                f"Found in our dataset of {known_total:,} articles, "
                f"{match.describe_basis()}.{note} "
                f"The 100% is how certain the match is."
            ),
        }

    # --- Step 2: the reading model ---
    # Refusing short text happens HERE, not in the form, because a bare headline is
    # a fine thing to paste when Step 1 can look it up.
    short = check_long_enough(f"{headline} {article_text}")
    if short:
        return {
            'score': 50,
            'label': 'Not Sure',
            'confidence': None,
            'method': 'too_short',
            'explanation': short.message,
        }

    # Joined, because the model was trained on the two joined.
    full_text = f"{headline} {article_text}".strip()

    reading = _load_reading_model()

    if reading:
        # Strip publisher fingerprints so the model judges the writing, not where
        # it was published. Not optional: it was trained on cleaned text. Keep the
        # original if cleaning removes everything, since a blank string tells the
        # model nothing and would still come back with a confident percentage.
        cleaned = clean_article_text(full_text)
        model_input = cleaned if cleaned else full_text

        fake_score = _fake_score(reading, model_input)
        verdict = reading['verdict']
        score = round(fake_score * 100)

        # Not 0.5: tune_cutoff.py measured this model leaning towards "real" on
        # unseen text, so the line that maximises accuracy sits lower.
        says_fake = fake_score > verdict['score_above']

        # How often THIS KIND of verdict turns out correct, never the model's own
        # certainty.
        if says_fake:
            label = 'Likely Fake'
            reliability = verdict['fake_precision']
        else:
            label = 'Likely Real'
            reliability = verdict['real_precision']

        result = {
            'score': score,
            'label': label,
            'confidence': reliability,
            'method': 'reading_model',
            # Three facts, one sentence: not in the dataset, how often this verdict
            # holds up, where that number came from.
            'explanation': (
                f"Not in our dataset, so the reading model read the article "
                f"itself. Verdicts like this are right {reliability}% of the "
                f"time, measured on {verdict['measured_on']:,} articles it never "
                f"saw in training."
            ),
        }

        # --- Step 3: ask the web, where the model is unreliable or on request ---
        # search_web WIDENS this gate and never narrows it: the unsure band still
        # fires when the box is unticked. Deliberate, because nobody sees the raw
        # score, so a switch that could turn the band off would quietly remove the
        # search from the only 15% that needed it.
        #
        # Additive too — the verdict and reliability above are untouched by whatever
        # comes back. Those are measured; the web check is not. check_online() never
        # raises; any failure comes back as status 'unavailable'.
        model_unsure = should_check(fake_score)
        if model_unsure or search_web:
            web = check_online(headline, article_text)
            # The page must not imply the model was unsure when it was not. Ticking
            # the box on an article scoring 0.0002 has to read as "you asked".
            web['reason'] = 'unsure' if model_unsure else 'requested'
            result['web'] = web

        return result

    # --- Step 2 fallback: the old word-counter ---
    if _ml_ready:
        # Only clean when this model was trained that way.
        if _clean_before_predict:
            cleaned = clean_article_text(full_text)
            if cleaned:
                full_text = cleaned

        text_vector = _vectorizer.transform([full_text])
        prediction = _model.predict(text_vector)[0]   # 0 = REAL, 1 = FAKE

        probabilities = _model.predict_proba(text_vector)[0]
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
                f"Not in our dataset, so the word-counting model read the "
                f"article itself. It is {confidence}% sure this is "
                f"{verdict_word}.{_measured_performance()}"
            ),
        }

    # --- Nothing available at all ---
    return {
        'score': 0,
        'label': 'Unknown',
        'confidence': 0.0,
        'method': 'unavailable',
        'explanation': 'Analysis system is not available. Please ensure the model files are present.',
    }
