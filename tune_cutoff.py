"""
tune_cutoff.py — Pick a better cutoff for the reading model, honestly
======================================================================

What this fixes
---------------
The reading model in analyzer/reading_model/ does not answer "fake" or "real".
It gives every article a score between 0 and 1, where higher means "more
likely fake". Something has to turn that score into a verdict, and right now
that something is the number 0.5, which nobody chose. It is just the default.

On the honest test (taught on ISOT, examined on McIntire) that default gives:

    81.8% of genuinely real articles called real
    52.4% of genuinely fake articles called fake

Those should be closer together. The model is leaning towards "real" whenever
it is unsure, so it waves through nearly half the fake articles. Lowering the
cutoff makes it quicker to say "fake": that wins back fake articles and costs
some real ones. Somewhere in between is a better trade than 0.5.

Nothing is retrained here. The model is untouched. We are only choosing where
to draw the line on scores it already produces.

Why the data is split in half
-----------------------------
Choosing the cutoff on the same articles used to report the final number would
be cheating — of course the line looks good on the articles used to draw it.
So the McIntire articles are split in two:

    half A  used to choose the cutoff       (never reported)
    half B  used to report the final number (never influences the choice)

The number printed at the end comes only from half B, which had no say in the
decision. That is the number worth quoting.

One honest caveat: the model weights never saw any McIntire article, but the
cutoff is one number fitted on half A. So this is very slightly less pure than
"never saw this collection at all". It is a standard and mild step, and it is
disclosed here so nobody has to wonder.

Usage
-----
  python tune_cutoff.py              score the articles, then tune
  python tune_cutoff.py --retune     reuse saved scores, just tune again (instant)

Scoring 6,000+ articles on a processor takes a few minutes, so the scores are
cached in analyzer/reading_model/mcintire_scores.csv. --retune skips straight to
the interesting part.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# Setup Django first — the shared text helpers live inside the analyzer app.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fake_news_project.settings')
import django
django.setup()

# Reuse the loading and de-duplicating from train_honest_model.py rather than
# copying it. If that file's rules ever change, this script follows along
# instead of quietly testing on a different set of articles.
from train_honest_model import (drop_cross_collection_duplicates,
                                load_collections)

from analyzer.text_cleaning import clean_article_text

MODEL_DIR = os.path.join('analyzer', 'reading_model')
SCORES_PATH = os.path.join(MODEL_DIR, 'mcintire_scores.csv')
CUTOFF_PATH = os.path.join(MODEL_DIR, 'cutoff.json')

# Half A chooses, half B reports. Fixed seed so the split is the same every run
# — otherwise the "final" number would wobble every time you ran this.
SPLIT_SEED = 42
BATCH_SIZE = 16

# How reliable a "fake" flag has to be before the app is allowed to show it.
#
# This is a product decision, not a measurement: we will not tell a user an
# article is fake unless that claim has been measured right at least 3 times in
# 4. Raise it and the app flags fewer articles but is wrong less often; lower it
# and the reverse. It is written down here rather than buried in the app so the
# trade-off is visible and arguable.
FLAG_PRECISION_FLOOR = 0.75

# Below this many flagged articles an accuracy estimate is too shaky to trust,
# so such a band is rejected however good it looks.
MIN_FLAGGED = 50

# The widest slice of articles we are willing to send to an outside web search.
# Every article inside the "unsure" band costs one API call and the free Gemini
# allowance is 500 grounded searches a day, so this is a spending limit as much
# as a statistical one.
UNSURE_MAX_COVERAGE = 0.20

# Gemini's free grounded-search allowance per day, as documented on
# 24 Aug 2026. Only used to print a rough capacity estimate.
FREE_SEARCHES_PER_DAY = 500

# How reliable the verdict has to be before we are willing to let it stand
# alone. Below this, the app should go and look the story up instead of leaning
# on a guess. A product decision, not a measurement: 75% means we accept being
# wrong 1 time in 4 without seeking a second opinion, which is the same floor
# FLAG_PRECISION_FLOOR uses for a different question.
UNSURE_ACCURACY_FLOOR = 75.0

# Candidate unsure bands, as (low, high) score pairs. Deliberately a short fixed
# list rather than a fine search: this picks ONE band, and we would rather it be
# a round explicable number than the winner of a thousand-way contest fitted to
# noise in a few dozen articles.
UNSURE_CANDIDATES = [
    (0.01, 0.99), (0.05, 0.99), (0.05, 0.95),
    (0.10, 0.90), (0.20, 0.80), (0.30, 0.70),
]


# ============================================================================
# STEP 1 — GET THE MODEL'S SCORE FOR EVERY McINTIRE ARTICLE
# ============================================================================

def build_test_set():
    """
    The McIntire articles, prepared exactly as the model was trained to expect:
    headline and body joined, publisher fingerprints stripped out.
    """
    print("Reading the CSV files ...")
    df = load_collections()
    df = drop_cross_collection_duplicates(df)

    mcintire = df[df['collection'] == 'McIntire'].copy()

    print("Stripping publisher fingerprints ...")
    mcintire['cleaned'] = mcintire['raw'].map(clean_article_text)
    mcintire = mcintire[mcintire['cleaned'].str.len() > 0].reset_index(drop=True)

    fake = int((mcintire['label_binary'] == 1).sum())
    print(f"  {len(mcintire):,} articles  ({fake:,} fake / "
          f"{len(mcintire) - fake:,} real)")

    return mcintire


def score_articles(texts):
    """
    Run every article through the model and return its "how fake is this"
    score, from 0 to 1.

    Slow on a processor, so it prints progress. Nothing here changes the model;
    torch.no_grad() tells it we only want answers, not learning.
    """
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    with open(os.path.join(MODEL_DIR, 'model_meta.json'), encoding='utf-8') as f:
        max_length = json.load(f)['max_length']

    print(f"\nLoading the model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Confirm which output column means "fake" rather than assuming it is the
    # second one. If the model was ever retrained with the labels the other way
    # round, this catches it instead of silently inverting every answer.
    fake_column = model.config.label2id['FAKE']

    print(f"Scoring {len(texts):,} articles "
          f"(reading {max_length} word-pieces of each) ...")

    scores = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        encoded = tokenizer(batch, truncation=True, padding=True,
                            max_length=max_length, return_tensors='pt')

        with torch.no_grad():
            logits = model(**encoded).logits

        probabilities = torch.softmax(logits, dim=1)[:, fake_column]
        scores.extend(probabilities.tolist())

        done = start + len(batch)
        if done % (BATCH_SIZE * 20) == 0 or done == len(texts):
            print(f"  {done:,} / {len(texts):,}")

    return scores


def load_or_build_scores(retune):
    """Reuse cached scores when asked, otherwise run the model."""
    if retune:
        if not os.path.exists(SCORES_PATH):
            raise SystemExit(
                f"--retune needs {SCORES_PATH}, which does not exist yet.\n"
                f"Run 'python tune_cutoff.py' once without --retune first."
            )
        print(f"Reusing saved scores from {SCORES_PATH}")
        return pd.read_csv(SCORES_PATH)

    mcintire = build_test_set()
    mcintire['score'] = score_articles(mcintire['cleaned'].tolist())

    saved = mcintire[['title', 'label_binary', 'score']]
    saved.to_csv(SCORES_PATH, index=False)
    print(f"\nScores saved to {SCORES_PATH} — use --retune to skip this next time.")

    return saved


# ============================================================================
# STEP 2 — MEASURE A CUTOFF
# ============================================================================

def measure(scores, truth, cutoff):
    """
    How a given cutoff performs. Returns overall accuracy plus one score per
    category, because the overall figure hides the interesting part.

    'real_recall' means: out of the articles that genuinely were real, what
    share did it correctly call real. 'fake_recall' is the same for fake ones.
    """
    predicted_fake = scores >= cutoff
    actually_fake = truth == 1

    return {
        'cutoff': cutoff,
        'accuracy': float((predicted_fake == actually_fake).mean() * 100),
        'real_recall': float((~predicted_fake[~actually_fake]).mean()),
        'fake_recall': float(predicted_fake[actually_fake].mean()),
    }


# ============================================================================
# STEP 3 — FIND THE BAND WHERE A "FAKE" FLAG IS ACTUALLY WORTH SHOWING
# ============================================================================

def cutoff_candidates(scores, count=2000):
    """
    Candidate cutoffs taken from the scores themselves.

    A fixed ladder -- 0.05, 0.06, ... 0.95 -- only works if the scores are
    spread across that range. A confident model squashes nearly everything up
    against 1.0: this one puts half its REAL articles above 0.9997. Every rung
    of the old ladder then sits below the entire distribution, so the search
    reports the same near-useless split at every step and settles for the top
    rung by default. That is how a model worth 76% got measured at 62%.

    Taking candidates from the actual scores means the search always looks
    where the articles are, however squashed they happen to be. The coarse
    ladder stays in the mix so a well-spread model behaves exactly as before.
    """
    inner = np.quantile(scores, np.linspace(0.001, 0.999, count))
    return np.unique(np.concatenate([inner, np.arange(0.05, 0.96, 0.05)]))


def fmt_cutoff(value):
    """
    Print a cutoff with enough decimals to mean something.

    '%.2f' turns 0.99993793 into '1.00', which makes two very different cutoffs
    look identical and hides the whole point.
    """
    if 0.001 < value < 0.999:
        return f"{value:.3f}"
    return f"{value:.8f}"


def find_flag_band(scores, truth):
    """
    Find the lowest score above which "this is fake" is right at least
    FLAG_PRECISION_FLOOR of the time. Returns None if no band qualifies.

    Why this exists rather than just using the best cutoff above: a single
    cutoff forces a verdict on every article, and we measured that the model's
    "real" verdicts are barely better than a coin toss. Splitting the score
    range into "reliable enough to report" and "say nothing" lets the app use
    the model only where it has earned trust.

    Lowest qualifying score, not highest, because that covers the most articles
    while still clearing the reliability floor.
    """
    qualifying = None

    # Highest scores first. Candidates come from the distribution for the same
    # reason as in cutoff_candidates -- a fixed ladder cannot reach a saturated
    # model, and this band search was silently suffering from it too.
    for threshold in cutoff_candidates(scores, count=400)[::-1]:
        flagged = scores > threshold
        if flagged.sum() < MIN_FLAGGED:
            continue

        precision = truth[flagged].mean()
        if precision >= FLAG_PRECISION_FLOOR:
            qualifying = float(threshold)
        elif qualifying is not None:
            # Reliability has dropped through the floor and will not recover as
            # we go lower, so the previous threshold is the answer.
            break

    return qualifying


def describe_band(scores, truth, threshold):
    """
    How the band performs: how often a flag is right, how many articles it
    covers, and how trustworthy the leftovers are.
    """
    flagged = scores > threshold

    return {
        # NEVER round a cutoff. This model's thresholds live at 0.99993797,
        # and round(_, 2) makes that 1.0 -- a line no article can cross, so
        # the app would answer REAL to everything. The percentages below are
        # rounded because they are only ever displayed; this one is used.
        'score_above': float(threshold),
        # Of the articles we would flag as fake, how many genuinely are.
        'precision': round(float(truth[flagged].mean()) * 100, 1),
        # What share of all articles get a verdict at all.
        'coverage': round(float(flagged.mean()) * 100, 1),
        # Of everything below the band, how many are genuinely real. This is the
        # number that justifies saying nothing: if it is near 50% the model
        # knows nothing useful about these articles.
        'below_band_real': round(float((1 - truth[~flagged]).mean()) * 100, 1),
        'below_band_coverage': round(float((~flagged).mean()) * 100, 1),
        'measured_on': int(len(scores)),
    }


def find_unsure_band(scores, truth, cutoff):
    """
    Find the score range where this model's verdict is not worth trusting.

    Why this exists: the model is bimodal. It puts most articles hard against 0
    or hard against 1 and is right about 92% of the time on those, but the
    handful landing in the middle it gets barely better than a coin flip. Those
    middle articles are exactly the ones worth spending an outside web search
    on; the rest are exactly the ones not to waste an API call on.

    The band is MEASURED rather than assumed, for the same reason the cutoff is:
    it belongs to this particular set of weights. Retrain and the middle moves.
    A band typed into the app by hand would quietly go stale, and the app would
    start paying for searches on articles it already knew the answer to.

    Chosen as the WIDEST band whose verdicts are still under
    UNSURE_ACCURACY_FLOOR reliable — i.e. help as many doubtful articles as we
    can, while only calling doubtful what really is.

    An earlier version maximised the gap between inside and outside accuracy
    instead. That sounds right and is not: it rewards the narrowest, purest band,
    so it picked a range covering 3% of articles and left the other two thirds of
    the unreliable ones unhelped. Optimise for how many users get a better
    answer, not for how bad the worst slice looks. The free allowance is 500
    searches a day and even the widest candidate here needs about 150 per 1,000
    articles, so coverage is what is scarce, not budget.

    Returns None if nothing qualifies, which is the honest answer for a model
    that is reliable everywhere.
    """
    said_fake = (scores >= cutoff).astype(int)
    correct = said_fake == truth

    best = None
    for low, high in UNSURE_CANDIDATES:
        inside = (scores > low) & (scores < high)

        # Too few articles to conclude anything, or so many that "unsure" would
        # be the normal case and nearly every analysis would cost a search.
        if inside.sum() < MIN_FLAGGED or inside.mean() > UNSURE_MAX_COVERAGE:
            continue

        in_acc = float(correct[inside].mean()) * 100
        out_acc = float(correct[~inside].mean()) * 100

        # Not an unsure band if the verdict holds up inside it.
        if in_acc >= UNSURE_ACCURACY_FLOOR:
            continue

        # Nor if the model is no better outside than in — then the band is not
        # isolating anything, it is just a slice of a uniformly weak model.
        if out_acc <= in_acc:
            continue

        coverage = float(inside.mean())
        if best is None or coverage > best['coverage_fraction']:
            best = {'low': low, 'high': high, 'coverage_fraction': coverage,
                    'accuracy_inside': in_acc, 'accuracy_outside': out_acc}
    return best


def describe_unsure_band(scores, truth, cutoff, band):
    """The band's figures on the half that had no say in choosing it."""
    said_fake = (scores >= cutoff).astype(int)
    correct = said_fake == truth
    inside = (scores > band['low']) & (scores < band['high'])
    return {
        'low': float(band['low']),
        'high': float(band['high']),
        'coverage': round(float(inside.mean()) * 100, 1),
        'accuracy_inside': round(float(correct[inside].mean()) * 100, 1),
        'accuracy_outside': round(float(correct[~inside].mean()) * 100, 1),
        'searches_per_1000_articles': int(round(float(inside.mean()) * 1000)),
        'measured_on': int(len(scores)),
    }


def describe_verdict(scores, truth, cutoff):
    """
    How the app performs when it is required to answer about EVERY article.

    This is what analysis_engine.py actually uses. The key numbers are the two
    'precision' figures, because they answer the only question a user really
    asks: when this thing tells me something, how often is it right?

    Note that precision is NOT the same as the recall figures printed further
    up. Recall asks "of all the fake articles, how many did it catch". Precision
    asks "of everything it called fake, how much really was". A user reading a
    verdict cares about the second one, so that is what gets shown to them.
    """
    said_fake = scores > cutoff
    said_real = ~said_fake

    return {
        'score_above': float(cutoff),   # never rounded -- see find_flag_band
                                        # for what rounding a cutoff destroys
        'accuracy': round(float((said_fake == (truth == 1)).mean()) * 100, 1),
        'fake_precision': round(float(truth[said_fake].mean()) * 100, 1),
        'fake_coverage': round(float(said_fake.mean()) * 100, 1),
        'real_precision': round(float((1 - truth[said_real]).mean()) * 100, 1),
        'real_coverage': round(float(said_real.mean()) * 100, 1),
        'measured_on': int(len(scores)),
    }


# ============================================================================
# THE RUN
# ============================================================================

def run(retune):
    data = load_or_build_scores(retune)

    scores = data['score'].to_numpy()
    truth = data['label_binary'].to_numpy()

    # Split in half, keeping the fake/real balance the same in both halves so
    # neither half is an easier exam than the other.
    from sklearn.model_selection import train_test_split
    index_a, index_b = train_test_split(
        np.arange(len(scores)), test_size=0.5,
        random_state=SPLIT_SEED, stratify=truth,
    )

    scores_a, truth_a = scores[index_a], truth[index_a]
    scores_b, truth_b = scores[index_b], truth[index_b]

    print(f"\nhalf A — choosing the cutoff : {len(index_a):,} articles")
    print(f"half B — the final number     : {len(index_b):,} articles")

    # Candidates drawn from half A's own scores, so the search reaches wherever
    # this model puts them. Still chosen on half A alone.
    candidates = [measure(scores_a, truth_a, c)
                  for c in cutoff_candidates(scores_a)]

    best = max(candidates, key=lambda r: r['accuracy'])

    # The cutoff where the two categories are treated most evenly. Often more
    # useful for a fake-news app than raw accuracy: a detector that misses half
    # the fakes is not much of a detector, even if the total looks fine.
    fairest = min(candidates,
                  key=lambda r: abs(r['real_recall'] - r['fake_recall']))

    # ---- show the shape of the trade-off ----
    print(f"\n{'=' * 68}")
    print("  WHAT THE CUTOFF DOES  (measured on half A)")
    print(f"{'=' * 68}")
    print(f"  {'cutoff':>12}  {'accuracy':>9}  {'real right':>11}  {'fake right':>11}")
    print("  " + "-" * 50)

    # A dozen rows spread across the candidates, plus the two interesting ones.
    # Picked by POSITION, not by value: the candidates are no longer evenly
    # spaced numbers, so arithmetic on the value can no longer choose them.
    show = set(np.linspace(0, len(candidates) - 1, 12).round().astype(int).tolist())
    show.add(candidates.index(best))
    show.add(candidates.index(fairest))

    for i in sorted(show):
        row = candidates[i]
        mark = ''
        if row is best:
            mark = '  <- best accuracy'
        elif row is fairest:
            mark = '  <- most even'
        print(f"  {fmt_cutoff(row['cutoff']):>12}  {row['accuracy']:>8.2f}%  "
              f"{row['real_recall']:>10.1%}  {row['fake_recall']:>10.1%}{mark}")

    # ---- the honest result, on the half that had no say ----
    default = measure(scores_b, truth_b, 0.5)
    tuned = measure(scores_b, truth_b, best['cutoff'])
    even = measure(scores_b, truth_b, fairest['cutoff'])

    print(f"\n{'=' * 68}")
    print(f"  THE RESULT — measured on half B ({len(index_b):,} articles that")
    print("  had no influence on the cutoff)")
    print(f"{'=' * 68}")
    print(f"  {'':22}  {'accuracy':>9}  {'real right':>11}  {'fake right':>11}")
    print("  " + "-" * 60)
    for name, row in [('old cutoff (0.50)', default),
                      (f"best accuracy ({fmt_cutoff(best['cutoff'])})", tuned),
                      (f"most even ({fmt_cutoff(fairest['cutoff'])})", even)]:
        print(f"  {name:22}  {row['accuracy']:>8.2f}%  "
              f"{row['real_recall']:>10.1%}  {row['fake_recall']:>10.1%}")

    change = tuned['accuracy'] - default['accuracy']
    print(f"\n  Moving the cutoff changed accuracy by {change:+.2f} points.")

    if tuned['accuracy'] >= 70:
        print("  That clears 70% — on articles from a collection the model")
        print("  never learned from.")
    else:
        short = 70 - tuned['accuracy']
        print(f"  Still {short:.2f} points short of 70%. Closing that gap means")
        print("  better data, not a better cutoff — the cutoff is now optimal.")

    # ---- what the app will actually report ----
    # The app answers about every article, so these are the figures users see.
    verdict = describe_verdict(scores_b, truth_b, best['cutoff'])

    print(f"\n{'=' * 68}")
    print("  WHAT USERS WILL SEE  (the app answers about every article)")
    print(f"{'=' * 68}")
    print(f"  cutoff {fmt_cutoff(verdict['score_above'])}, overall accuracy "
          f"{verdict['accuracy']}%")
    print()
    print(f"  says LIKELY FAKE on {verdict['fake_coverage']}% of articles"
          f"  ->  right {verdict['fake_precision']}% of the time")
    print(f"  says LIKELY REAL on {verdict['real_coverage']}% of articles"
          f"  ->  right {verdict['real_precision']}% of the time")
    print()
    print("  Those two percentages are what the result page quotes. They are")
    print("  'when it says this, how often is it correct' — not the model's own")
    print("  confidence, which is routinely 99% while being wrong.")

    # ---- the band where a "fake" flag is worth showing ----
    # Chosen on half A, measured on half B, same discipline as the cutoff above.
    #
    # Kept as a recorded measurement even though the app no longer uses it: it
    # documents the alternative design (stay silent unless highly reliable) and
    # the numbers that would justify switching to it.
    threshold = find_flag_band(scores_a, truth_a)

    if threshold is None:
        print(f"\n  (No score range reaches {FLAG_PRECISION_FLOOR:.0%} reliability, "
              f"so a 'stay silent unless confident' mode is not available.)")
        band = None
    else:
        band = describe_band(scores_b, truth_b, threshold)
        print(f"\n  (For reference: a 'stay silent unless confident' mode would "
              f"flag\n   only scores above {fmt_cutoff(band['score_above'])} and be right "
              f"{band['precision']}% of the time, but would\n   answer about just "
              f"{band['coverage']}% of articles. Not what the app does.)")

    # ---- where the verdict is not worth trusting ----
    # Chosen on half A, reported on half B, the same discipline as the cutoff.
    unsure_choice = find_unsure_band(scores_a, truth_a, best['cutoff'])
    unsure = (describe_unsure_band(scores_b, truth_b, best['cutoff'], unsure_choice)
              if unsure_choice else None)

    print()
    print("=" * 68)
    print("  WHERE THE MODEL IS NOT WORTH TRUSTING")
    print("=" * 68)
    if unsure is None:
        print(f"  No band qualifies: nowhere is this model's verdict worse than")
        print(f"  {UNSURE_ACCURACY_FLOOR:.0f}% reliable. An outside web search has nothing "
              f"obvious to be")
        print("  called in for.")
    else:
        print(f"  scores between {unsure['low']} and {unsure['high']}")
        print()
        print(f"  {'':22} {'articles':>9} {'verdict right':>14}")
        print("  " + "-" * 48)
        print(f"  {'inside the band':22} {unsure['coverage']:>8.1f}% "
              f"{unsure['accuracy_inside']:>13.1f}%")
        print(f"  {'outside it':22} {100 - unsure['coverage']:>8.1f}% "
              f"{unsure['accuracy_outside']:>13.1f}%")
        print()
        print("  Inside that band the verdict is close to a coin flip, so it is")
        print("  where an outside web search earns its keep. Outside it the model")
        print("  already knows the answer and a search would be wasted money.")
        print()
        print(f"  Cost: about {unsure['searches_per_1000_articles']} searches per "
              f"1,000 articles analysed.")
        print(f"  Gemini's free allowance is {FREE_SEARCHES_PER_DAY} grounded "
              f"searches a day, so")
        daily = int(FREE_SEARCHES_PER_DAY / max(unsure['coverage'], 0.1) * 100)
        print(f"  roughly {daily:,} articles a day before it runs out.")

    # ---- record the choice next to the model ----
    with open(CUTOFF_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'cutoff': float(best['cutoff']),
            'chosen_for': 'best accuracy on a held-out half of McIntire',
            'accuracy': round(tuned['accuracy'], 2),
            'real_recall': round(tuned['real_recall'], 3),
            'fake_recall': round(tuned['fake_recall'], 3),
            'cutoff_chosen_on': int(len(index_a)),
            'measured_on': int(len(index_b)),
            'evaluation': 'cross_source',
            'alternative_even_cutoff': float(fairest['cutoff']),
            # What analysis_engine.py reads to build its verdicts.
            'verdict': verdict,
            # Recorded but unused — see the comment above find_flag_band's call.
            'flag': band,
            'flag_precision_floor': FLAG_PRECISION_FLOOR,
            # The score range where the verdict is barely better than a coin
            # flip. analyzer/web_check.py reads this to decide when an outside
            # web search is worth an API call. Re-measured on every run, so it
            # follows the model instead of going stale.
            'unsure_band': unsure,
            'unsure_max_coverage': UNSURE_MAX_COVERAGE,
            'unsure_accuracy_floor': UNSURE_ACCURACY_FLOOR,
        }, f, indent=2)

    print(f"\n  Written to {CUTOFF_PATH}")
    print("  analysis_engine.py reads this on its next restart — the app will")
    print("  quote these figures without anything being typed into it.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retune', action='store_true',
                        help='reuse saved scores instead of running the model')
    run(parser.parse_args().retune)
