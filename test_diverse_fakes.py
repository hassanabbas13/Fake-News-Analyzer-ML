"""
test_diverse_fakes.py — Do the new fake articles actually help?
==============================================================

The question
-----------
diverse_fakes.csv holds 7,257 fake articles from 150 different websites. Adding
them to training should make the model better at catching fakes. But there is a
trap: adding 7,257 fake articles and no real ones tilts the training pile
towards fake, and a tilted pile makes ANY model shout "fake" more often. The
fake score would go up and it would look like progress.

So this runs three training sessions and compares them:

  A  before          Fake.csv + True.csv, exactly what you train on today
  B  new websites    the same, plus the 7,257 articles from 150 new websites
  C  tilt only       the same, plus 7,257 articles COPIED from Fake.csv

C is the control. It has the identical tilt as B and not one new website. So:

  B better than A, and C also better than A by the same amount
      -> the new websites taught nothing. It was only the tilt. Throw them out.

  B better than C
      -> the gap between them is what the new websites are actually worth.

All three are examined on exactly the same articles: McIntire, which none of
them ever learned from. Nothing is saved and no model files are touched — this
only prints numbers.

How it is measured, and why not by plain accuracy
------------------------------------------------
The headline figure here is a SORTING SCORE, not an accuracy: hand the model one
real article and one fake one, and how often does it rate the fake as the faker
of the two? 50% is a coin toss.

Plain accuracy was the wrong tool for this comparison. Accuracy depends on where
you draw the line between "fake" and "real", and this app does not use the
default line — it reads a tuned cutoff out of analyzer/reading_model/cutoff.json.
Two models can sort articles equally well and still post very different
accuracies purely because of where their line happens to fall. The sorting score
has no line in it at all, so it measures the thing we actually want to know:
did the model learn more?

Accuracy is still printed alongside, with the line chosen on one half of the
exam and the score taken from the other half — the same split discipline
tune_cutoff.py uses, so a lucky cutoff cannot flatter the number.

A warning about this model
--------------------------
The quick model here is TF-IDF plus logistic regression, not DistilBERT, and it
leans the opposite way: it over-calls FAKE, while DistilBERT over-calls REAL. So
read this as evidence about whether the new DATA carries signal. It is not a
prediction of what DistilBERT would score.

Usage
-----
  python test_diverse_fakes.py
  python test_diverse_fakes.py --extra other_fakes.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# train_honest_model sets up Django and holds the pieces we want to reuse. Reuse
# rather than reimplement: if its cleaning or its duplicate rule changes, this
# comparison changes with it instead of quietly drifting out of step.
from train_honest_model import (
    ISOT_FILES,
    MCINTIRE_FILE,
    RANDOM_STATE,
    read_csv_or_exit,
    train_once,
)
from analyzer.text_cleaning import clean_article_text
from analyzer.text_matching import normalize_headline


def load_isot():
    """Fake.csv + True.csv, the pile the model learns from today."""
    frames = []
    for path, label in ISOT_FILES:
        df = read_csv_or_exit(path)[['title', 'text']].copy()
        df['label'] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_mcintire():
    """The exam. Never trained on, by anyone, in any of the three runs."""
    df = read_csv_or_exit(MCINTIRE_FILE)[['title', 'text', 'label']].copy()
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    return df


def load_extra(path):
    """The new fake articles. 'domain' is kept only so we can count websites."""
    df = read_csv_or_exit(path)
    keep = ['title', 'text', 'label']
    if 'domain' in df.columns:
        keep.append('domain')
    df = df[keep].copy()
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    return df


def prepare(df):
    """Headline and body joined, then cleaned, the same way the trainer does it."""
    df = df.dropna(subset=['text']).copy()
    df['title'] = df['title'].fillna('').astype(str)
    df['text'] = df['text'].astype(str)
    df['raw'] = (df['title'] + ' ' + df['text']).str.strip()
    df = df[df['raw'].str.len() > 0].copy()
    df['cleaned'] = df['raw'].map(clean_article_text)
    df['key'] = df['title'].map(normalize_headline)
    df['y'] = (df['label'] == 'FAKE').astype(int)
    return df.reset_index(drop=True)


def score(name, train_df, test_df):
    """
    One training session, measured in a way that does not depend on where the
    line between "fake" and "real" is drawn.

    Why that matters: the app does not use the default line. It reads a tuned
    cutoff out of analyzer/reading_model/cutoff.json. Judging a model at the
    default line therefore judges it at a setting nobody uses, and a model that
    happens to sit badly at the default can still be the better model once the
    line is moved. So we report two things:

      sorting score  how well it orders articles from most-fake to most-real,
                     with no line drawn at all. This is the real measure of
                     whether the model learned anything.
      at best line   accuracy with the line placed where it works best. The
                     line is chosen on one half of the exam and the score comes
                     from the other half, the same discipline tune_cutoff.py
                     uses, so the choice cannot flatter the result.
    """
    print(f"  training {name} on {len(train_df):,} articles ...")
    out = train_once(
        train_df['cleaned'], train_df['y'],
        test_df['cleaned'], test_df['y'],
    )

    # Probability of FAKE for each exam article
    vectors = out['vectorizer'].transform(test_df['cleaned'])
    fake_score = out['model'].predict_proba(vectors)[:, 1]
    truth = test_df['y'].to_numpy()

    sorting = roc_auc_score(truth, fake_score) * 100

    # Half picks the line, the other half reports. Same seed every run and every
    # condition, so all three are judged on identical articles.
    half_a, half_b = train_test_split(
        np.arange(len(truth)), test_size=0.5,
        random_state=RANDOM_STATE, stratify=truth,
    )

    lines = np.arange(0.05, 0.96, 0.01)
    best_line = max(lines, key=lambda t: accuracy_score(
        truth[half_a], (fake_score[half_a] >= t).astype(int)))

    pred_b = (fake_score[half_b] >= best_line).astype(int)
    return {
        'name': name,
        'n_train': len(train_df),
        'sorting': sorting,
        'best_line': best_line,
        'accuracy': accuracy_score(truth[half_b], pred_b) * 100,
        'fake_recall': recall_score(truth[half_b], pred_b, pos_label=1) * 100,
        'real_recall': recall_score(truth[half_b], pred_b, pos_label=0) * 100,
    }


def run():
    parser = argparse.ArgumentParser(
        description="Check whether extra fake articles help, or only tilt the pile.")
    parser.add_argument('--extra', default='diverse_fakes.csv',
                        help="CSV of extra fake articles. Default diverse_fakes.csv")
    args = parser.parse_args()

    if not os.path.exists(args.extra):
        raise SystemExit(f"ERROR: {args.extra} not found. Run fetch_diverse_fakes.py first.")

    print("Loading ...")
    isot = prepare(load_isot())
    test = prepare(load_mcintire())
    extra = prepare(load_extra(args.extra))

    # The exam must contain nothing any run has studied. train_honest_model.py
    # already drops McIntire rows that appear in ISOT; the new file has to face
    # the same check, or run B would be examined on its own homework.
    isot_keys = set(isot['key']) - {''}
    extra_keys = set(extra['key']) - {''}

    before = len(test)
    test = test[~test['key'].isin(isot_keys | extra_keys)].reset_index(drop=True)
    dropped = before - len(test)
    if dropped:
        print(f"  removed {dropped:,} exam articles that appear in some training pile")

    print(f"  train pile (ISOT):  {len(isot):,}  "
          f"({int(isot['y'].sum()):,} fake / {int((1 - isot['y']).sum()):,} real)")
    print(f"  new fake articles:  {len(extra):,}", end='')
    if 'domain' in extra.columns:
        print(f"  from {extra['domain'].nunique():,} websites")
    else:
        print()
    print(f"  exam (McIntire):    {len(test):,}  "
          f"({int(test['y'].sum()):,} fake / {int((1 - test['y']).sum()):,} real)")
    print()

    # C: the same number of extra fake rows, but copied from the pile the model
    # has already seen. Same tilt, zero new information.
    copies = isot[isot['y'] == 1].sample(
        n=min(len(extra), int(isot['y'].sum())),
        random_state=RANDOM_STATE, replace=False,
    )

    print("Three training sessions (about a minute each) ...")
    rows = [
        score('A before', isot, test),
        score('B new websites', pd.concat([isot, extra], ignore_index=True), test),
        score('C tilt only', pd.concat([isot, copies], ignore_index=True), test),
    ]

    print()
    print("=" * 78)
    print("  ALL THREE EXAMINED ON THE SAME McINTIRE ARTICLES, NONE OF WHICH")
    print("  ANY OF THEM LEARNED FROM")
    print("=" * 78)
    print(f"  {'run':16} {'sorting':>9} {'at best':>9} {'catches':>9} {'catches':>9} "
          f"{'line':>7}")
    print(f"  {'':16} {'score':>9} {'line':>9} {'fakes':>9} {'reals':>9} {'used':>7}")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['name']:16} {r['sorting']:>8.1f}% {r['accuracy']:>8.1f}% "
              f"{r['fake_recall']:>8.1f}% {r['real_recall']:>8.1f}% "
              f"{r['best_line']:>7.2f}")
    print("  " + "-" * 74)
    print("  SORTING SCORE is the one to read. It asks: if you handed the model one")
    print("  real and one fake article, how often does it rate the fake as faker?")
    print("  50% is a coin toss. It ignores where the line is drawn, so it cannot")
    print("  be flattered by a lucky cutoff.")
    print()

    a, b, c = rows
    gain_new = b['sorting'] - a['sorting']
    gain_tilt = c['sorting'] - a['sorting']
    worth = b['sorting'] - c['sorting']

    print("  Reading it:")
    print(f"    adding the new websites moved the sorting score by {gain_new:+.1f} points")
    print(f"    copied articles alone moved it by                  {gain_tilt:+.1f} points")
    print(f"    so the new websites are worth about                {worth:+.1f} points")
    print()
    print("  (Copying articles cannot teach a model anything new, so whatever")
    print("   run C gained is noise and tilt. Only the gap B minus C is real.)")
    print()

    if worth < 0.5:
        print("  VERDICT: the new websites taught it nothing measurable.")
        print("  Do NOT spend hours retraining DistilBERT on this data.")
    elif worth < 2:
        print("  VERDICT: a small but real gain. Probably not worth hours of")
        print("  retraining on its own — better to gather more data first.")
    else:
        print("  VERDICT: a real gain from the new websites. Worth retraining")
        print("  DistilBERT on this data and measuring properly.")
    print()
    print("  Caution: this quick model is not DistilBERT. It leans the opposite way")
    print("  (it over-calls fake; DistilBERT over-calls real), so treat this as")
    print("  evidence about whether the DATA carries signal, not as a prediction")
    print("  of DistilBERT's final score.")
    print()
    print("Nothing was saved. model.pkl, vectorizer.pkl and the DistilBERT folder")
    print("are all untouched.")


if __name__ == '__main__':
    run()
