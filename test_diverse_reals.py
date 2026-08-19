"""
test_diverse_reals.py — Do the new real articles actually help?
==============================================================

The question
-----------
diverse_reals.csv holds real articles from about 15 different newsrooms. Adding
them should stop the model treating "does not sound like Reuters" as evidence of
being fake. But there are two ways it could look like an improvement without
being one, so this script runs four training sessions instead of two.

  A  before          Fake.csv + True.csv + diverse_fakes.csv  (today's pile)
  B  new newsrooms   the same, plus the real articles from ~15 newsrooms
  C  balance only    the same, plus real articles COPIED from True.csv
  D  Reuters trimmed the same, but with the SAME NUMBER of Reuters articles
                     removed as B adds

Why four, when the fakes only needed three:

  C is the same control as last time. Today's pile leans fake (30,738 fake to
  21,417 real). Adding real articles corrects that lean, and correcting a lean
  moves scores on its own. C has the identical correction and not one new
  newsroom, so anything C gains is bookkeeping.

  D is new, and it matters here. Adding 9,000 non-Reuters real articles makes
  Reuters a smaller SHARE of the real side. That dilution alone could help,
  regardless of whether the new newsrooms taught anything. D dilutes Reuters by
  the same amount by deletion instead of addition — no new writing at all.

So:
  B beats C and D  ->  the new newsrooms genuinely taught the model something.
  D alone matches B ->  we did not need new data, just less Reuters. Cheaper.
  C alone matches B ->  it was only the class balance. Throw the data out.

All four are examined on the same articles: McIntire, which none of them ever
trained on. Nothing is saved and no model file is touched.

How it is measured
------------------
By SORTING SCORE, not accuracy: hand the model one real and one fake article,
how often does it rate the fake as the faker? 50% is a coin toss.

Accuracy is not used as the headline because it depends on where the line
between fake and real is drawn, and the app does not use the default line — it
reads a tuned cutoff from analyzer/reading_model/cutoff.json. Judging at the
default line judges a model that does not exist. This is the same mistake that
made a 76% model look like 62% on 19 Aug 2026; see WHERE_I_LEFT_OFF.txt.

Accuracy at a properly chosen line is printed alongside, with the line picked
on one half of the exam and scored on the other.

A warning about this model
--------------------------
The model here is TF-IDF plus logistic regression, not DistilBERT. It is fast
and free, which is the point — it answers "does this data carry signal?" before
anyone spends 40 minutes on a GPU. It is not a prediction of DistilBERT's score.

Usage
-----
  python test_diverse_reals.py
  python test_diverse_reals.py --extra other_reals.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from train_honest_model import (ISOT_FILES, MCINTIRE_FILE, RANDOM_STATE,
                                read_csv_or_exit, train_once)
from analyzer.text_cleaning import clean_article_text
from analyzer.text_matching import normalize_headline

DIVERSE_FAKES = 'diverse_fakes.csv'


def load_isot():
    frames = []
    for path, label in ISOT_FILES:
        part = read_csv_or_exit(path)[['title', 'text']].copy()
        part['label'] = label
        part['source'] = 'ISOT-real' if label == 'REAL' else 'ISOT-fake'
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def load_labelled(path, source):
    df = read_csv_or_exit(path)[['title', 'text', 'label']].copy()
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df['source'] = source
    return df


def prepare(df):
    """Headline and body joined then cleaned, exactly as the trainer does it."""
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
    """One training session, measured without depending on where the line sits."""
    print(f"  training {name:18} on {len(train_df):,} articles "
          f"({int(train_df['y'].sum()):,} fake / {int((1 - train_df['y']).sum()):,} real)")
    out = train_once(train_df['cleaned'], train_df['y'],
                     test_df['cleaned'], test_df['y'])

    vectors = out['vectorizer'].transform(test_df['cleaned'])
    fake_score = out['model'].predict_proba(vectors)[:, 1]
    truth = test_df['y'].to_numpy()

    sorting = roc_auc_score(truth, fake_score) * 100

    half_a, half_b = train_test_split(
        np.arange(len(truth)), test_size=0.5,
        random_state=RANDOM_STATE, stratify=truth)

    # Candidates drawn from the scores themselves, never a fixed 0.05-0.95
    # ladder. A confident model squashes everything against 1.0 and a fixed
    # ladder then cannot reach the decision at all.
    grid = np.unique(np.quantile(fake_score[half_a], np.linspace(0.001, 0.999, 2000)))
    best = grid[int(np.argmax([
        accuracy_score(truth[half_a], (fake_score[half_a] >= c).astype(int))
        for c in grid]))]

    pred = (fake_score[half_b] >= best).astype(int)
    return {
        'name': name,
        'sorting': sorting,
        'accuracy': accuracy_score(truth[half_b], pred) * 100,
        'real_recall': recall_score(truth[half_b], pred, pos_label=0) * 100,
        'fake_recall': recall_score(truth[half_b], pred, pos_label=1) * 100,
    }


def run():
    parser = argparse.ArgumentParser(
        description="Check whether extra real articles help, or only rebalance.")
    parser.add_argument('--extra', default='diverse_reals.csv',
                        help="CSV of extra real articles. Default diverse_reals.csv")
    args = parser.parse_args()

    for path in (args.extra, DIVERSE_FAKES):
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: {path} not found.")

    print("Loading ...")
    isot = prepare(load_isot())
    fakes = prepare(load_labelled(DIVERSE_FAKES, 'diverse-fake'))
    reals = prepare(load_labelled(args.extra, 'diverse-real'))
    test = prepare(load_labelled(MCINTIRE_FILE, 'McIntire'))

    # Today's pile is the baseline: ISOT plus the diverse fakes already shipped.
    base = pd.concat([isot, fakes], ignore_index=True)

    # The exam must hold nothing any run has studied.
    taught = (set(base['key']) | set(reals['key'])) - {''}
    before = len(test)
    test = test[~test['key'].isin(taught)].reset_index(drop=True)
    if before - len(test):
        print(f"  removed {before - len(test):,} exam articles found in a training pile")

    n_new = len(reals)
    outlets = reals['outlet'].nunique() if 'outlet' in reals.columns else '?'
    print(f"  baseline pile      : {len(base):,}")
    print(f"  new real articles  : {n_new:,} from {outlets} newsrooms")
    print(f"  exam (McIntire)    : {len(test):,} "
          f"({int(test['y'].sum()):,} fake / {int((1 - test['y']).sum()):,} real)")
    print()

    # C: same number of extra REAL rows, copied from the Reuters pile we already
    # have. Same class-balance correction, zero new writing.
    isot_real = base[(base['y'] == 0) & (base['source'] == 'ISOT-real')]
    copies = isot_real.sample(n=min(n_new, len(isot_real)),
                              random_state=RANDOM_STATE, replace=False)

    # D: dilute Reuters by deletion instead of addition. Same shift in how much
    # of the real side is Reuters, again with no new writing.
    drop_ids = set(isot_real.sample(n=min(n_new, len(isot_real) - 1),
                                    random_state=RANDOM_STATE + 1).index)
    trimmed = base.drop(index=drop_ids)

    print("Four training sessions (about a minute each) ...")
    rows = [
        score('A before', base, test),
        score('B new newsrooms', pd.concat([base, reals], ignore_index=True), test),
        score('C balance only', pd.concat([base, copies], ignore_index=True), test),
        score('D Reuters trimmed', trimmed, test),
    ]

    print()
    print("=" * 74)
    print("  ALL FOUR EXAMINED ON THE SAME McINTIRE ARTICLES")
    print("=" * 74)
    print(f"  {'run':20} {'sorting':>9} {'at best':>9} {'catches':>9} {'catches':>9}")
    print(f"  {'':20} {'score':>9} {'line':>9} {'reals':>9} {'fakes':>9}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['name']:20} {r['sorting']:>8.2f}% {r['accuracy']:>8.2f}% "
              f"{r['real_recall']:>8.1f}% {r['fake_recall']:>8.1f}%")
    print("  " + "-" * 70)
    print("  SORTING SCORE is the one to read. No cutoff in it, so it cannot be")
    print("  flattered by the model simply leaning a different way.")
    print()

    a, b, c, d = rows
    vs_balance = b['sorting'] - c['sorting']
    vs_dilution = b['sorting'] - d['sorting']
    worth = min(vs_balance, vs_dilution)

    print("  Reading it:")
    print(f"    new newsrooms moved the sorting score   {b['sorting'] - a['sorting']:+.2f}")
    print(f"    fixing the balance alone moved it       {c['sorting'] - a['sorting']:+.2f}")
    print(f"    just deleting Reuters moved it          {d['sorting'] - a['sorting']:+.2f}")
    print()
    print(f"    new newsrooms beat rebalancing by       {vs_balance:+.2f}")
    print(f"    new newsrooms beat deleting Reuters by   {vs_dilution:+.2f}")
    print(f"    so the new newsrooms are worth about    {worth:+.2f}")
    print()

    if worth < 0.5:
        print("  VERDICT: the new newsrooms taught it nothing measurable.")
        if d['sorting'] > a['sorting'] + 0.5:
            print("  BUT deleting Reuters helped on its own — that is free. Worth")
            print("  trying a Reuters-trimmed retrain without any new data.")
        else:
            print("  Do NOT spend a GPU run on this.")
    elif worth < 2:
        print("  VERDICT: a small real gain. Marginal for a 40-minute GPU run;")
        print("  consider collecting more newsrooms first.")
    else:
        print("  VERDICT: a real gain from the new newsrooms. Worth retraining")
        print("  DistilBERT on ISOT + diverse_fakes + diverse_reals.")
    print()
    print("  Reminder: this is the quick TF-IDF model, not DistilBERT. Evidence")
    print("  about the data, not a prediction of the final score.")
    print()
    print("Nothing was saved. model.pkl, vectorizer.pkl and the DistilBERT folder")
    print("are all untouched.")


if __name__ == '__main__':
    run()
