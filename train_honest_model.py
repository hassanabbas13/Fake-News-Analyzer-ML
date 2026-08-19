"""
train_honest_model.py — Train the model, and measure it in a way that means something
=====================================================================================

Why this exists
---------------
train_mega_model.py shuffles all three CSV files together, hides one article in
five, and tests on those. It reports about 96%. That number is close to
meaningless, for two reasons:

  1. The hidden articles came from the same piles as the studied ones. Four
     fifths of the Reuters articles taught the model "Reuters = real", and then
     it was tested on the remaining fifth of the Reuters articles.

  2. Every article it was tested on is also sitting in the KnownArticle table.
     Step 1 of the app catches those before the model is ever consulted. So the
     figure describes a situation the model never actually faces.

This script measures the thing we actually care about: hand it a whole
collection of articles it has never seen, from different outlets, and see how
it does. That is the number worth quoting.

It also runs the experiment we had never actually run — does stripping the
publisher fingerprints out of the text help? — and prints both answers side by
side so you can see for yourself rather than take anyone's word for it.

What it writes
--------------
Same three files as train_mega_model.py, so the app picks the new model up with
no other changes:

  analyzer/model.pkl        the classifier
  analyzer/vectorizer.pkl   the word-weighting it expects
  analyzer/model_meta.json  the measurements, including which test produced them

The saved model learns from everything available, cleaned. The accuracy recorded
alongside it is the cross-collection figure, because that is the honest one.

Usage
-----
  python train_honest_model.py             run the experiment and save the model
  python train_honest_model.py --measure   run the experiment, save nothing
"""

import argparse
import json
import os
import pickle
from datetime import date

import django
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Setup Django environment (needed for the shared text helpers in analyzer/)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fake_news_project.settings')
django.setup()

from analyzer.text_cleaning import clean_article_text, count_fingerprints
from analyzer.text_matching import normalize_headline

# Same settings train_mega_model.py used, so the comparison is fair — the only
# things changing between runs are the text and the split.
MAX_FEATURES = 50000
RANDOM_STATE = 42

# The two collections. Keeping them apart is the whole point: one is used to
# teach and the other to examine, so neither can leak into the other.
ISOT_FILES = [('Fake.csv', 'FAKE'), ('True.csv', 'REAL')]
MCINTIRE_FILE = 'fake_or_real_news.csv'


# ============================================================================
# READING THE DATA
# ============================================================================

def read_csv_or_exit(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"ERROR: {path} not found in this folder.\n"
            f"       See the Setup section of README.md for where to get it."
        )
    return pd.read_csv(path)


def load_collections():
    """
    Read the CSVs into one table with a 'collection' column recording which
    pile each row came from, plus 'label_binary' (1 = FAKE, 0 = REAL).

    Rows with no article text are dropped — there is nothing to read.
    """
    frames = []

    for path, label in ISOT_FILES:
        print(f"  reading {path} ...")
        df = read_csv_or_exit(path)[['title', 'text']].copy()
        df['label'] = label
        df['collection'] = 'ISOT'
        frames.append(df)

    print(f"  reading {MCINTIRE_FILE} ...")
    df = read_csv_or_exit(MCINTIRE_FILE)[['title', 'text', 'label']].copy()
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df['collection'] = 'McIntire'
    frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=['text'])
    combined['title'] = combined['title'].fillna('').astype(str)
    combined['text'] = combined['text'].astype(str)

    # Headline and body together, the same way analysis_engine.py builds it
    combined['raw'] = (combined['title'] + ' ' + combined['text']).str.strip()
    combined = combined[combined['raw'].str.len() > 0]

    combined['label_binary'] = (combined['label'] == 'FAKE').astype(int)

    return combined.reset_index(drop=True)


def drop_cross_collection_duplicates(df):
    """
    Remove McIntire rows whose headline already appears in ISOT.

    Without this the "unseen collection" test is not really unseen — any
    article carried by both piles would be taught and then examined, which is
    the exact mistake this script exists to avoid.
    """
    keys = df['title'].map(normalize_headline)
    isot_keys = set(keys[df['collection'] == 'ISOT']) - {''}

    overlapping = (df['collection'] == 'McIntire') & keys.isin(isot_keys)
    n = int(overlapping.sum())

    if n:
        print(f"  dropped {n:,} McIntire rows that also appear in ISOT")
    else:
        print("  no headline overlap between the two collections")

    return df[~overlapping].reset_index(drop=True)


# ============================================================================
# THE FINGERPRINT SURVEY
# ============================================================================

def survey_fingerprints(df):
    """
    Count how often each cleaning rule fires in each collection, and print it.

    This is the evidence for the whole exercise: if 'agency dateline' fires on
    essentially every ISOT real article and never on a fake one, then that one
    marker alone can separate the piles, and no reading is required.
    """
    print("\nHow much publisher fingerprint is in the text?")
    print("(number of articles containing each marker)")

    groups = [
        ('ISOT real', (df['collection'] == 'ISOT') & (df['label_binary'] == 0)),
        ('ISOT fake', (df['collection'] == 'ISOT') & (df['label_binary'] == 1)),
        ('McIntire real', (df['collection'] == 'McIntire') & (df['label_binary'] == 0)),
        ('McIntire fake', (df['collection'] == 'McIntire') & (df['label_binary'] == 1)),
    ]

    tallies = {}
    sizes = {}
    for name, mask in groups:
        subset = df.loc[mask, 'raw']
        sizes[name] = len(subset)
        counts = {}
        for text in subset:
            for rule in count_fingerprints(text):
                counts[rule] = counts.get(rule, 0) + 1
        tallies[name] = counts

    names = [n for n, _ in groups]
    width = max(len(r) for r in _all_rules(tallies)) if tallies else 20

    header = f"  {'marker'.ljust(width)}  " + "  ".join(n.rjust(13) for n in names)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rule in _all_rules(tallies):
        cells = []
        for name in names:
            hits = tallies[name].get(rule, 0)
            total = sizes[name] or 1
            cells.append(f"{hits:>6,} ({hits * 100 // total:>3}%)".rjust(13))
        print(f"  {rule.ljust(width)}  " + "  ".join(cells))

    print("  " + "-" * (len(header) - 2))
    print(f"  {'articles'.ljust(width)}  "
          + "  ".join(f"{sizes[n]:,}".rjust(13) for n in names))


def _all_rules(tallies):
    """Every rule name that fired anywhere, in a stable order."""
    seen = []
    for counts in tallies.values():
        for rule in counts:
            if rule not in seen:
                seen.append(rule)
    return sorted(seen)


# ============================================================================
# ONE TRAINING RUN
# ============================================================================

def train_once(train_text, train_labels, test_text, test_labels):
    """
    Train on one set of articles, score on another, return everything we might
    want to look at afterwards.
    """
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
    train_vectors = vectorizer.fit_transform(train_text)
    test_vectors = vectorizer.transform(test_text)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(train_vectors, train_labels)

    predictions = model.predict(test_vectors)

    return {
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': accuracy_score(test_labels, predictions) * 100,
        'predictions': predictions,
        'truth': test_labels,
        'n_train': len(train_text),
        'n_test': len(test_text),
    }


# ============================================================================
# THE EXPERIMENT
# ============================================================================

def run_experiment(df):
    """
    Six runs: three ways of splitting the data, each with the text left alone
    and with the fingerprints stripped out.

    Returns a dict keyed by (split name, 'raw' or 'clean').
    """
    isot = df[df['collection'] == 'ISOT']
    mcintire = df[df['collection'] == 'McIntire']

    splits = [
        # (name, train rows, test rows) — None means "shuffle and hide a fifth"
        ('shuffled (the old way)', None, None),
        ('ISOT -> McIntire', isot, mcintire),
        ('McIntire -> ISOT', mcintire, isot),
    ]

    results = {}

    for split_name, train_rows, test_rows in splits:
        for version in ('raw', 'clean'):
            column = 'raw' if version == 'raw' else 'cleaned'

            if train_rows is None:
                train_text, test_text, train_y, test_y = train_test_split(
                    df[column], df['label_binary'],
                    test_size=0.2, random_state=RANDOM_STATE,
                    stratify=df['label_binary'],
                )
            else:
                train_text, train_y = train_rows[column], train_rows['label_binary']
                test_text, test_y = test_rows[column], test_rows['label_binary']

            label = f"{split_name} / {version}"
            print(f"  training: {label} ...")
            results[(split_name, version)] = train_once(
                train_text, train_y, test_text, test_y
            )

    return results


def print_results(results):
    """The whole point of the script, in one table."""
    splits = []
    for split_name, _version in results:
        if split_name not in splits:
            splits.append(split_name)

    width = max(len(s) for s in splits)

    print(f"\n{'=' * 72}")
    print("  RESULTS — percentage of test articles labelled correctly")
    print(f"{'=' * 72}")
    print(f"  {'split'.ljust(width)}   {'text as-is':>12}  {'cleaned':>12}   {'change':>8}")
    print("  " + "-" * (width + 40))

    for split_name in splits:
        raw = results[(split_name, 'raw')]['accuracy']
        clean = results[(split_name, 'clean')]['accuracy']
        change = clean - raw
        print(f"  {split_name.ljust(width)}   {raw:>11.2f}%  {clean:>11.2f}%   "
              f"{change:>+7.2f}")

    print("  " + "-" * (width + 40))
    print()
    print("  'shuffled' is the old measurement — inflated, because the hidden")
    print("  articles came from the same piles as the studied ones.")
    print("  The arrow rows are the honest ones: taught on one collection,")
    print("  examined on a different collection it had never seen.")

    # The headline result deserves a closer look than one number. An accuracy
    # of 70% could mean "decent at both" or "calls everything fake" — the
    # per-class breakdown is what tells them apart.
    honest = results[('ISOT -> McIntire', 'clean')]
    print(f"\n{'-' * 72}")
    print("  The honest result in detail (taught on ISOT, examined on McIntire,")
    print("  fingerprints removed):")
    print(f"{'-' * 72}")
    print(classification_report(
        honest['truth'], honest['predictions'],
        target_names=['REAL', 'FAKE'], digits=3,
    ))


# ============================================================================
# SAVING THE MODEL
# ============================================================================

def save_model(df, results):
    """
    Train the model we actually deploy — on everything available, cleaned —
    and record the honest measurement next to it.

    Learning from all the data and reporting a figure measured on a held-back
    collection is the normal way round: you want the deployed model to have
    seen as much as possible, and you want the number you quote to come from
    articles nothing had seen.
    """
    print("\nTraining the model to deploy (all articles, cleaned) ...")

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
    vectors = vectorizer.fit_transform(df['cleaned'])

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(vectors, df['label_binary'])

    honest = results[('ISOT -> McIntire', 'clean')]

    model_path = os.path.join('analyzer', 'model.pkl')
    vectorizer_path = os.path.join('analyzer', 'vectorizer.pkl')
    meta_path = os.path.join('analyzer', 'model_meta.json')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  saved {model_path}")

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  saved {vectorizer_path}")

    # analysis_engine.py reads this rather than having numbers typed into it.
    # 'evaluation' tells it which sentence to use, so a cross-collection figure
    # is never described as if it came from a shuffled split.
    metadata = {
        'test_accuracy': round(honest['accuracy'], 2),
        'training_samples': honest['n_train'],
        'test_samples': honest['n_test'],
        'total_samples': len(df),
        'vocabulary_size': len(vectorizer.vocabulary_),
        'trained_at': date.today().isoformat(),

        'evaluation': 'cross_source',
        'trained_on': 'Kaggle ISOT (Fake.csv + True.csv)',
        'tested_on': 'McIntire (fake_or_real_news.csv)',
        'text_cleaned': True,

        # Kept for comparison, and clearly named so nobody mistakes either of
        # these for the headline figure.
        'inflated_shuffled_accuracy': round(
            results[('shuffled (the old way)', 'clean')]['accuracy'], 2),
        'uncleaned_cross_source_accuracy': round(
            results[('ISOT -> McIntire', 'raw')]['accuracy'], 2),
        'reverse_cross_source_accuracy': round(
            results[('McIntire -> ISOT', 'clean')]['accuracy'], 2),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"  saved {meta_path}")

    return metadata


# ============================================================================
# COMMAND LINE
# ============================================================================

def run():
    parser = argparse.ArgumentParser(
        description="Train the model and measure it on a collection it never saw.",
    )
    parser.add_argument(
        '--measure', action='store_true',
        help="Run the experiment and print the numbers, but do not overwrite "
             "model.pkl, vectorizer.pkl or model_meta.json.",
    )
    args = parser.parse_args()

    print("Loading the collections ...")
    df = load_collections()
    df = drop_cross_collection_duplicates(df)

    print(f"\n  {len(df):,} articles usable")
    for collection in ('ISOT', 'McIntire'):
        rows = df[df['collection'] == collection]
        fake = int((rows['label_binary'] == 1).sum())
        print(f"    {collection:9} {len(rows):>7,}  "
              f"({fake:,} fake / {len(rows) - fake:,} real)")

    survey_fingerprints(df)

    print("\nStripping the fingerprints out ...")
    df['cleaned'] = df['raw'].map(clean_article_text)

    # A rule that is too greedy would empty articles out entirely. Worth
    # knowing about rather than silently training on blanks.
    emptied = int((df['cleaned'].str.len() == 0).sum())
    if emptied:
        print(f"  WARNING: {emptied:,} articles are empty after cleaning "
              f"— a rule is too greedy")
    shrink = 100 - (df['cleaned'].str.len().sum() * 100
                    // max(df['raw'].str.len().sum(), 1))
    print(f"  removed about {shrink}% of the total text")

    print("\nRunning the experiment (six training runs, this takes a minute) ...")
    results = run_experiment(df)

    print_results(results)

    if args.measure:
        print("\n--measure given: nothing was saved.")
        return

    metadata = save_model(df, results)

    print(f"\n{'=' * 72}")
    print(f"  The app will now report {metadata['test_accuracy']}%, measured on "
          f"{metadata['test_samples']:,} articles")
    print(f"  from a collection it never learned from.")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    run()
