"""
load_mega_data.py — Fill the fact-check list from CSV files
===========================================================

This TOPS UP the list. It adds articles it does not already have and leaves
everything else alone, so:

  * running it twice does not create duplicates
  * adding one new source only reads that one file
  * anything you fixed by hand, or added from somewhere else, survives

How it knows what it already has: every saved article carries a tidied-up
version of its headline (see analyzer/text_matching.py). Two spellings of the
same headline produce the same tidied version, so "do I already have this?" has
a reliable answer. Before that existed, the only safe way to avoid duplicates
was to delete everything and start over — which is what this script used to do.

Usage
-----
  python load_mega_data.py                     add anything new from all three known files
  python load_mega_data.py --only true         just True.csv
  python load_mega_data.py --list              show the known files and what is loaded
  python load_mega_data.py --reset             delete everything first (the old behaviour)

Adding a brand new file without editing this script:

  python load_mega_data.py --file bbc.csv --source "BBC" --label REAL
  python load_mega_data.py --file checked.csv --source "PolitiFact" --basis fact_check

If your file uses different column names:

  python load_mega_data.py --file x.csv --source "X" --title-col headline --text-col body
"""

import argparse
import os
import sys

import django
import pandas as pd

# Setup Django environment so we can use the models
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fake_news_project.settings')
django.setup()

from analyzer.models import KnownArticle
from analyzer.text_matching import normalize_headline

# How much of each article body to keep. Enough for the result page to show the
# user what was matched, without storing 140MB of article text in SQLite.
EXTRACT_CHARS = 600

# Insert in batches so SQLite doesn't choke on one enormous statement
BATCH_SIZE = 5000

# The files this project ships with.
#   label=None  -> read the label from the file's own label column
#   basis       -> why the label deserves trust; see KnownArticle.BASIS_CHOICES
KNOWN_DATASETS = {
    'fake': {
        'path': 'Fake.csv',
        'source': 'Kaggle ISOT',
        'label': 'FAKE',
        'basis': KnownArticle.BASIS_SOURCE,
    },
    'true': {
        'path': 'True.csv',
        'source': 'Kaggle ISOT',
        'label': 'REAL',
        'basis': KnownArticle.BASIS_SOURCE,
    },
    'mixed': {
        'path': 'fake_or_real_news.csv',
        'source': 'McIntire',
        'label': None,          # this file has its own label column
        'basis': KnownArticle.BASIS_SOURCE,
    },
}


# ============================================================================
# WHAT WE ALREADY HAVE
# ============================================================================

def load_existing():
    """
    Map every tidied headline we already hold to the best evidence we hold for
    it, as a rank (lower is better — see KnownArticle.BASIS_RANK).

    Keeping the rank rather than just the headline lets us do something more
    useful than blindly skipping: if a new file brings a genuine fact-check for
    a headline we only had a publisher guess for, that is worth adding, because
    the lookup prefers fact-checked entries.
    """
    best = {}

    rows = KnownArticle.objects.values_list('headline_key', 'label_basis').iterator()
    for key, basis in rows:
        rank = KnownArticle.BASIS_RANK.get(basis, 99)
        if key not in best or rank < best[key]:
            best[key] = rank

    return best


# ============================================================================
# READING ONE FILE
# ============================================================================

def read_dataset(path, source, label, basis, title_col, text_col, label_col):
    """
    Read one CSV into a tidy DataFrame with the columns we need.
    Exits with a clear message rather than a traceback if something is missing.
    """
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found in this folder.")
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"  ERROR reading {path}: {exc}")
        sys.exit(1)

    if title_col not in df.columns:
        print(f"  ERROR: {path} has no '{title_col}' column. Found: {list(df.columns)}")
        print(f"  Use --title-col to point at the right one.")
        sys.exit(1)

    # Every article needs a headline; the body is optional.
    # fillna before astype, or missing bodies become the literal string "nan".
    df = df.dropna(subset=[title_col])
    out = pd.DataFrame()
    out['headline'] = df[title_col].astype(str)
    if text_col in df.columns:
        out['body'] = df[text_col].fillna('').astype(str)
    else:
        out['body'] = ''

    # The label either comes from the command line (whole file is FAKE or REAL)
    # or from a column in the file itself
    if label:
        out['label'] = label.upper()
    else:
        if label_col not in df.columns:
            print(f"  ERROR: {path} has no '{label_col}' column and no --label was given.")
            sys.exit(1)
        keep = df[label_col].notna()
        df, out = df[keep], out[keep]
        out['label'] = df[label_col].astype(str).str.strip().str.upper()

    out['source'] = source
    out['basis'] = basis

    # Build the lookup key with the SAME function analysis_engine.py searches
    # with — see analyzer/text_matching.py. Never tidy headlines by hand here,
    # or the two sides drift apart and lookups silently start missing.
    out['key'] = out['headline'].map(normalize_headline)

    return out


def build_extract(body):
    """Keep a short opening extract so a match can show what it found."""
    text = ('' if body is None else str(body)).strip()
    if len(text) <= EXTRACT_CHARS:
        return text
    return text[:EXTRACT_CHARS] + '...'


# ============================================================================
# ADDING WHAT IS NEW
# ============================================================================

def add_new_articles(df, existing, name, verbose=True):
    """
    Insert only the articles we do not already hold, and report exactly what was
    skipped. `existing` is updated as we go, so duplicates inside this same file
    get caught too.

    Pass verbose=False to stay silent (the test suite does).
    """
    to_create = []
    counts = {'added': 0, 'better_evidence': 0, 'already_had': 0, 'unusable': 0}

    for row in df.itertuples(index=False):
        key = row.key

        # A blank key means the headline was punctuation or whitespace only
        if not key:
            counts['unusable'] += 1
            continue

        new_rank = KnownArticle.BASIS_RANK.get(row.basis, 99)
        held_rank = existing.get(key)

        if held_rank is not None:
            if new_rank >= held_rank:
                # Already have this, and the new one is no better. Skip.
                counts['already_had'] += 1
                continue
            # We only had a weaker basis for this headline — a real fact-check
            # is worth adding, because best_match() prefers it.
            counts['better_evidence'] += 1
        else:
            counts['added'] += 1

        to_create.append(
            KnownArticle(
                headline=row.headline.strip()[:1000],
                headline_key=key[:1000],
                article_text=build_extract(row.body),
                label=row.label,
                source=row.source,
                label_basis=row.basis,
            )
        )

        # Remember it immediately so a repeat later in this same file is skipped
        existing[key] = new_rank

    if to_create:
        # bulk_create skips save(), which is why headline_key is set by hand above
        KnownArticle.objects.bulk_create(to_create, batch_size=BATCH_SIZE)

    total_new = counts['added'] + counts['better_evidence']
    if verbose:
        print(f"  {name}: {len(df)} rows read, {total_new} inserted")
        print(f"      new headlines:        {counts['added']}")
        print(f"      better evidence:      {counts['better_evidence']}")
        print(f"      already had:          {counts['already_had']}")
        print(f"      unusable headline:    {counts['unusable']}")

    return counts


# ============================================================================
# COMMAND LINE
# ============================================================================

def show_status():
    total = KnownArticle.objects.count()
    print(f"Currently loaded: {total:,} articles")
    if total:
        for basis, human in KnownArticle.BASIS_CHOICES:
            n = KnownArticle.objects.filter(label_basis=basis).count()
            print(f"  {human}: {n:,}")
        print("  By source:")
        for src in KnownArticle.objects.values_list('source', flat=True).distinct():
            n = KnownArticle.objects.filter(source=src).count()
            print(f"    {src}: {n:,}")
    print()
    print("Known files:")
    for name, spec in KNOWN_DATASETS.items():
        there = "found" if os.path.exists(spec['path']) else "MISSING"
        print(f"  {name:6} {spec['path']:24} ({there})")


def parse_args():
    p = argparse.ArgumentParser(
        description="Top up the fact-check list from CSV files. Adds only what is new.",
    )
    p.add_argument('--only', nargs='+', choices=sorted(KNOWN_DATASETS),
                   help="Load only these known files instead of all of them.")
    p.add_argument('--file',
                   help="Load a CSV that is not one of the known files.")
    p.add_argument('--source',
                   help="Where --file came from, e.g. \"BBC\". Required with --file.")
    p.add_argument('--label', choices=['FAKE', 'REAL', 'fake', 'real'],
                   help="Use when every row in --file has the same label. "
                        "Leave out to read it from the file's label column.")
    p.add_argument('--basis', choices=[KnownArticle.BASIS_SOURCE, KnownArticle.BASIS_FACT_CHECK],
                   default=KnownArticle.BASIS_SOURCE,
                   help="Why the label deserves trust. Use fact_check only when a "
                        "fact-checker actually investigated the claims.")
    p.add_argument('--title-col', default='title', help="Headline column name in --file.")
    p.add_argument('--text-col', default='text', help="Body column name in --file.")
    p.add_argument('--label-col', default='label', help="Label column name in --file.")
    p.add_argument('--reset', action='store_true',
                   help="DELETE every saved article before loading. Destroys anything "
                        "you fixed by hand. Off by default.")
    p.add_argument('--list', action='store_true',
                   help="Show what is loaded and which files are present, then exit.")
    return p.parse_args()


def run():
    args = parse_args()

    if args.list:
        show_status()
        return

    if args.file and not args.source:
        print("ERROR: --file also needs --source, so we can record where it came from.")
        sys.exit(1)

    # Work out which files to read
    if args.file:
        jobs = [('custom', {
            'path': args.file,
            'source': args.source,
            'label': args.label,
            'basis': args.basis,
        })]
    else:
        names = args.only or list(KNOWN_DATASETS)
        jobs = [(n, KNOWN_DATASETS[n]) for n in names]

    print("Fact-check list loader")
    print(f"  starting with {KnownArticle.objects.count():,} articles already saved")

    if args.reset:
        print("  --reset given: DELETING every saved article first")
        deleted, _ = KnownArticle.objects.all().delete()
        print(f"  deleted {deleted:,} rows")

    # Read what we already hold, once
    existing = load_existing()
    print(f"  recognising {len(existing):,} distinct headlines already held")
    print()

    totals = {'added': 0, 'better_evidence': 0, 'already_had': 0, 'unusable': 0}

    for name, spec in jobs:
        print(f"Reading {spec['path']} ...")
        df = read_dataset(
            path=spec['path'],
            source=spec['source'],
            label=spec.get('label'),
            basis=spec.get('basis', KnownArticle.BASIS_SOURCE),
            title_col=args.title_col,
            text_col=args.text_col,
            label_col=args.label_col,
        )
        counts = add_new_articles(df, existing, name)
        for k in totals:
            totals[k] += counts[k]
        print()

    inserted = totals['added'] + totals['better_evidence']
    print("Done.")
    print(f"  inserted this run:    {inserted:,}")
    print(f"  skipped (already had): {totals['already_had']:,}")
    if totals['unusable']:
        print(f"  skipped (no usable headline): {totals['unusable']:,}")
    print(f"  total now saved:      {KnownArticle.objects.count():,}")


if __name__ == '__main__':
    run()
