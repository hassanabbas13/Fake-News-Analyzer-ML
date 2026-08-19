"""
fetch_diverse_reals.py — Collect real articles from many different newsrooms
===========================================================================

Why this exists
---------------
Every REAL article this project trains on comes from Reuters. True.csv is 100%
Reuters wire copy. So the model has not learned what a real article looks like,
it has learned what Reuters looks like — and anything written differently
starts to look suspicious to it.

fetch_diverse_fakes.py already fixed the other side of this problem: the fake
articles now come from 150 different websites. This script does the same job
for the real side.

Where the articles come from
----------------------------
"All the News 2.1" (rjac/all-the-news-2-1-Component-one on Hugging Face):
2,688,878 articles from about 26 named news outlets, with the full article body
and a `publication` column saying who published it.

It is 5.3 GB, which is far too big to download. But Hugging Face serves any
row range of it over an ordinary web API, so we ask for 20 rows at a time from
places spread across the whole file and never fetch the rest. About 40 MB for
what we need.

Four things this script is careful about
----------------------------------------
1. NO REUTERS. Reuters is 25-35% of this collection — the exact thing we are
   trying to escape. Excluded outright. See EXCLUDED_OUTLETS.

2. HARD NEWS ONLY, and this one is subtle. The fake articles we train on are
   conspiracy and junk-science pieces, so they are mostly about politics and
   health. If the real articles were tech blogs and celebrity gossip, the model
   could get a good score by learning "tech topic = real, politics topic =
   fake" — which is not reading, it is topic-matching, and it would collapse
   the moment someone pasted a real political story. The exam (McIntire) is
   political news, so we keep the real side political news too. That is what
   NEWS_OUTLETS is for; --all-outlets turns the restriction off if you want to
   test the idea yourself.

3. NO ONE NEWSROOM TAKES OVER. There is a cap per outlet (--per-outlet).
   Without it the New York Times alone would supply most of the file and we
   would have swapped one dominant newsroom for another.

4. NOTHING FROM THE TEST SET GETS IN. McIntire (fake_or_real_news.csv) is the
   exam. This collection covers 2016-2020 and McIntire is 2016-era political
   news, so overlap is entirely plausible. Every candidate headline is checked
   against the exam using the app's own tidying function
   (analyzer/text_matching.py).

One known trap: Washington Post rows in this collection have EMPTY article
bodies (median 0 characters). It is in EXCLUDED_OUTLETS for that reason, not
because of anything to do with the paper.

Usage
-----
  python fetch_diverse_reals.py                    ~40 MB, writes diverse_reals.csv
  python fetch_diverse_reals.py --target 5000      stop once 5,000 are collected
  python fetch_diverse_reals.py --per-outlet 400   stricter diversity cap
  python fetch_diverse_reals.py --all-outlets      include tech/celebrity too
  python fetch_diverse_reals.py --dry-run          report, write nothing

The output CSV has columns: title, text, label, outlet, url, date
'title'/'text'/'label' match what load_mega_data.py and the notebook read.
"""

import argparse
import collections
import concurrent.futures as futures
import csv
import json
import os
import sys
import urllib.error
import threading
import time
import urllib.parse
import urllib.request

from analyzer.text_matching import normalize_headline

# ---------------------------------------------------------------------------
# The remote collection
# ---------------------------------------------------------------------------

DATASET = "rjac/all-the-news-2-1-Component-one"
ROWS_API = "https://datasets-server.huggingface.co/rows"
SIZE_API = "https://datasets-server.huggingface.co/size"

# 20 rows per request. Larger pages return responses big enough that the API
# refuses them — these articles are thousands of characters each.
PAGE = 20

# Outlets whose writing is general/political news, i.e. the same kind of thing
# the exam is made of. See note 2 in the docstring for why this list exists.
NEWS_OUTLETS = {
    'CNN', 'The Hill', 'Politico', 'Axios', 'The New York Times', 'Vox',
    'CNBC', 'Economist', 'Buzzfeed News', 'Business Insider', 'Vice News',
    'Vice', 'Fox News', 'New Republic', 'New Yorker',
}

# Never wanted, for two different reasons.
EXCLUDED_OUTLETS = {
    'Reuters',          # the flaw we are fixing
    'Washington Post',  # article bodies are empty in this collection
}

DEFAULT_EXCLUDE = ['fake_or_real_news.csv', 'Fake.csv', 'True.csv']

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


# ============================================================================
# READING ROWS OUT OF A 5 GB COLLECTION WITHOUT DOWNLOADING IT
# ============================================================================

# The API's rate limit, measured rather than guessed: sending requests as fast
# as they would go got HTTP 429 after 36 successes in 45 seconds, which is about
# 0.8 per second. There is no Retry-After header to obey, so we simply stay
# under that rate ourselves. 1.4s between requests is ~0.71/sec, a comfortable
# margin, and it makes the whole run predictable instead of a gamble.
MIN_INTERVAL = 1.4

# Waits before each retry, in seconds. Long, because a 429 means we have already
# used up the window and a two-second pause would just be refused again.
RETRY_WAITS = [20, 45, 90]

_rate_lock = threading.Lock()
_last_call = [0.0]


def _wait_turn():
    """Hold every caller to MIN_INTERVAL apart, however many threads there are."""
    with _rate_lock:
        gap = time.time() - _last_call[0]
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        _last_call[0] = time.time()


def api_get(url, timeout=90, retries=True):
    """
    Fetch one page, retrying politely if the server pushes back.

    Without this a single refused request silently costs 20 articles, and the
    refusals arrive in bursts, so the loss is never small.
    """
    attempts = RETRY_WAITS if retries else []
    for wait in list(attempts) + [None]:
        try:
            _wait_turn()
            req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception:
            if wait is None:
                raise
            time.sleep(wait)


def total_rows():
    """How many articles the collection holds, asked rather than assumed."""
    url = f"{SIZE_API}?dataset={urllib.parse.quote(DATASET, safe='')}"
    try:
        data = api_get(url, timeout=60)
        n = data.get('size', {}).get('dataset', {}).get('num_rows')
        if n:
            return int(n)
    except Exception as exc:
        print(f"  could not ask how big it is ({type(exc).__name__})")
    print("  falling back to the size measured on 19 Aug 2026")
    return 2_688_878


def fetch_page(offset):
    """One page of articles, starting at `offset`."""
    url = (f"{ROWS_API}?dataset={urllib.parse.quote(DATASET, safe='')}"
           f"&config=default&split=train&offset={offset}&length={PAGE}")
    data = api_get(url)
    if 'rows' not in data:
        raise urllib.error.URLError(str(data.get('error', 'no rows in reply'))[:120])
    return [item['row'] for item in data['rows']]


def collect(size, pages, workers=2):
    """
    Read `pages` pages from places spread across the whole collection.

    Spread out deliberately: the collection is grouped by outlet, so twenty
    pages from twenty different places gives twenty different newsrooms, while
    one long read from the start gives one.
    """
    step = max(1, size // pages)
    offsets = [i * step for i in range(pages)]

    rows, failed, done = [], 0, 0
    with futures.ThreadPoolExecutor(workers) as pool:
        jobs = [pool.submit(fetch_page, off) for off in offsets]
        for job in futures.as_completed(jobs):
            done += 1
            try:
                rows.extend(job.result())
            except Exception as exc:
                failed += 1
                if failed <= 3:
                    print(f"    a page failed ({type(exc).__name__}) — carrying on")
            if done % max(1, pages // 10) == 0:
                print(f"    {done}/{pages} pages, {len(rows):,} articles so far")

    if failed:
        print(f"    {failed} of {pages} pages failed; kept what the rest gave us")
    return rows


# ============================================================================
# WHAT WE ARE NOT ALLOWED TO KEEP
# ============================================================================

def load_excluded_headlines(paths):
    """Tidied headlines we must not hand back — above all, the exam."""
    blocked = set()
    for path in paths:
        if not os.path.exists(path):
            print(f"  {path}: not found, skipping")
            continue
        before = len(blocked)
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            if 'title' not in (reader.fieldnames or []):
                print(f"  {path}: no 'title' column, skipping")
                continue
            for row in reader:
                key = normalize_headline(row.get('title'))
                if key:
                    blocked.add(key)
        print(f"  {path}: {len(blocked) - before:,} headlines to keep out")
    return blocked


# ============================================================================
# CHOOSING WHAT TO KEEP
# ============================================================================

def select(rows, wanted_outlets, per_outlet, min_chars, blocked):
    """
    Filter down to a diverse set of real articles.

    Returns the kept rows plus a tally of why everything else went — "we kept
    9,000 of 25,000" only means something if you can see where the rest went.
    """
    kept, seen, per_pub = [], set(), collections.Counter()
    why = collections.Counter()

    # Rarest outlets first, so the cap spends its budget on variety rather than
    # being used up by whichever newsroom the early pages happened to land on.
    counts = collections.Counter(r.get('publication') for r in rows)
    rows = sorted(rows, key=lambda r: counts[r.get('publication')])

    for row in rows:
        outlet = (row.get('publication') or '').strip()

        if outlet in EXCLUDED_OUTLETS:
            why[f"excluded outlet ({outlet})"] += 1
            continue
        if wanted_outlets is not None and outlet not in wanted_outlets:
            why["not a general-news outlet"] += 1
            continue

        body = (row.get('article') or '').strip()
        if len(body) < min_chars:
            why["too short to learn from"] += 1
            continue

        title = (row.get('title') or '').strip()
        key = normalize_headline(title)
        if not key:
            why["no usable headline"] += 1
            continue
        if key in blocked:
            why["IN THE TEST SET - excluded"] += 1
            continue
        if key in seen:
            why["duplicate of one we kept"] += 1
            continue
        if per_pub[outlet] >= per_outlet:
            why["outlet already at its cap"] += 1
            continue

        seen.add(key)
        per_pub[outlet] += 1
        kept.append({
            'title': title,
            'text': body,
            'label': 'REAL',
            'outlet': outlet,
            'url': row.get('url') or '',
            'date': row.get('date') or '',
        })

    return kept, why, per_pub


# ============================================================================
# COMMAND LINE
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect real news articles from many different newsrooms.")
    p.add_argument('--pages', type=int, default=1300,
                   help=f"How many places to read from, {PAGE} articles each. "
                        f"Default 1300, so about 40 MB.")
    p.add_argument('--target', type=int, default=0,
                   help="Stop once this many articles are kept. 0 means no limit.")
    p.add_argument('--per-outlet', type=int, default=800,
                   help="Most articles to keep from any one newsroom. Default 800. "
                        "This is what stops one paper dominating.")
    p.add_argument('--min-chars', type=int, default=400,
                   help="Skip articles shorter than this. Default 400.")
    p.add_argument('--all-outlets', action='store_true',
                   help="Include tech and celebrity outlets too. Off by default — "
                        "see note 2 in the file header for why.")
    p.add_argument('--exclude', nargs='+', default=DEFAULT_EXCLUDE,
                   help="CSVs whose headlines must not appear in the output.")
    p.add_argument('--out', default='diverse_reals.csv',
                   help="Where to write the result. Default diverse_reals.csv")
    p.add_argument('--workers', type=int, default=2,
                   help="Parallel requests. Keep this LOW. The API rejects bursts: "
                        "5 workers lost 85%% of pages, 2 works fine. Default 2.")
    p.add_argument('--merge', action='store_true',
                   help="Add to the existing --out file instead of replacing it. "
                        "Use this to build the set up over several runs, which is "
                        "the practical way to work around the source's rate limit.")
    p.add_argument('--dry-run', action='store_true',
                   help="Report what would be kept and write no file.")
    return p.parse_args()


def run():
    args = parse_args()
    wanted = None if args.all_outlets else NEWS_OUTLETS

    print("Collecting real articles from many newsrooms")
    print(f"  plan: {args.pages} pages x {PAGE} rows = up to "
          f"{args.pages * PAGE:,} articles read")
    print(f"  outlets: {'all of them' if wanted is None else 'general news only'}")
    print(f"  never: {', '.join(sorted(EXCLUDED_OUTLETS))}")
    print(f"  cap per outlet: {args.per_outlet}")
    print()

    print("Headlines we must keep out (the test set above all):")
    blocked = load_excluded_headlines(args.exclude)
    print(f"  {len(blocked):,} headlines blocked in total")
    print()

    print("Asking how big the collection is ...")
    size = total_rows()
    print(f"  {size:,} articles; we will look at "
          f"{args.pages * PAGE / size * 100:.2f}% of them")
    print()

    print("Reading pages ...")
    rows = collect(size, args.pages, workers=args.workers)
    if not rows:
        print("\nNothing came back. The API or the dataset may have changed.")
        sys.exit(1)
    print(f"  {len(rows):,} articles read from "
          f"{len(set(r.get('publication') for r in rows))} outlets")
    print()

    kept, why, per_pub = select(
        rows, wanted, args.per_outlet, args.min_chars, blocked)

    if args.target and len(kept) > args.target:
        kept = kept[:args.target]
        per_pub = collections.Counter(k['outlet'] for k in kept)
        print(f"  trimmed to the --target of {args.target:,}")
        print()

    print("What happened to the rest:")
    for reason, n in why.most_common():
        print(f"  {n:7,}  {reason}")
    print()

    if not kept:
        print("Nothing survived the filters. Try --pages 2000 or --all-outlets.")
        sys.exit(1)

    print("Result:")
    print(f"  {len(kept):,} real articles")
    print(f"  from {len(per_pub)} different newsrooms")
    print()
    print(f"  {'newsroom':24} {'articles':>9}")
    print("  " + "-" * 34)
    for outlet, n in per_pub.most_common():
        print(f"  {outlet:24} {n:>9,}")
    print()

    if args.dry_run:
        print("--dry-run given: no file written.")
        return

    # Accumulate across runs. The source throttles hard, so collecting 9,000
    # articles in one pass is fragile; several passes that each add what they
    # can is not. Duplicates are dropped on tidied headline.
    if args.merge and os.path.exists(args.out):
        existing = list(csv.DictReader(
            open(args.out, encoding='utf-8', errors='replace', newline='')))
        have = {normalize_headline(r.get('title')) for r in existing}
        added = [k for k in kept if normalize_headline(k['title']) not in have]
        print(f"--merge: {len(existing):,} already in {args.out}, "
              f"adding {len(added):,} new ones "
              f"({len(kept) - len(added):,} were already there)")
        kept = existing + added
        per_pub = collections.Counter(k['outlet'] for k in kept)
        print(f"  combined total: {len(kept):,} from {len(per_pub)} newsrooms")
        print()

    with open(args.out, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(
            fh, fieldnames=['title', 'text', 'label', 'outlet', 'url', 'date'])
        writer.writeheader()
        writer.writerows(kept)

    print(f"Written to {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print()
    print("Nothing has been trained or loaded yet — this is just a file on disk.")
    print("Next: python test_diverse_reals.py")


if __name__ == '__main__':
    run()
