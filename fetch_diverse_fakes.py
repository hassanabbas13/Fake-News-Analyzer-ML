"""
fetch_diverse_fakes.py — Collect fake articles from hundreds of websites
=======================================================================

Why this exists
---------------
The model is good at spotting real news and weak at spotting fakes. That is a
data problem, not a model problem: every fake article it has ever seen came
from the same small handful of places, so it learned a few house styles instead
of learning what a fake article actually looks like.

This script fixes the fake side of the training data by pulling fake articles
from hundreds of different websites.

Where the articles come from
----------------------------
The OpenSources/FakeNewsCorpus collection, mirrored as one big CSV on Hugging
Face. Every article carries the label of the website that published it:

    fake, conspiracy, junksci, rumor, unreliable   <- we want these
    satire, bias, hate, clickbait, political       <- we skip these, see below
    reliable                                       <- only 9 websites, skip

That file is 20 GB, which is far too big to download. But it is one plain CSV
served over ordinary HTTP, so we can ask for slices out of the middle of it and
never fetch the rest. About 100 MB of slices, spread evenly through the file,
gives a wide sample of websites. That is the whole trick.

Three things this script is careful about
-----------------------------------------
1. NO ONE WEBSITE TAKES OVER. There is a cap on how many articles we keep per
   website (--per-site). Without it, whichever site happens to be biggest would
   dominate and we would be right back to learning one house style.

2. NOTHING FROM THE TEST SET GETS IN. McIntire (fake_or_real_news.csv) is the
   exam this project is scored on. If an article from it leaked into training,
   the score would go up while the model got no better. Every candidate has its
   headline checked against the exam papers first, using the same tidying
   function the app itself uses (analyzer/text_matching.py).

3. STANCE IS NOT THE SAME AS FALSEHOOD. 'bias' and 'hate' describe how angry
   or one-sided a site is, not whether its facts are wrong. Training on those
   teaches "angry writing = fake", which is not what we want. 'satire' is
   skipped for a related reason: The Onion is deliberately false but it is
   written to be funny, and that comedy style is not what real disinformation
   looks like. Add them with --types if you want to test that yourself.

Usage
-----
  python fetch_diverse_fakes.py                       ~100 MB, writes diverse_fakes.csv
  python fetch_diverse_fakes.py --slices 100          quicker, smaller sample
  python fetch_diverse_fakes.py --per-site 30         stricter diversity cap
  python fetch_diverse_fakes.py --types fake satire   choose the labels yourself
  python fetch_diverse_fakes.py --dry-run             report what it would keep, write nothing

The output CSV has columns: title, text, label, domain, type, url
'title'/'text'/'label' are named to match what load_mega_data.py already reads.
"""

import argparse
import collections
import concurrent.futures as futures
import csv
import io
import os
import re
import sys
import urllib.error
import urllib.request

from analyzer.text_matching import normalize_headline

# ---------------------------------------------------------------------------
# The remote file
# ---------------------------------------------------------------------------

SOURCE_URL = (
    "https://huggingface.co/datasets/andyP/fake_news_en_opensources"
    "/resolve/main/opensources_fake_news_cleaned.csv"
)

# Column order in that CSV. Checked against its real header line, not guessed.
SOURCE_COLS = ["id", "type", "domain", "scraped_at", "url", "authors", "title", "content"]

# Website labels we treat as FAKE. See note 3 in the docstring for the ones
# left out and why.
DEFAULT_TYPES = ["fake", "conspiracy", "junksci", "rumor", "unreliable"]

# Files whose headlines must never appear in the output. McIntire is the exam;
# the ISOT pair is already loaded, so re-adding it wastes space.
DEFAULT_EXCLUDE = ["fake_or_real_news.csv", "Fake.csv", "True.csv"]

# A record in this CSV starts at a line beginning with:
#   <id>,<type>,<domain>,<yyyy-mm-dd>,
# Article bodies contain newlines, so a bare "\n" is NOT a safe record
# boundary. This pattern is specific enough to find a real one.
RECORD_START = re.compile(rb"\n(\d{4,},[a-z]*,[^,\n]{3,60},\d{4}-\d{2}-\d{2},)")

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


# ============================================================================
# READING SLICES OUT OF THE MIDDLE OF A 20 GB FILE
# ============================================================================

def remote_size(url):
    """Ask how big the file is without downloading any of it."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": BROWSER_UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        size = resp.headers.get("Content-Length") or resp.headers.get("X-Linked-Size")
    if not size:
        print("ERROR: the server would not say how big the file is, so we cannot")
        print("       work out where to take slices from.")
        sys.exit(1)
    return int(size)


def fetch_slice(url, offset, nbytes):
    """Download just the bytes from `offset` to `offset + nbytes`."""
    req = urllib.request.Request(url, headers={
        "Range": f"bytes={offset}-{offset + nbytes - 1}",
        "User-Agent": BROWSER_UA,
    })
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 206:
            raise urllib.error.URLError(
                f"server ignored the range request (status {resp.status})"
            )
        return resp.read()


def parse_slice(chunk):
    """
    Turn one downloaded chunk into whole article records.

    A chunk starts and ends mid-article, so both ends are unusable: we skip
    forward to the first real record boundary and throw away the final record,
    which is always cut off partway through.
    """
    start = RECORD_START.search(chunk)
    if not start:
        return []

    text = chunk[start.start(1):].decode("utf-8", errors="replace")
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error:
        return []

    out = []
    for row in rows[:-1]:                       # last one is truncated
        if len(row) == len(SOURCE_COLS) and row[0].isdigit():
            out.append(dict(zip(SOURCE_COLS, row)))
    return out


def collect(url, size, slices, slice_bytes, workers=8):
    """
    Download `slices` evenly-spaced chunks and return every article in them.

    Spread out on purpose: the file groups articles by website, so a hundred
    slices from a hundred different places gives a hundred different
    neighbourhoods, while one big read from the start gives one website.
    """
    step = size // slices
    offsets = [i * step + 1000 for i in range(slices)]

    records, failed, done = [], 0, 0
    with futures.ThreadPoolExecutor(workers) as pool:
        jobs = [pool.submit(fetch_slice, url, off, slice_bytes) for off in offsets]
        for job in futures.as_completed(jobs):
            done += 1
            try:
                records.extend(parse_slice(job.result()))
            except Exception as exc:
                failed += 1
                if failed <= 3:
                    print(f"    a slice failed ({type(exc).__name__}) — carrying on")
            if done % max(1, slices // 10) == 0:
                print(f"    {done}/{slices} slices, {len(records):,} articles so far")

    if failed:
        print(f"    {failed} of {slices} slices failed; kept what the rest gave us")
    return records


# ============================================================================
# WHAT WE ARE NOT ALLOWED TO KEEP
# ============================================================================

def load_excluded_headlines(paths):
    """
    Tidied headlines of every article we must not hand back — above all the
    test set. Missing files are reported and skipped, not fatal: someone may
    reasonably not have all of them on disk.
    """
    blocked = set()
    for path in paths:
        if not os.path.exists(path):
            print(f"  {path}: not found, skipping (nothing from it can be excluded)")
            continue
        before = len(blocked)
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            column = "title" if "title" in (reader.fieldnames or []) else None
            if column is None:
                print(f"  {path}: no 'title' column, skipping")
                continue
            for row in reader:
                key = normalize_headline(row.get(column))
                if key:
                    blocked.add(key)
        print(f"  {path}: {len(blocked) - before:,} headlines to keep out")
    return blocked


# ============================================================================
# CHOOSING WHAT TO KEEP
# ============================================================================

def select(records, wanted_types, per_site, min_chars, blocked):
    """
    Filter down to a diverse set of fake articles.

    Returns the kept rows plus a tally of why everything else went, because
    "we kept 9,000 of 28,000" is only trustworthy if you can see where the
    other 19,000 went.
    """
    wanted = set(wanted_types)
    kept, seen_keys, per_domain = [], set(), collections.Counter()
    why = collections.Counter()

    # Rarest websites first, so the per-site cap spends its budget on variety
    # instead of being used up by whichever site the early slices happened to
    # land on.
    by_domain = collections.Counter(r["domain"] for r in records)
    records = sorted(records, key=lambda r: by_domain[r["domain"]])

    for rec in records:
        if rec["type"] not in wanted:
            why["label we are not using"] += 1
            continue

        body = (rec["content"] or "").strip()
        if len(body) < min_chars:
            why["too short to learn from"] += 1
            continue

        key = normalize_headline(rec["title"])
        if not key:
            why["no usable headline"] += 1
            continue
        if key in blocked:
            why["IN THE TEST SET - excluded"] += 1
            continue
        if key in seen_keys:
            why["duplicate of one we kept"] += 1
            continue

        domain = rec["domain"] or "(unknown)"
        if per_domain[domain] >= per_site:
            why["website already at its cap"] += 1
            continue

        seen_keys.add(key)
        per_domain[domain] += 1
        kept.append({
            "title": rec["title"].strip(),
            "text": body,
            "label": "FAKE",
            "domain": domain,
            "type": rec["type"],
            "url": rec["url"],
        })

    return kept, why, per_domain


# ============================================================================
# COMMAND LINE
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect fake news articles from hundreds of different websites.",
    )
    p.add_argument("--slices", type=int, default=400,
                   help="How many places in the file to read from. More places = "
                        "more websites. Default 400.")
    p.add_argument("--slice-kb", type=int, default=256,
                   help="How much to read at each place, in KB. Default 256, so "
                        "the default run downloads about 100 MB.")
    p.add_argument("--per-site", type=int, default=60,
                   help="Most articles to keep from any one website. Default 60. "
                        "This is what stops one site dominating.")
    p.add_argument("--min-chars", type=int, default=400,
                   help="Skip articles shorter than this. Default 400.")
    p.add_argument("--types", nargs="+", default=DEFAULT_TYPES,
                   help=f"Website labels to treat as fake. Default: {' '.join(DEFAULT_TYPES)}")
    p.add_argument("--exclude", nargs="+", default=DEFAULT_EXCLUDE,
                   help="CSV files whose headlines must not appear in the output. "
                        "The test set belongs here. Default: " + " ".join(DEFAULT_EXCLUDE))
    p.add_argument("--out", default="diverse_fakes.csv",
                   help="Where to write the result. Default diverse_fakes.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be kept and write no file.")
    return p.parse_args()


def run():
    args = parse_args()
    slice_bytes = args.slice_kb * 1024
    budget_mb = args.slices * slice_bytes / 1e6

    print("Collecting fake articles from many websites")
    print(f"  plan: {args.slices} slices x {args.slice_kb} KB = about {budget_mb:.0f} MB to download")
    print(f"  keeping labels: {', '.join(args.types)}")
    print(f"  cap per website: {args.per_site}")
    print()

    print("Headlines we must keep out (the test set above all):")
    blocked = load_excluded_headlines(args.exclude)
    print(f"  {len(blocked):,} headlines blocked in total")
    print()

    print("Checking the remote file ...")
    size = remote_size(SOURCE_URL)
    print(f"  it is {size / 1e9:.1f} GB - we will read {budget_mb / (size / 1e6) * 100:.2f}% of it")
    print()

    print("Downloading slices ...")
    records = collect(SOURCE_URL, size, args.slices, slice_bytes)
    if not records:
        print("\nNothing came back. The download or the file layout may have changed.")
        sys.exit(1)
    print(f"  {len(records):,} whole articles recovered from "
          f"{len(set(r['domain'] for r in records)):,} websites")
    print()

    kept, why, per_domain = select(
        records, args.types, args.per_site, args.min_chars, blocked
    )

    print("What happened to the rest:")
    for reason, n in why.most_common():
        print(f"  {n:7,}  {reason}")
    print()

    if not kept:
        print("Nothing survived the filters. Try --slices 800 or a longer --types list.")
        sys.exit(1)

    print("Result:")
    print(f"  {len(kept):,} fake articles")
    print(f"  from {len(per_domain):,} different websites")
    print(f"  busiest website: {per_domain.most_common(1)[0][0]} "
          f"({per_domain.most_common(1)[0][1]} articles)")
    print(f"  label mix: " + ", ".join(
        f"{t}={n}" for t, n in collections.Counter(k['type'] for k in kept).most_common()
    ))
    print()

    if args.dry_run:
        print("--dry-run given: no file written.")
        return

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["title", "text", "label", "domain", "type", "url"])
        writer.writeheader()
        writer.writerows(kept)

    print(f"Written to {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print()
    print("Nothing has been trained or loaded yet — this is just a file on disk.")


if __name__ == "__main__":
    run()
