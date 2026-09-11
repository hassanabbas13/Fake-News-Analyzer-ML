"""
Fetch the two big files if they are missing.

Only needed when `git clone` did NOT bring them down. They ship through git-lfs,
so normally a clone already has everything and running this prints "nothing to
do". The case it exists for: GitHub's free LFS allowance is 1GB of bandwidth a
month, and these two files are ~361MB together, so after roughly three clones in
a month LFS downloads start failing and the files arrive as small text pointers
instead of the real thing.

This script notices that and downloads them from the GitHub Release instead,
which has no bandwidth limit.

    python setup.py             fetch whatever is missing
    python setup.py --check     say what is missing, download nothing
    python setup.py --force     download again even if present

Nothing here touches the training CSVs. Those are only needed to retrain the
model and are deliberately not shipped; see README.md.
"""

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request

RELEASE = ('https://github.com/hassanabbas13/Fake-News-Analyzer-/'
           'releases/download/v1.0')

HERE = os.path.dirname(os.path.abspath(__file__))

# expected_mb is a floor, not an exact size: it only has to be big enough to
# tell a real file apart from an LFS pointer (a few hundred bytes) or a half
# finished download.
FILES = [
    {
        'path': 'analyzer/reading_model/model.safetensors',
        'url': f'{RELEASE}/model.safetensors',
        'expected_mb': 200,
        'what': 'the trained reading model',
        'without_it': 'the app falls back to the old word-counter',
    },
    {
        'path': 'db.sqlite3',
        'url': f'{RELEASE}/db.sqlite3',
        'expected_mb': 80,
        'what': 'the fact-check database, 59,660 articles',
        'without_it': 'Step 1 finds nothing and every article goes to the model',
    },
]

LFS_POINTER = b'version https://git-lfs'


def looks_like_lfs_pointer(path):
    """True if this is git-lfs's placeholder rather than the real file. Happens
    when the LFS download was skipped or the bandwidth quota was hit."""
    try:
        with open(path, 'rb') as fh:
            return fh.read(len(LFS_POINTER)) == LFS_POINTER
    except OSError:
        return False


def status(entry):
    """'ok', 'missing', 'pointer' or 'truncated' for one file."""
    path = os.path.join(HERE, entry['path'])
    if not os.path.exists(path):
        return 'missing'
    if looks_like_lfs_pointer(path):
        return 'pointer'
    if os.path.getsize(path) < entry['expected_mb'] * 1024 * 1024:
        return 'truncated'
    return 'ok'


def human(num_bytes):
    return f'{num_bytes / (1024 * 1024):.0f}MB'


def download(entry):
    """
    Download one file, showing progress. Writes to a .part file and only moves it
    into place once complete, so an interrupted download cannot leave a corrupt
    model sitting where a working one should be.
    """
    target = os.path.join(HERE, entry['path'])
    partial = target + '.part'
    os.makedirs(os.path.dirname(target) or '.', exist_ok=True)

    print(f"  downloading {entry['path']} ...")

    try:
        with urllib.request.urlopen(entry['url'], timeout=30) as response:
            total = int(response.headers.get('Content-Length', 0))
            done = 0
            with open(partial, 'wb') as out:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        bar = '#' * (pct // 4)
                        print(f'\r    [{bar:<25}] {pct:>3}%  '
                              f'{human(done)} / {human(total)}',
                              end='', flush=True)
            print()
    except urllib.error.HTTPError as exc:
        _cleanup(partial)
        if exc.code == 404:
            print(f"    FAILED: not found at {entry['url']}")
            print(f"    The release may not have been published yet.")
        else:
            print(f'    FAILED: HTTP {exc.code}')
        return False
    except Exception as exc:
        _cleanup(partial)
        print(f'    FAILED: {type(exc).__name__}: {exc}')
        return False

    shutil.move(partial, target)
    print(f'    done, {human(os.path.getsize(target))}')
    return True


def _cleanup(path):
    try:
        os.remove(path)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Fetch the model and database if the clone did not bring them.')
    parser.add_argument('--check', action='store_true',
                        help='report what is missing, download nothing')
    parser.add_argument('--force', action='store_true',
                        help='download again even if the files look fine')
    args = parser.parse_args()

    print()
    print('Fake News Analyzer - checking the two big files')
    print('=' * 62)

    problems = []
    for entry in FILES:
        state = status(entry)
        if state == 'ok' and not args.force:
            size = os.path.getsize(os.path.join(HERE, entry['path']))
            print(f"  OK       {entry['path']}  ({human(size)})")
        else:
            reason = {
                'missing': 'not here',
                'pointer': 'git-lfs placeholder, the real file never downloaded',
                'truncated': 'too small, the download was cut short',
                'ok': 'forced',
            }[state]
            print(f"  NEEDED   {entry['path']}  ({reason})")
            problems.append(entry)

    if not problems:
        print()
        print('Nothing to do. Run: python manage.py runserver')
        print()
        return 0

    if args.check:
        print()
        for entry in problems:
            print(f"  {entry['path']}")
            print(f"      is {entry['what']}")
            print(f"      without it, {entry['without_it']}")
        print()
        print('Run without --check to download.')
        print()
        return 1

    print()
    failed = [entry for entry in problems if not download(entry)]

    print()
    if failed:
        print('Some files could not be downloaded.')
        for entry in failed:
            print(f"  {entry['path']}  -  {entry['what']}")
            print(f"      without it, {entry['without_it']}")
        print()
        print('The app will still start, with those parts degraded.')
        print()
        return 1

    print('All set. Run: python manage.py runserver')
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
