"""
Ask the internet whether a story is real.

Meant for the ~15% of articles inside cutoff.json's `unsure_band`, where the
model is right only 61% of the time against 93% outside it. Every call spends one
of 500 free grounded searches a day, and the confident articles are already right.

Asks Gemini with Google Search on, and asks a narrower question than "is this
fake": WHO is reporting this, and what are they saying. Two traps make a naive
"did I find it online" check worse than useless:

  1. A viral fabrication is covered heavily — as a debunking. Counting hits on
     "bleach cures cancer" concludes "widely reported, must be real". So outlets
     REPORTING and outlets DEBUNKING come back as two separate lists.
  2. Real news from an hour ago has no corroboration yet, so thin evidence
     returns UNCLEAR and never FAKE.

Produces no number and never touches the model's verdict: that one is measured on
thousands of articles with known answers, this is measured on nothing. The app
shows both and says plainly when they disagree.

Failure is normal and never an exception — no key, no network, timeout, exhausted
quota all return status 'unavailable'. That is NOT the same answer as 'nothing':
"no outlet is carrying this" is evidence against an article, "we could not check"
is evidence about nothing.

Standalone: python -m analyzer.web_check "Some headline to look up"
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

# 2.5 Flash, not a 3.x model: as of 24 Aug 2026 the free tier offers grounded
# search only on 2.5 Flash and Flash-Lite, so "upgrading" here costs money.
MODEL = 'gemini-2.5-flash'
ENDPOINT = ('https://generativelanguage.googleapis.com/v1beta/models/'
            '{model}:generateContent')

# Someone is watching a page load, and the model's answer is already ready.
TIMEOUT = 12

# The headline plus the opening carries the claim; the rest costs tokens.
MAX_CHARS = 1500

# Where the key is read from, in order. Never hardcoded, never committed.
ENV_VAR = 'GEMINI_API_KEY'
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV = os.path.join(_HERE, '.env')
CUTOFF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'reading_model', 'cutoff.json')

PROMPT = """You are helping check whether a news story is genuine. Search the web.

Do NOT judge the writing style. Judge only what you can find: is this story
being reported by news organisations, and what are they saying about it?

Be careful of one trap. A false story is often covered heavily BY DEBUNKINGS.
An outlet publishing "no, this did not happen" is evidence AGAINST the story,
not for it. Keep those two groups separate.

Reply with only a JSON object, no other text, in exactly this shape:

{{
  "verdict": "REAL" or "FAKE" or "UNCLEAR",
  "confidence": "high" or "medium" or "low",
  "reporting": ["outlets reporting this story as something that happened"],
  "debunking": ["outlets or fact-checkers saying it is false or misleading"],
  "factcheck_rating": "the rating if a fact-checker has ruled on it, else null",
  "summary": "one or two plain sentences a general reader can follow"
}}

Rules for the verdict:
- REAL: several independent news organisations report it as having happened.
- FAKE: a fact-checker has rated it false, or credible outlets say it did not
  happen, or the only sources are ones that fabricate stories.
- UNCLEAR: you cannot find it, or you find only one source, or the picture is
  mixed. Recent genuine news often has thin coverage, so thin evidence means
  UNCLEAR and never FAKE.

The story:
HEADLINE: {headline}

OPENING: {body}"""


def _read_key():
    """Environment first, then .env. Returns None if there is no key anywhere."""
    key = os.environ.get(ENV_VAR)
    if key:
        return key.strip()
    try:
        with open(DOTENV, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line.startswith('#') or '=' not in line:
                    continue
                name, _, value = line.partition('=')
                if name.strip() == ENV_VAR:
                    return value.strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def has_key():
    """Whether an online check is possible at all. Cheap; no network."""
    return _read_key() is not None


def unsure_band():
    """
    The score range where the verdict is not worth trusting, as measured by
    tune_cutoff.py. Read from disk on purpose: the band belongs to one set of model
    weights, so numbers typed in here would go stale silently and the app would
    start searching for articles it already had right.
    """
    try:
        with open(CUTOFF_PATH, encoding='utf-8') as fh:
            band = json.load(fh).get('unsure_band')
        if band and 'low' in band and 'high' in band:
            return float(band['low']), float(band['high'])
    except (OSError, ValueError, TypeError):
        pass
    return None


def should_check(fake_score):
    """True when this article falls in the band where the model is unreliable.
    False with no key, no measured band or no score: skip rather than guess at
    when a search is needed."""
    if fake_score is None or not has_key():
        return False
    band = unsure_band()
    if band is None:
        return False
    low, high = band
    return low < float(fake_score) < high


def _blank(status, error=None, elapsed=0.0):
    """A result the template can always render, whatever went wrong."""
    return {
        'status': status,          # 'found' | 'nothing' | 'unavailable'
        'verdict': None,           # 'REAL' | 'FAKE' | 'UNCLEAR' | None
        'confidence': None,
        'summary': '',
        'reporting': [],
        'debunking': [],
        'factcheck_rating': None,
        'sources': [],
        'queries': [],
        'error': error,
        'elapsed': round(elapsed, 2),
    }


def _extract_json(text):
    """Pull the JSON object out of the reply. Asked for bare JSON, models still
    sometimes add a ```json fence or a sentence first, so rather than fail the
    check over formatting, find the outermost braces and parse that."""
    text = re.sub(r'^\s*```(?:json)?|```\s*$', '', text.strip(),
                  flags=re.MULTILINE)
    try:
        return json.loads(text)
    except ValueError:
        pass
    start, end = text.find('{'), text.rfind('}')
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            pass
    return None


def _clean_list(value, limit=8):
    """Whatever the model returned, coerced into a short list of clean strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        name = str(item).strip()
        if name and name.lower() not in ('none', 'null', 'n/a'):
            out.append(name[:80])
    return out[:limit]


def check_online(headline, article_text='', timeout=TIMEOUT):
    """
    Ask the web about this story. Never raises; returns the _blank() dict filled
    in. 'sources' comes from Gemini's grounding metadata rather than its prose —
    the pages it actually consulted, which is the part it cannot invent.
    """
    started = time.time()

    key = _read_key()
    if not key:
        return _blank('unavailable', f'no {ENV_VAR} set')

    headline = (headline or '').strip()
    if not headline:
        return _blank('unavailable', 'no headline to search for')

    prompt = PROMPT.format(headline=headline[:300],
                           body=(article_text or '').strip()[:MAX_CHARS])
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'tools': [{'google_search': {}}],
        # A lookup, not a piece of writing: same story, same answer twice.
        'generationConfig': {'temperature': 0.1},
    }

    url = ENDPOINT.format(model=MODEL) + f'?key={key}'
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        return _blank('unavailable', _http_reason(exc), time.time() - started)
    except Exception as exc:
        # Timeout, DNS failure, no internet, malformed reply — all the same to
        # the caller: the check did not happen.
        return _blank('unavailable', type(exc).__name__, time.time() - started)

    return _read_reply(data, time.time() - started)


def _http_reason(exc):
    """A short, plain reason from an HTTP error, for logs and the page."""
    try:
        message = json.loads(exc.read().decode())['error'].get('message', '')
    except Exception:
        message = ''
    if exc.code == 429:
        return 'daily free search allowance used up'
    if exc.code in (401, 403):
        return 'API key rejected'
    return f'HTTP {exc.code}: {message[:120]}' if message else f'HTTP {exc.code}'


def _read_reply(data, elapsed):
    """Turn Gemini's reply into the flat dict the app renders."""
    try:
        candidate = data['candidates'][0]
        text = ''.join(part.get('text', '')
                       for part in candidate['content']['parts'])
    except (KeyError, IndexError, TypeError):
        return _blank('unavailable', 'reply had no usable content', elapsed)

    result = _blank('found', None, elapsed)

    # The pages actually consulted. Trustworthy in a way the prose is not, so
    # these are kept even if the JSON parse below fails completely.
    meta = candidate.get('groundingMetadata') or {}
    for chunk in (meta.get('groundingChunks') or [])[:8]:
        web = chunk.get('web') or {}
        if web.get('uri'):
            result['sources'].append({
                'title': (web.get('title') or web['uri'])[:100],
                'url': web['uri'],
            })
    result['queries'] = _clean_list(meta.get('webSearchQueries'), limit=5)

    parsed = _extract_json(text)
    if not isinstance(parsed, dict):
        # Grounding worked but the shape did not. Keep the sources, admit the
        # rest is missing, rather than inventing a verdict.
        result['status'] = 'found' if result['sources'] else 'unavailable'
        result['verdict'] = 'UNCLEAR'
        result['summary'] = text.strip()[:300]
        result['error'] = 'could not read the reply as JSON'
        return result

    verdict = str(parsed.get('verdict', '')).strip().upper()
    result['verdict'] = verdict if verdict in ('REAL', 'FAKE', 'UNCLEAR') else 'UNCLEAR'

    confidence = str(parsed.get('confidence', '')).strip().lower()
    result['confidence'] = confidence if confidence in ('high', 'medium', 'low') else None

    result['reporting'] = _clean_list(parsed.get('reporting'))
    result['debunking'] = _clean_list(parsed.get('debunking'))
    result['summary'] = str(parsed.get('summary') or '').strip()[:400]

    rating = parsed.get('factcheck_rating')
    if rating and str(rating).strip().lower() not in ('null', 'none', 'n/a', ''):
        result['factcheck_rating'] = str(rating).strip()[:60]

    # 'nothing' is its own answer: the search ran and came back empty. That is
    # weak evidence against the story, and the page says so differently from
    # "we could not check".
    if (not result['reporting'] and not result['debunking']
            and not result['factcheck_rating'] and result['verdict'] == 'UNCLEAR'):
        result['status'] = 'nothing'

    return result


def _demo(headline, body=''):
    print('=' * 70)
    print(f'  {headline[:66]}')
    print('=' * 70)
    result = check_online(headline, body)
    print(f"  status     : {result['status']}")
    print(f"  verdict    : {result['verdict']}  ({result['confidence']})")
    print(f"  took       : {result['elapsed']}s")
    if result['error']:
        print(f"  error      : {result['error']}")
    if result['factcheck_rating']:
        print(f"  fact-check : {result['factcheck_rating']}")
    if result['reporting']:
        print(f"  reporting  : {', '.join(result['reporting'])}")
    if result['debunking']:
        print(f"  debunking  : {', '.join(result['debunking'])}")
    if result['summary']:
        print(f"  summary    : {result['summary'][:200]}")
    if result['queries']:
        print(f"  searched   : {result['queries']}")
    print(f"  sources    : {len(result['sources'])}")
    for source in result['sources'][:4]:
        print(f"     - {source['title']}")
    print()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        _demo(' '.join(sys.argv[1:]))
    else:
        print(f"key found: {has_key()}   unsure band: {unsure_band()}")
        print()
        _demo('Scientists confirm drinking bleach cures all known cancers',
              'Researchers have PROVEN that household bleach eliminates every '
              'form of cancer within 48 hours. Big Pharma has suppressed this '
              'miracle cure for decades.')
        _demo('NASA lands Perseverance rover on Mars',
              'The Perseverance rover touched down in Jezero Crater after a '
              'seven month journey, NASA confirmed.')
