# Fake News Analyzer

A Django web app that checks whether a news article is real or fake, and shows how it reached that answer.

Every accuracy figure on the page is read live from a measurement file, so retraining the model updates what users see without editing a template.

---

## How it works

Paste a headline with the article body below it. Three steps run in order, and the first one that can answer wins.

**1. Fact-check database.** The headline is looked up in a corpus of 59,660 articles. If that misses, the article's first 20 words are used as a second key. Both are exact matches on normalised text, so the "100%" on the result page describes the match, not the truth of the story.

**2. The reading model.** A fine-tuned DistilBERT scores how fake the writing looks, from 0.0 to 1.0. Publisher fingerprints (wire datelines, agency names, image credits) are stripped first, so it judges the writing rather than the source. Text under 25 words is turned away.

**3. Web search second opinion.** When the score falls in the range where the model is unreliable, or the user asks for it, the headline is searched through Gemini with Google Search grounding. It returns two separate lists: outlets reporting the story and outlets debunking it. The result appears beside the verdict, never merged into it.

---

## What the percentage means

It is not the model's confidence. It's the measured reliability of that kind of verdict: out of every hundred times the model said "Likely Fake" on an article it had never seen, how often it was right.

Measured 2026-08-19, read live from `analyzer/reading_model/cutoff.json`:

| Verdict | How often it's correct | Share of articles |
|---|---|---|
| Likely Fake | 86.9% | 51.9% |
| Likely Real | 90.1% | 48.1% |
| Overall | 88.4% | |

**These are cross-source numbers.** The model trains on Kaggle ISOT plus two scraped collections, then is tested on McIntire, a separate dataset of different newsrooms it never saw in training. Tested the easy way, on held-out articles from its own training collections, it scores in the high nineties. That figure doesn't survive contact with real articles: an earlier version looked strong in-domain and managed 67.97% on outside sources.

McIntire is split in half again, one half to pick the decision cutoff and the other to measure it, which is why the page says 3,160 articles and not 6,321.

**The unsure band.** `tune_cutoff.py` also finds the range where verdicts aren't worth trusting: scores between 0.01 and 0.99, about 14.8% of articles, where the model is right 61.1% of the time against 93.2% outside it. Articles landing there trigger the web search automatically.

---

## Input rules

The model only ever learned English news, and has no way to signal "I can't read this" — anything unfamiliar comes back as a confident Likely Fake. So unreadable input is refused instead.

| Check | Limit | Measured |
|---|---|---|
| Latin script | 85% of letters | Real English news never fell below 0.973; Urdu, Hindi, Arabic, Chinese, Russian all score 0.000 |
| English function words | 10% of words | English median 0.359, Spanish 0.015, French 0.033, German 0.041, romanised Urdu 0.019 |
| Length | 25 to 400 words | The model reads the first ~200 words |

The 25-word minimum applies only from Step 2 onward, so a short headline is still worth pasting if the database can find it.

---

## Setup

Requires **Python 3.12** and **[Git LFS](https://git-lfs.com)**, which carries the model and database.

```bash
git lfs install
git clone https://github.com/hassanabbas13/Fake-News-Analyzer-ML.git
cd Fake-News-Analyzer-ML

python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

That's the whole setup. The trained model and the 59,660-article database come down with the clone, so the app works from the first run.

`torch` and `transformers` are ~2.5 GB installed. Without them the app answers from the database only.

### If the model or database didn't arrive

GitHub's free LFS allowance is 1 GB of bandwidth a month and these two files are ~379 MB, so after a few clones they arrive as small placeholder files instead.

```bash
python setup.py --check    # report what's missing
python setup.py            # download it from the GitHub Release
```

Without them, the app reports "analysis unavailable" rather than guessing.

### Optional: web search

Step 3 needs a Gemini API key. Without one, the checkbox doesn't appear.

```bash
echo "GEMINI_API_KEY=your-key-here" > .env
```

Uses `gemini-2.5-flash`, where grounded Google Search is available on the free tier (500 searches/day).

---

## Rebuilding from scratch

Not needed to run the app — the shipped model and database are ready to go. This is only for retraining.

The training data is Kaggle ISOT plus scraped fake and real collections (`fetch_diverse_fakes.py`, `fetch_diverse_reals.py`), tested against McIntire, a separate dataset kept out of training. To rebuild:

- **Database:** `python load_mega_data.py` (safe to re-run; adds what isn't already there)
- **Reading model:** run `train_reading_model_v3.ipynb` on Colab, then `python tune_cutoff.py` to measure it and write `cutoff.json`. The app won't use a model without that measurement.
- **Word-counter:** a TF-IDF fallback for machines without PyTorch, off by default. Build with `python train_honest_model.py`.

---

## Tests

```bash
python manage.py test analyzer      # 156 tests
python -m analyzer.input_check      # language checks
python -m analyzer.web_check "some headline"
```

`REAL TESTING.md` holds ten hand-written articles that aren't in the database. `news articles.md` holds twenty that are, for testing Step 1.

---

## Limitations

- **"REAL" means "published by a real newsroom", not "fact-checked".** All 59,660 rows are labelled by source.
- **A "Likely Real" verdict is not clearance.** Nothing here checks whether the events described happened.
- **Step 1 is exact matching.** A reworded headline on a paraphrased body falls through to the model.
- **The web check has no accuracy figure.** It has never been scored, which is why it's shown separately.
- **Dashboard percentages aren't accuracy.** They show what people pasted, not whether the verdicts were right.
- **English news only.**

---

## Layout

```
analyzer/
  analysis_engine.py    the three-step classifier
  web_check.py          Gemini + Google Search grounding
  input_check.py        language and length checks
  text_matching.py      text normalisation for lookups
  text_cleaning.py      strips publisher fingerprints
  models.py             KnownArticle + NewsAnalysis
  views.py              request handling
  forms.py              the input form
  tests.py              156 tests
  templates/analyzer/   base, home, analyze, result, dashboard
  static/analyzer/      stylesheet
  reading_model/        DistilBERT + cutoff.json     (git-lfs)

fake_news_project/      settings, urls, wsgi
db.sqlite3              the 59,660-article corpus    (git-lfs)

setup.py                downloads the model/database if the clone missed them
load_mega_data.py       CSVs -> database
fetch_diverse_fakes.py  scrapes fake articles
fetch_diverse_reals.py  scrapes real articles
train_reading_model_v3.ipynb   trains the DistilBERT (Colab)
tune_cutoff.py          measures it, writes cutoff.json
train_honest_model.py   builds the TF-IDF fallback
test_diverse_fakes.py   measures whether the extra fake data helped
test_diverse_reals.py   measures whether the extra real data helped
```

`text_cleaning.py` is shared by the training scripts and the live app, so text is cleaned identically in both.

---

## Tech stack

Python 3.12 · Django 6.0.5 · PyTorch 2.10 · Transformers 5.1 · DistilBERT · scikit-learn 1.8 · Gemini 2.5 Flash · SQLite · Chart.js
