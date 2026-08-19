# Fake News Analyzer (Django)

A web application that classifies news headlines and articles as **Likely Real** or **Likely Fake** using a two-step hybrid system — a fact-check database lookup, falling back to a trained machine-learning model — and explains which route produced the verdict.

---

## How it works

Paste a headline (optionally with the article body below it) and the analyzer runs two steps:

**Step 1 — Database fact-check.** The headline is normalized (lowercased, trimmed) and looked up against every article in the `KnownArticle` table (roughly 45,000 with the datasets below). An exact match returns that dataset's label immediately, `method = "database"`.

**Step 2 — ML prediction (fallback).** If the headline isn't in the database, the text is vectorized with TF-IDF (50,000 features, English stop-words removed) and classified by a Logistic Regression model. Returns the predicted label plus a confidence percentage from `predict_proba()`, `method = "ml_prediction"`.

The model and vectorizer are unpickled once at import and held in memory, not reloaded per request.

The app never has article counts or accuracy figures typed into it. The corpus size is counted live from the database, and the model's measured scores are read from `analyzer/model_meta.json`, written by `train_mega_model.py`. Reload the data or retrain the model and what users see updates by itself.

Every analysis is saved to the database with its label, confidence, method, and explanation, and is viewable on a dashboard showing the Real/Fake split and how many verdicts came from the database versus the model.

---

## Setup

Requires **Python 3.12**.

```bash
python -m pip install -r requirements.txt
python manage.py migrate
```

### Data files (not in the repo)

Three datasets must be placed in the project root:

| File | Source |
|---|---|
| `Fake.csv`, `True.csv` | Kaggle ISOT Fake News Dataset |
| `fake_or_real_news.csv` | McIntire fake/real news dataset |

Then run the two one-off setup scripts:

```bash
python load_mega_data.py     # adds the articles to KnownArticle
python train_mega_model.py   # writes analyzer/model.pkl, vectorizer.pkl, model_meta.json
```

`load_mega_data.py` **tops up** — it adds articles it doesn't already have and leaves everything else alone, so running it twice is harmless and anything you corrected by hand survives. It recognises articles it already holds via the tidied headline, so a curly-vs-straight apostrophe no longer sneaks in a duplicate.

```bash
python load_mega_data.py --list                          # what's loaded, which files are present
python load_mega_data.py --only true                     # read just one known file
python load_mega_data.py --file bbc.csv --source "BBC" --label REAL
python load_mega_data.py --file checked.csv --source "PolitiFact" --basis fact_check
python load_mega_data.py --reset                         # delete everything first (destructive, opt-in)
```

Use `--basis fact_check` only when a fact-checker actually investigated the claims. Those entries outrank publisher-inferred ones, so a real fact-check will override an ISOT label for the same headline.

Without `model.pkl`/`vectorizer.pkl` the app still starts, but Step 2 is disabled and unknown headlines return `method = "unavailable"`. Without `model_meta.json` the app runs normally but quotes no accuracy figure, rather than inventing one.

### Run

```bash
python manage.py runserver
# then open http://127.0.0.1:8000
```

---

## Tests

```bash
python manage.py test analyzer     # unit tests, run against a temporary database
python test_logic.py               # manual smoke test against the real loaded dataset
```

---

## Known limitations

Read this before trusting the accuracy figure.

- **The measured test accuracy (recorded in `analyzer/model_meta.json`) is inflated, and the model is largely detecting *publisher style*, not deception.** `python inspect_model.py` prints the learned weights: the strongest "real" signals are `reuters` (−26.17), `said`, `tuesday`, `washington` and other weekday datelines; the strongest "fake" signals are `video`, `image`, `featured`, `getty`, `https`. ISOT's `True.csv` is entirely Reuters wire copy, so the model learned newswire boilerplate. Expect a genuine article from another outlet to be flagged Fake, and fabricated text written in wire style to pass as Real.
- **That accuracy is measured on the same corpus the database already covers.** The model is trained on all the articles that are also in `KnownArticle`, including its own held-out split. Step 2 only fires on text *outside* that corpus — precisely where the figure does not apply.
- **Step 1 is exact string matching.** One changed character, or a missing trailing period, and a known headline falls through to the model.
- **"REAL" means "published by Reuters", not "fact-checked."** The datasets are labelled by source, so the 100%-confidence database verdict is a statement about provenance, not about verified truth.

---

## Layout

```
analyzer/
  analysis_engine.py   the two-step hybrid classifier (core logic)
  text_matching.py     the ONE place headlines get tidied before comparing
  models.py            KnownArticle (fact-check corpus) + NewsAnalysis (saved results)
  views.py             request handling
  forms.py             input form
  tests.py             test suite
  templates/analyzer/  home, analyze, result, dashboard pages
  static/analyzer/     styling
  model.pkl            trained Logistic Regression classifier (generated)
  vectorizer.pkl       fitted TF-IDF vectorizer (generated)
  model_meta.json      the model's measured scores (generated)
fake_news_project/
  settings.py, urls.py, wsgi.py
load_mega_data.py      tops up KnownArticle from CSV files (safe to re-run)
train_mega_model.py    CSVs -> model.pkl + vectorizer.pkl + model_meta.json
inspect_model.py       prints the model's strongest word weights
manage.py
```

---

## Tech stack

Python 3.12 · Django 6 · scikit-learn · SQLite · Bootstrap 5 · Chart.js
