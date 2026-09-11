# Fake News Analyzer

A Django web app that reads a news article and says whether it looks **Likely Real** or **Likely Fake**, then shows exactly how it reached that answer.

The design goal was not the highest possible accuracy score. It was to never quote a number that isn't true. Every figure on the page is read live from a measurement file at request time, so retraining the model changes what users see without anyone editing a template.

---

## How it works

Paste a headline, optionally with the article body below it. Three steps run in order, and the first one that can answer wins.

### Step 1 — Fact-check database

The headline is tidied (lookalike quotes folded, accents stripped, lowercased, punctuation dropped) and looked up in a corpus of **59,660 articles**. If that misses, the article's first 20 tidied words are used as a second key, because headlines get edited and bodies mostly don't. Drop `(VIDEO)` off the end of a title and the headline lookup fails while the body lookup still finds it.

Both are **exact** matches on tidied text, not similarity scores. That's what makes the "100%" on the result page honest: it describes how certain the match is, not how true the story is.

`method = "database"`

### Step 2 — The reading model

A fine-tuned DistilBERT scores how fake the writing looks, from 0.0 to 1.0. Before it sees the text, publisher fingerprints are stripped out (wire datelines, agency names, "Reporting by ... Editing by ...", image credits, tracking URLs). That matters: the previous model's single strongest signal for "real" was literally the word *reuters*, because every real article it trained on came off the Reuters wire. It was recognising a publisher, not reading an article.

Text under 25 words is turned away before the model is reached. The model can't say "that isn't enough to go on" — asked about the single word "news" it answered 99.8% fake.

`method = "reading_model"` (or `"too_short"`)

### Step 3 — Web search second opinion

When the score lands in the band where the model is unreliable, or when the user ticks the box, the headline is looked up online through Gemini with Google Search grounding.

It asks a narrower question than "is this fake": **who is reporting this, and what are they saying?** Outlets *reporting* a story and outlets *debunking* it come back as two separate lists, because a viral fabrication is often covered heavily — as a debunking. Counting hits would conclude "widely reported, must be real."

The result is shown **beside** the verdict, never merged into it, and the page says so plainly when the two disagree. The model's number is measured on thousands of articles with known answers; this one is measured on nothing.

---

## What the percentage means

**It is not the model's confidence.** That was a deliberate decision, and it's the most important thing in the project.

A neural network's own certainty is close to worthless — it routinely reports 99% while being wrong. So the number shown is the **measured reliability of that kind of verdict**: out of every hundred times the model has said "Likely Fake" on an article it had never seen, how many times was it right?

As measured on 2026-08-19 (read live from `analyzer/reading_model/cutoff.json`):

| Verdict | How often it's correct | Share of articles |
|---|---|---|
| Likely Fake | 86.9% | 51.9% |
| Likely Real | 90.1% | 48.1% |
| Overall | 88.4% | |

The two directions are quoted separately on purpose. A single blended accuracy would hide it the moment one verdict became much weaker than the other, which has already happened twice in this project's history.

### Why these numbers are smaller than you'd expect

They're honest, which is expensive.

The model trains on Kaggle ISOT plus two collections scraped specifically to break its habits, then gets tested on **McIntire** — a completely separate dataset, different newsrooms, not one article of which it saw in training. That's called a cross-source test, and it's the only kind that means anything.

Test the same model the easy way, on held-out articles from the collections it trained on, and it scores in the high nineties. That number is a lie. Commit `cb51acc` has the receipts: an earlier model scoring well in-domain managed **67.97%** the first time it was shown outside articles. Adding diverse training data took that to **88.45%**.

McIntire is then split in half again: one half picks the decision cutoff, the other half measures it. A number measured on the same articles used to choose it isn't a measurement. That's why the page says "measured on 3,160 articles" and not 6,321.

### The unsure band

`tune_cutoff.py` also finds the score range where the verdict isn't worth trusting. Currently scores between 0.01 and 0.99 — about **14.8%** of articles, where the model is right only **61.1%** of the time, against **93.2%** outside it. That band is what triggers the automatic web search. Nobody sees the raw score, so nobody could tell which articles are guesses; the app has to notice for them.

---

## Input rules

The model was only ever taught to tell English fake news from English real news. It has no way to say "I can't read this", so anything unfamiliar comes back as a confident **Likely Fake** — with a reliability figure next to it that was measured on English news. Refusing is the only honest option.

| Check | Limit | Why |
|---|---|---|
| Latin script | 85% of letters | Urdu, Hindi, Arabic, Chinese and Russian all score 0.000. Real English news never fell below 0.973. |
| English function words | 10% of words | Catches Spanish (0.015), French (0.033), German (0.041), romanised Urdu (0.019), plus keyboard mash and emoji. Cost: 4 real articles rejected out of 6,147. |
| Length | 25 to 400 words | Under 25 the model has nothing to read. Over 400 is padding; it only reads the first ~200 words either way. |

The 25-word floor is applied **after** Step 1, not in the form. A four-word headline is a perfectly good thing to paste if the database can look it up.

---

## Setup

Requires **Python 3.12** and **[Git LFS](https://git-lfs.com)**, which is what carries the trained model and the database.

```bash
git lfs install
git clone https://github.com/hassanabbas13/Fake-News-Analyzer-.git
cd Fake-News-Analyzer-

python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

That is the whole setup. The trained model and the 59,660-article database come down with the clone, so the app works properly from the first run — no training, no data loading.

> `torch` and `transformers` are roughly 2.5 GB installed. They are what runs the reading model, so skipping them leaves the app able to answer only from the database.

### If the model or database didn't arrive

GitHub gives free accounts 1 GB of LFS bandwidth a month, and these two files are ~361 MB together. After roughly three clones in a month, LFS stops serving them and you get small text placeholders instead of the real files.

```bash
python setup.py --check    # says whether anything is missing, downloads nothing
python setup.py            # fetches them from the GitHub Release instead
```

The Release has no bandwidth limit, so this always works. The script writes to a `.part` file and only moves it into place when the download completes, so an interrupted download can't leave a corrupt model behind.

If you skip it: a missing database means Step 1 never matches, and a missing model means the app says **"analysis unavailable"** rather than guessing. It won't invent a verdict to fill the gap.

### Optional: the web search second opinion

Step 3 needs a Gemini API key. Without one the checkbox doesn't render at all, rather than offering a box that could only ever fail.

```bash
echo "GEMINI_API_KEY=your-key-here" > .env
```

`.env` is gitignored and the key is never read from anywhere else. The free tier allows 500 grounded searches a day, which is why the automatic search only fires inside the unsure band.

Use `gemini-2.5-flash`, not a 3.x model: as of 2026-08-24, grounded Google Search is free only on 2.5 Flash and Flash-Lite. "Upgrading" silently starts costing money.

---

## Rebuilding things from scratch

Nothing in this section is needed to run the app. It's here for changing how it works.

### The training data

CSVs are not shipped (~206 MB, and only needed to retrain). Put them in the project root:

| File | Source |
|---|---|
| `Fake.csv`, `True.csv` | Kaggle ISOT Fake News Dataset |
| `fake_or_real_news.csv` | McIntire — **the exam**, never train on this |
| `diverse_fakes.csv` | `python fetch_diverse_fakes.py` |
| `diverse_reals.csv` | `python fetch_diverse_reals.py` |

### Rebuilding the database

```bash
python load_mega_data.py
```

This **tops up**. It adds what it doesn't already have and leaves the rest alone, so running it twice is harmless and hand-corrected rows survive. Duplicates are caught on the tidied headline, so a curly-vs-straight apostrophe can't sneak one in.

```bash
python load_mega_data.py --list                       # what's loaded, which files are present
python load_mega_data.py --only true                  # read just one known file
python load_mega_data.py --file bbc.csv --source "BBC" --label REAL
python load_mega_data.py --file checked.csv --source "PolitiFact" --basis fact_check
python load_mega_data.py --reset                      # wipe first (destructive, opt-in)
```

Use `--basis fact_check` only when a fact-checker actually investigated the claim. Those rows outrank publisher-inferred ones when both exist for the same headline.

### Retraining the reading model

```bash
# 1. Run train_reading_model_v3.ipynb on Colab, with the T4 GPU turned on.
#    TRIAL_RUN = True finishes in about 4 minutes on 4,000 articles;
#    the full run is roughly 30-45 minutes. Keep the tab open.
# 2. Download the output into analyzer/reading_model/
# 3. Measure it — the app refuses to use the model without this:
python tune_cutoff.py
```

`tune_cutoff.py` writes `analyzer/reading_model/cutoff.json`. Without that file the app won't touch the reading model at all. That's deliberate: no measurement means no honest figure to put beside a verdict, and quoting the model's own confidence is exactly what this project refuses to do.

### The old word-counter

A TF-IDF classifier, kept as a fallback for machines that can't install PyTorch. **Not shipped**, and off by default, because it is much weaker than the reading model — it rates an ordinary council-funding story as fake at 51%, which is a coin toss. If you want it anyway:

```bash
python train_honest_model.py    # writes model.pkl, vectorizer.pkl, model_meta.json
```

Build it and the app uses it only when the reading model can't load, and labels the result as having come from the word-counter.

---

## Tests

```bash
python manage.py test analyzer      # 156 tests, temporary database
python -m analyzer.input_check      # language checks against 6 multilingual samples
python -m analyzer.web_check "some headline"   # live API check, needs a key
python inspect_model.py             # strongest word weights in the fallback model
```

`REAL TESTING.md` has ten hand-written articles for manual checks, five real and five fake, none of which are in the database so the model is forced to answer. `news articles.md` has twenty pulled from the database, for testing Step 1.

---

## Known limitations

Read this before trusting anything on the screen.

- **"REAL" in the database means "published by a real newsroom", not "fact-checked".** All 59,660 rows are currently labelled by source. Zero are `basis = fact_check`. The 100% is a statement about provenance, not verified truth.
- **A "Likely Real" verdict is not clearance.** The model never checked whether any of it happened. A careful lie in wire-service prose passes. The result page says this; it's not a footnote.
- **Step 1 is exact matching.** Two keys instead of one helps, but a reworded headline on a paraphrased body still falls through to the model.
- **The web check is unmeasured.** It has no accuracy figure because nobody has ever scored it. That's why it's shown beside the verdict and never folded into it.
- **The dashboard percentages are not accuracy.** Nobody has told the app whether any verdict was right. "64% Likely Fake" means 64% of what people pasted came back fake, which says more about what people paste.
- **English news only.** Not a temporary gap. See the input rules above.

---

## Layout

```
analyzer/
  analysis_engine.py    the three-step classifier (core logic)
  web_check.py          Step 3: Gemini + Google Search grounding
  input_check.py        is this English news, and is there enough of it
  text_matching.py      the ONE place text gets tidied before comparing
  text_cleaning.py      strips publisher fingerprints, used by training AND serving
  models.py             KnownArticle (corpus) + NewsAnalysis (saved results)
  views.py              request handling
  forms.py              the input form
  tests.py              156 tests
  templates/analyzer/   base, home, analyze, result, dashboard
  static/analyzer/      neo-brutalist stylesheet, no Bootstrap CSS
  reading_model/        fine-tuned DistilBERT + cutoff.json   (ships via git-lfs)
  model.pkl             optional TF-IDF fallback               (not shipped)

fake_news_project/      settings, urls, wsgi
db.sqlite3              the 59,660-article corpus             (ships via git-lfs)

setup.py                fetches the two big files if the clone missed them
load_mega_data.py       CSVs -> KnownArticle (safe to re-run)
fetch_diverse_fakes.py  scrapes fake articles from hundreds of sites
fetch_diverse_reals.py  scrapes real articles from many newsrooms
train_reading_model_v3.ipynb   trains the DistilBERT (Colab) — the live one
                               v1 and v2 are kept for history only
tune_cutoff.py          measures it, writes cutoff.json
train_honest_model.py   builds the fallback word-counter
test_diverse_fakes.py   proves the extra fake data helped, against a control
test_diverse_reals.py   proves the extra real data helped, against two controls
inspect_model.py        prints the word-counter's strongest weights
```

`text_cleaning.py` is imported by both the training scripts and the live app on purpose. If the two ever cleaned text differently, the model would be fed input it had never seen and its answers would quietly turn to noise, with nothing failing loudly enough to notice.

---

## Tech stack

Python 3.12 · Django 6.0.5 · PyTorch 2.10 · Transformers 5.1 · DistilBERT · scikit-learn 1.8 · Gemini 2.5 Flash · SQLite · Chart.js
