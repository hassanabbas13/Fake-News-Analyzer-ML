# Hybrid Fake News Analyzer (Revised Plan)

Replacing the basic keyword scanner with a professional Hybrid System that combines a 46,000+ article fact-check database with a trained Machine Learning model. This revised plan fixes all 7 weaknesses identified in the previous version.

## User Review Required

> [!IMPORTANT]
> This is a significant upgrade. It touches the database, the core logic, the views, and the templates. Please review carefully before approving.

---

## Proposed Changes

### 1. Database Layer

#### [MODIFY] models.py
**New table: `KnownArticle`**
- `headline` — The article headline, stored in **lowercase** (normalized) so searches never fail due to capitalization differences.
- `article_text` — The full article body text.
- `label` — Either "FAKE" or "REAL".
- `source` — Which dataset it came from (e.g., "Kaggle ISOT" or "McIntire").
- `db_index=True` on the headline column for instant searching across 46,000 rows.

**Update existing table: `NewsAnalysis`**
- Add `confidence` field — A number between 0 and 100 representing how sure the system is (e.g., 94.2%).
- Add `method` field — Either "database" (exact match found) or "ml_prediction" (AI had to guess).
- Keep all existing fields so the dashboard and history still work.

---

### 2. Setup Scripts (Run Once)

#### [NEW] load_mega_data.py
- Reads all three CSV files: `Fake.csv`, `True.csv`, and `fake_or_real_news.csv`.
- Standardizes the column names across all three (they have different structures).
- Normalizes all headlines to lowercase and strips extra whitespace.
- Removes duplicate articles that appear in more than one dataset.
- Uses Django's `bulk_create()` to insert all 46,000+ unique articles into the database in seconds instead of hours.

#### [NEW] train_mega_model.py
- Reads all three CSV files and combines them into one mega-dataset.
- Splits the data: **80% for training**, **20% for testing**.
- Converts text into numerical vectors using **TF-IDF** (Term Frequency-Inverse Document Frequency).
- Trains a **Logistic Regression** model on the 80% training data.
- Tests the model on the 20% test data and **prints the accuracy score** (expected 90%+ accuracy).
- Saves two files: `model.pkl` (the trained AI brain) and `vectorizer.pkl` (the text-to-numbers translator).

---

### 3. Core Logic Overhaul

#### [MODIFY] analysis_engine.py
Complete rewrite. The new `analyze_text()` function will work in two steps:

**Step 1 — Database Fact-Check:**
- Normalize the user's input headline (lowercase, strip spaces).
- Search the `KnownArticle` table using the indexed headline column.
- If a match is found: return the verdict immediately with 100% confidence and method = "database".

**Step 2 — AI Prediction (Fallback):**
- If no database match is found, load the pre-trained `model.pkl` and `vectorizer.pkl`.
- These files are loaded **once when Django starts** and kept in memory (not reloaded on every request).
- Use `model.predict_proba()` to get a prediction AND a confidence percentage.
- Return the verdict with the confidence score and method = "ml_prediction".

---

### 4. Views & Templates Update

#### [MODIFY] views.py
- Update the `analyze` view to save the new `confidence` and `method` fields to the database.
- Update the `result` view to pass confidence and method data to the template.
- Update `dashboard_data` API to include method distribution (how many were DB matches vs. AI predictions).

#### [MODIFY] result.html
- If method is "database": Show a message like "This headline was found in our fact-check database of 46,000 verified articles. Verdict: FAKE."
- If method is "ml_prediction": Show a message like "This headline was not in our database. Our AI model analyzed the text patterns and is 94% confident this is Fake News."
- Display a confidence meter/bar to visually represent the percentage.

#### [MODIFY] dashboard.html
- Add a new chart or stat showing the split between Database Matches vs. AI Predictions.

---

## Verification Plan

### Automated Verification
1. Run `python manage.py makemigrations` and `python manage.py migrate` to create the new database tables.
2. Run `python load_mega_data.py` and verify it prints the total number of articles inserted (expected: 40,000+).
3. Run `python train_mega_model.py` and verify it prints the model accuracy (expected: 90%+) and creates `model.pkl` and `vectorizer.pkl`.

### Manual Browser Testing
1. Start the server with `python manage.py runserver`.
2. Paste a headline copied directly from one of the CSV files to confirm the **Database Lookup** returns an instant 100% confidence result.
3. Type a completely made-up headline to confirm the **ML Prediction** fallback returns a result with a confidence percentage.
4. Check the Dashboard to verify charts update correctly with the new data.
