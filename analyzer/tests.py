"""
tests.py — Test Suite for the Analyzer App
============================================

Run with:  python manage.py test analyzer

These run against a temporary database, so they do NOT need Fake.csv/True.csv
or a loaded KnownArticle table — each test creates the rows it needs.

Tests that exercise the ML fallback are skipped automatically when
analyzer/model.pkl and analyzer/vectorizer.pkl are absent, so a fresh clone
can still run the suite.

For a smoke test against the real loaded dataset, use test_logic.py.
"""

import unittest

from django.test import TestCase
from django.urls import reverse

from .analysis_engine import analyze_text, best_match, _ml_ready
from .models import KnownArticle, NewsAnalysis
from .text_matching import normalize_headline


# Applied to tests that need model.pkl / vectorizer.pkl to be present
requires_ml = unittest.skipUnless(
    _ml_ready, "model.pkl / vectorizer.pkl not present — run train_mega_model.py"
)

KNOWN_FAKE = "a known fake headline"
KNOWN_REAL = "a known real headline"

# The real headline that exposed the too-fussy matching: the apostrophe in
# "Year's" is the curly kind (U+2019), not the one on a keyboard.
CURLY = "Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing"


class KnownArticleMixin:
    """Gives each test a tiny stand-in for the fact-check corpus."""

    def setUp(self):
        super().setUp()
        KnownArticle.objects.create(headline=KNOWN_FAKE, label='FAKE', source='Test Corpus')
        KnownArticle.objects.create(headline=KNOWN_REAL, label='REAL', source='Test Corpus')


# ============================================================================
# TIDYING HEADLINES UP BEFORE COMPARING THEM
# ============================================================================

class NormalizeHeadlineTests(TestCase):

    def test_curly_and_straight_apostrophes_agree(self):
        self.assertEqual(normalize_headline("It’s fake"), normalize_headline("It's fake"))

    def test_case_is_ignored(self):
        self.assertEqual(normalize_headline("SHOUTING HEADLINE"), "shouting headline")

    def test_punctuation_is_ignored(self):
        self.assertEqual(
            normalize_headline("Breaking: aliens land!!!"),
            normalize_headline("Breaking - aliens land"),
        )

    def test_repeated_whitespace_collapses(self):
        self.assertEqual(normalize_headline("  too   many \n spaces  "), "too many spaces")

    def test_dashes_of_all_kinds_agree(self):
        self.assertEqual(
            normalize_headline("Trump — Disturbing"),   # em dash
            normalize_headline("Trump - Disturbing"),        # hyphen
        )

    def test_accents_are_folded(self):
        self.assertEqual(normalize_headline("café scandal"), "cafe scandal")

    def test_apostrophes_are_deleted_not_split(self):
        """"Trump's" should become "trumps", so it also matches "Trumps"."""
        self.assertEqual(normalize_headline("Trump's win"), "trumps win")

    def test_empty_and_none_are_safe(self):
        self.assertEqual(normalize_headline(''), '')
        self.assertEqual(normalize_headline(None), '')

    def test_punctuation_only_headline_yields_nothing(self):
        self.assertEqual(normalize_headline('!!! ---  ???'), '')


# ============================================================================
# FINDING A KNOWN ARTICLE (Step 1)
# ============================================================================

class BestMatchTests(TestCase):

    def setUp(self):
        self.article = KnownArticle.objects.create(
            headline=CURLY, label='FAKE', source='Kaggle ISOT',
        )

    def test_exact_paste_matches(self):
        self.assertEqual(best_match(CURLY), self.article)

    def test_straight_apostrophe_still_matches(self):
        """Regression: this used to miss completely and fall through to the model."""
        typed = CURLY.replace('’', "'")
        self.assertEqual(best_match(typed), self.article)

    def test_trailing_full_stop_still_matches(self):
        self.assertEqual(best_match(CURLY + '.'), self.article)

    def test_different_punctuation_still_matches(self):
        self.assertEqual(best_match(CURLY.replace(';', ':')), self.article)

    def test_shouting_and_double_spaces_still_match(self):
        self.assertEqual(best_match('  ' + CURLY.upper().replace(' ', '  ')), self.article)

    def test_genuinely_different_headline_does_not_match(self):
        self.assertIsNone(best_match('Something else entirely happened today'))

    def test_blank_input_does_not_match(self):
        """A punctuation-only input must not match rows with a blank key."""
        KnownArticle.objects.create(headline='...', label='FAKE', source='Junk')
        self.assertIsNone(best_match('!!!'))
        self.assertIsNone(best_match(''))

    def test_fact_checked_row_beats_publisher_guess(self):
        """When two sources disagree, the fact-checker wins — not whichever came first."""
        KnownArticle.objects.create(
            headline=CURLY, label='REAL', source='PolitiFact',
            label_basis=KnownArticle.BASIS_FACT_CHECK,
        )

        winner = best_match(CURLY)

        self.assertEqual(winner.source, 'PolitiFact')
        self.assertEqual(winner.label, 'REAL')

    def test_tie_break_is_stable(self):
        """Same input, same answer, every time — no luck involved."""
        KnownArticle.objects.create(headline=CURLY, label='REAL', source='Another Corpus')

        picks = {best_match(CURLY).id for _ in range(5)}

        self.assertEqual(len(picks), 1)


class LabelBasisTests(TestCase):

    def test_publisher_basis_admits_nobody_checked(self):
        article = KnownArticle.objects.create(
            headline='x', label='REAL', source='Kaggle ISOT',
        )

        self.assertFalse(article.is_fact_checked)
        self.assertIn('where it was published', article.describe_basis())
        self.assertIn('nobody checked', article.describe_basis())

    def test_fact_check_basis_says_who_checked_it(self):
        article = KnownArticle.objects.create(
            headline='y', label='FAKE', source='PolitiFact',
            label_basis=KnownArticle.BASIS_FACT_CHECK,
        )

        self.assertTrue(article.is_fact_checked)
        self.assertIn('PolitiFact fact-checked this claim', article.describe_basis())

    def test_headline_key_is_filled_in_automatically(self):
        """No caller should be able to create a row that lookups cannot find."""
        article = KnownArticle.objects.create(
            headline="It’s Fake!", label='FAKE', source='Test',
        )

        self.assertEqual(article.headline_key, 'its fake')


# ============================================================================
# THE ANALYSIS ENGINE (Step 1: database, Step 2: ML)
# ============================================================================

class AnalysisEngineTests(KnownArticleMixin, TestCase):

    def test_database_match_returns_full_confidence(self):
        result = analyze_text(KNOWN_FAKE)

        self.assertEqual(result['method'], 'database')
        self.assertEqual(result['label'], 'Likely Fake')
        self.assertEqual(result['confidence'], 100.0)
        self.assertEqual(result['score'], 100)
        self.assertIn('Test Corpus', result['explanation'])

    def test_database_match_real_article(self):
        result = analyze_text(KNOWN_REAL)

        self.assertEqual(result['method'], 'database')
        self.assertEqual(result['label'], 'Likely Real')
        self.assertEqual(result['score'], 0)

    def test_database_lookup_normalizes_case_and_whitespace(self):
        """Headlines are stored lowercased, so lookup must normalize too."""
        result = analyze_text("   A KNOWN Fake HEADLINE   ")

        self.assertEqual(result['method'], 'database')
        self.assertEqual(result['label'], 'Likely Fake')

    def test_result_carries_no_dead_keyword_keys(self):
        """The old rule-based scorer's matched_* keys should be gone."""
        result = analyze_text(KNOWN_FAKE)

        for dead_key in ('matched_fake', 'matched_reliable', 'matched_emotional'):
            self.assertNotIn(dead_key, result)

    def test_engine_never_returns_suspicious(self):
        """'Suspicious' belonged to the removed keyword scorer."""
        for text in (KNOWN_FAKE, KNOWN_REAL, "an entirely unseen headline"):
            self.assertNotEqual(analyze_text(text)['label'], 'Suspicious')

    @requires_ml
    def test_unknown_headline_falls_back_to_a_model(self):
        """
        An unknown headline goes to Step 2. Which model answers depends on
        whether the reading model could be loaded, so both are accepted.
        """
        result = analyze_text(
            "Scientists confirm pizza cures cancer",
            "A study from an unknown university claims that pizza is the ultimate "
            "cure for every known disease. The researchers said they had tested "
            "the theory on a group of volunteers over several months and found "
            "that all of them reported feeling much better afterwards, according "
            "to a statement released on Tuesday by the institute.",
        )

        self.assertIn(result['method'], ('reading_model', 'ml_prediction'))
        self.assertIn(result['label'], ('Likely Fake', 'Likely Real'))
        self.assertTrue(result['explanation'])
        self.assertGreaterEqual(result['confidence'], 50.0)
        self.assertLessEqual(result['confidence'], 100.0)

    @requires_ml
    def test_reading_model_always_answers(self):
        """
        Step 2 must not go silent. An earlier version returned 'Not Sure' when
        the model was not confident enough; that was reverted because a tool
        that mostly declines to answer is not usable.
        """
        from analyzer.analysis_engine import _load_reading_model

        if _load_reading_model() is None:
            self.skipTest('reading model not available')

        # Long enough for the model to actually read. A separate test covers
        # what happens below that length, which is a different question: this one
        # is about never going silent on text it CAN read.
        for text in (
            "An entirely unseen headline about nothing in particular. The report "
            "said that officials had met on Tuesday to discuss the matter and "
            "would publish their findings at some point in the coming weeks, "
            "according to two people with knowledge of the discussions.",

            "Central bank holds benchmark interest rate steady at 4.5 percent. "
            "Policymakers voted seven to two to keep the current range in place, "
            "citing persistent inflation and a labour market that has begun to "
            "cool over the past several months.",

            "SHOCKING one weird trick doctors HATE cures everything instantly. "
            "The mainstream media will never tell you about this because they are "
            "paid by the same people who profit from keeping you sick, and that "
            "is why you need to share this with everyone you know right now.",
        ):
            result = analyze_text(text)
            self.assertIn(result['label'], ('Likely Fake', 'Likely Real'))
            self.assertIsNotNone(result['confidence'])

    @requires_ml
    def test_reading_model_quotes_measured_reliability_not_self_confidence(self):
        """
        The percentage shown must be the measured reliability of that verdict
        from cutoff.json — never the model's own softmax certainty, which is
        routinely 99% while being wrong. Guards the core promise of the design.
        """
        from analyzer.analysis_engine import _load_reading_model

        state = _load_reading_model()
        if state is None:
            self.skipTest('reading model not available')

        expected = {
            'Likely Fake': state['verdict']['fake_precision'],
            'Likely Real': state['verdict']['real_precision'],
        }

        for text in ("an entirely unseen headline about nothing in particular",
                     "SHOCKING one weird trick doctors HATE cures everything instantly"):
            result = analyze_text(text)
            if result['method'] != 'reading_model':
                continue
            self.assertEqual(result['confidence'], expected[result['label']])


# ============================================================================
# THE ANALYZE VIEW (this is where the form POST used to crash)
# ============================================================================

class AnalyzeViewTests(KnownArticleMixin, TestCase):

    def test_get_renders_the_form(self):
        response = self.client.get(reverse('analyze'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'content-input')

    def test_post_creates_analysis_and_redirects(self):
        """Regression: this raised KeyError('headline') and never saved anything."""
        response = self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})

        analysis = NewsAnalysis.objects.get()
        self.assertRedirects(response, reverse('result', args=[analysis.id]))
        self.assertEqual(analysis.result_label, 'Likely Fake')
        self.assertEqual(analysis.method, 'database')
        self.assertEqual(analysis.confidence, 100.0)

    def test_post_persists_the_explanation(self):
        """Regression: the explanation was computed but thrown away."""
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})

        analysis = NewsAnalysis.objects.get()
        self.assertTrue(analysis.explanation)
        self.assertIn('Test Corpus', analysis.explanation)

    def test_post_splits_first_line_as_headline(self):
        self.client.post(reverse('analyze'), {
            'news_content': 'The headline goes here\nAnd the article body follows it.',
        })

        analysis = NewsAnalysis.objects.get()
        self.assertEqual(analysis.headline, 'The headline goes here')
        self.assertEqual(analysis.article_text, 'And the article body follows it.')

    def test_post_single_line_is_headline_only(self):
        self.client.post(reverse('analyze'), {'news_content': KNOWN_REAL})

        analysis = NewsAnalysis.objects.get()
        self.assertEqual(analysis.headline, KNOWN_REAL)
        self.assertEqual(analysis.article_text, '')

    def test_long_single_line_paste_is_not_lost(self):
        """Headline column caps at 500 chars — the full text must survive."""
        # Real English, because the form now rejects text that is not. The old
        # fixture was 'lorem ipsum' repeated, which is Latin and gets turned away
        # before it ever reaches the truncation this test is about.
        long_text = ('The central bank left its benchmark interest rate unchanged '
                     'on Wednesday. ') * 14  # ~1000 chars, no newline
        self.client.post(reverse('analyze'), {'news_content': long_text})

        analysis = NewsAnalysis.objects.get()
        self.assertEqual(len(analysis.headline), 500)
        self.assertTrue(analysis.headline.endswith('...'))
        self.assertEqual(analysis.article_text, long_text.strip())

    def test_empty_post_redisplays_form_without_saving(self):
        response = self.client.post(reverse('analyze'), {'news_content': ''})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(NewsAnalysis.objects.count(), 0)


# ============================================================================
# THE RESULT VIEW
# ============================================================================

class ResultViewTests(KnownArticleMixin, TestCase):

    def test_result_page_shows_the_explanation(self):
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})
        analysis = NewsAnalysis.objects.get()

        response = self.client.get(reverse('result', args=[analysis.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Test Corpus')
        self.assertContains(response, 'Likely Fake')

    def test_result_page_does_not_leak_raw_json(self):
        """Regression: the explanation panel used to print the keyword JSON blob."""
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})
        analysis = NewsAnalysis.objects.get()

        body = self.client.get(reverse('result', args=[analysis.id])).content.decode()

        self.assertNotIn('{"fake"', body)
        self.assertNotIn('{&quot;fake&quot;', body)

    def test_unknown_analysis_id_returns_404(self):
        response = self.client.get(reverse('result', args=[9999]))

        self.assertEqual(response.status_code, 404)


# ============================================================================
# SHOWING THE READER THE ARTICLE WE MATCHED
# ============================================================================

class MatchedArticleEvidenceTests(TestCase):

    def setUp(self):
        KnownArticle.objects.create(
            headline='A Known Fake Headline',
            article_text='The opening paragraph of the matched article goes here.',
            label='FAKE',
            source='Kaggle ISOT',
        )

    def _analyse(self, text):
        self.client.post(reverse('analyze'), {'news_content': text})
        return NewsAnalysis.objects.get()

    def test_match_is_copied_onto_the_analysis(self):
        analysis = self._analyse('a known fake headline')

        self.assertEqual(analysis.matched_headline, 'A Known Fake Headline')
        self.assertIn('opening paragraph', analysis.matched_extract)

    def test_match_shows_the_published_capitalisation(self):
        """We store the headline as published, not the tidied lookup version."""
        analysis = self._analyse('A KNOWN FAKE HEADLINE')

        self.assertEqual(analysis.matched_headline, 'A Known Fake Headline')

    def test_matched_article_is_rendered_on_the_result_page(self):
        analysis = self._analyse('a known fake headline')

        response = self.client.get(reverse('result', args=[analysis.id]))

        self.assertContains(response, 'The Article We Matched')
        self.assertContains(response, 'opening paragraph')

    def test_explanation_says_why_the_label_is_believed(self):
        analysis = self._analyse('a known fake headline')

        self.assertIn('where it was published', analysis.explanation)
        self.assertIn('nobody checked', analysis.explanation)

    def test_nothing_is_shown_when_no_article_matched(self):
        analysis = self._analyse('an utterly unseen headline about nothing at all')

        self.assertEqual(analysis.matched_headline, '')
        response = self.client.get(reverse('result', args=[analysis.id]))
        self.assertNotContains(response, 'The Article We Matched')


# ============================================================================
# THE DASHBOARD AND ITS CHART API
# ============================================================================

class DashboardTests(KnownArticleMixin, TestCase):

    def test_dashboard_renders_empty_state(self):
        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'No Analyses Yet')

    def test_dashboard_counts_results(self):
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})
        self.client.post(reverse('analyze'), {'news_content': KNOWN_REAL})

        response = self.client.get(reverse('dashboard'))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['total'], 2)
        self.assertEqual(response.context['fake_count'], 1)
        self.assertEqual(response.context['real_count'], 1)

    def test_dashboard_has_no_suspicious_category(self):
        """'Suspicious' can never occur, so it must not be reported."""
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})

        response = self.client.get(reverse('dashboard'))

        self.assertNotIn('suspicious_count', response.context)
        self.assertNotContains(response, 'Suspicious')

    def test_chart_api_returns_pie_and_method_data(self):
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})

        data = self.client.get(reverse('dashboard_data')).json()

        # 'Not Sure' is a first-class outcome, not an absence of one — Step 2
        # returns it for most unknown articles, so it must be counted or the
        # chart will not add up to the total.
        self.assertEqual(data['pie_data']['labels'],
                         ['Likely Real', 'Likely Fake', 'Not Sure'])
        self.assertEqual(data['pie_data']['counts'], [0, 1, 0])
        self.assertEqual(data['method_data']['labels'],
                         ['Database Match', 'Reading Model', 'Word-Counter'])
        self.assertEqual(data['method_data']['counts'], [1, 0, 0])

    def test_chart_api_drops_the_dead_keyword_chart(self):
        """bar_data counted keywords the engine no longer produces."""
        data = self.client.get(reverse('dashboard_data')).json()

        self.assertNotIn('bar_data', data)
        self.assertNotIn('Suspicious', str(data))


# ============================================================================
# THE LOADER: TOPS UP, NEVER WIPES
# ============================================================================

def _incoming(rows):
    """
    Build the little table the loader passes around, so these tests don't need
    a CSV on disk. Columns match read_dataset()'s output.
    """
    import pandas as pd

    return pd.DataFrame(
        [
            {
                'headline': headline,
                'body': body,
                'label': label,
                'source': source,
                'basis': basis,
                'key': normalize_headline(headline),
            }
            for headline, body, label, source, basis in rows
        ]
    )


SOURCE = KnownArticle.BASIS_SOURCE
FACT_CHECK = KnownArticle.BASIS_FACT_CHECK


class LoaderTopUpTests(TestCase):
    """
    The loader used to delete every saved article before refilling, which meant
    adding one new source re-imported all of them and destroyed anything fixed
    by hand. These lock in the top-up behaviour that replaced it.
    """

    def setUp(self):
        from load_mega_data import add_new_articles, load_existing
        self.add_new_articles = add_new_articles
        self.load_existing = load_existing

        KnownArticle.objects.create(
            headline='A Headline We Already Have', label='FAKE', source='Old Corpus',
        )

    def _load(self, rows):
        return self.add_new_articles(
            _incoming(rows), self.load_existing(), 'test', verbose=False
        )

    def test_brand_new_articles_are_added(self):
        counts = self._load([('Something Genuinely New', 'body', 'REAL', 'BBC', SOURCE)])

        self.assertEqual(counts['added'], 1)
        self.assertEqual(KnownArticle.objects.count(), 2)

    def test_articles_we_already_have_are_skipped(self):
        counts = self._load([('A Headline We Already Have', 'body', 'FAKE', 'BBC', SOURCE)])

        self.assertEqual(counts['already_had'], 1)
        self.assertEqual(counts['added'], 0)
        self.assertEqual(KnownArticle.objects.count(), 1)

    def test_punctuation_differences_count_as_already_had(self):
        """The whole reason top-up is possible: near-spellings are recognised."""
        counts = self._load([("a headline we ALREADY have!!!", 'b', 'FAKE', 'BBC', SOURCE)])

        self.assertEqual(counts['already_had'], 1)
        self.assertEqual(KnownArticle.objects.count(), 1)

    def test_running_twice_creates_no_duplicates(self):
        rows = [('Something Genuinely New', 'body', 'REAL', 'BBC', SOURCE)]

        self._load(rows)
        self._load(rows)

        self.assertEqual(KnownArticle.objects.filter(source='BBC').count(), 1)

    def test_duplicates_inside_one_file_are_caught(self):
        counts = self._load([
            ('Repeated In The Same File', 'body', 'REAL', 'BBC', SOURCE),
            ('repeated in the same file.', 'body', 'REAL', 'BBC', SOURCE),
        ])

        self.assertEqual(counts['added'], 1)
        self.assertEqual(counts['already_had'], 1)

    def test_nothing_existing_is_deleted(self):
        self._load([('Something Genuinely New', 'body', 'REAL', 'BBC', SOURCE)])

        self.assertTrue(KnownArticle.objects.filter(source='Old Corpus').exists())

    def test_a_real_fact_check_is_added_over_a_publisher_guess(self):
        counts = self._load([
            ('A Headline We Already Have', 'body', 'REAL', 'PolitiFact', FACT_CHECK),
        ])

        self.assertEqual(counts['better_evidence'], 1)
        self.assertEqual(counts['already_had'], 0)
        # and the lookup now prefers it, flipping the verdict
        self.assertEqual(best_match('A Headline We Already Have').source, 'PolitiFact')

    def test_a_publisher_guess_does_not_override_a_fact_check(self):
        KnownArticle.objects.create(
            headline='Already Fact Checked', label='REAL',
            source='PolitiFact', label_basis=FACT_CHECK,
        )

        counts = self._load([('Already Fact Checked', 'b', 'FAKE', 'Blog', SOURCE)])

        self.assertEqual(counts['already_had'], 1)
        self.assertEqual(best_match('Already Fact Checked').source, 'PolitiFact')

    def test_unusable_headlines_are_reported_not_saved(self):
        counts = self._load([('!!! ---', 'body', 'FAKE', 'BBC', SOURCE)])

        self.assertEqual(counts['unusable'], 1)
        self.assertEqual(KnownArticle.objects.count(), 1)

    def test_long_bodies_are_shortened_to_an_extract(self):
        from load_mega_data import EXTRACT_CHARS

        self._load([('New One', 'x' * (EXTRACT_CHARS + 500), 'REAL', 'BBC', SOURCE)])

        saved = KnownArticle.objects.get(source='BBC')
        self.assertEqual(len(saved.article_text), EXTRACT_CHARS + 3)
        self.assertTrue(saved.article_text.endswith('...'))

    def test_bulk_inserted_rows_still_get_a_lookup_key(self):
        """bulk_create skips save(), so the loader must set the key itself."""
        self._load([('New One', 'body', 'REAL', 'BBC', SOURCE)])

        saved = KnownArticle.objects.get(source='BBC')
        self.assertEqual(saved.headline_key, 'new one')
        self.assertEqual(best_match('NEW ONE!'), saved)


# ============================================================================
# THE WEB CHECK — WHEN IT RUNS, AND WHAT THE PAGE SAYS
# ============================================================================
#
# Two separate risks are covered here.
#
# The first is spending money. The search is only worth an API call inside the
# band where the reading model is unreliable, which tune_cutoff.py measures and
# records in cutoff.json. A regression that fired it on every article would not
# break any output — it would quietly burn the daily free allowance and slow
# every page, which is exactly the kind of fault nobody notices until it bites.
#
# The second is misleading the reader. 'nothing' and 'unavailable' look similar
# in code and mean opposite things: one is evidence against the article, the
# other is no evidence at all. The page must never dress a failed check up as a
# finding, so each state is asserted separately.

FOUND_REAL = {
    'status': 'found', 'verdict': 'REAL', 'confidence': 'high',
    'summary': 'Several outlets report this happened.',
    'reporting': ['Reuters', 'AP', 'BBC'], 'debunking': [],
    'factcheck_rating': None,
    'sources': [{'title': 'bbc.co.uk', 'url': 'https://example.com/story'}],
    'queries': ['probe'], 'error': None, 'elapsed': 5.0,
}
FOUND_FAKE = dict(FOUND_REAL, verdict='FAKE', reporting=[],
                  debunking=['Snopes', 'PolitiFact'],
                  factcheck_rating='False',
                  summary='Fact-checkers rated this false.')
FOUND_NOTHING = dict(FOUND_REAL, status='nothing', verdict='UNCLEAR',
                     reporting=[], debunking=[], summary='')
UNAVAILABLE = dict(FOUND_REAL, status='unavailable', verdict=None,
                   reporting=[], debunking=[], sources=[], summary='',
                   error='daily free search allowance used up')


class ShouldCheckTests(TestCase):
    """Only spend an API call where the model has been measured unreliable."""

    def setUp(self):
        from . import web_check
        self.web_check = web_check
        self._key = web_check._read_key
        web_check._read_key = lambda: 'test-key'
        self._band = web_check.unsure_band
        web_check.unsure_band = lambda: (0.01, 0.99)

    def tearDown(self):
        self.web_check._read_key = self._key
        self.web_check.unsure_band = self._band

    def test_fires_inside_the_unsure_band(self):
        for score in (0.02, 0.4, 0.5, 0.97):
            self.assertTrue(self.web_check.should_check(score), score)

    def test_stays_quiet_where_the_model_is_confident(self):
        for score in (0.0, 0.001, 0.005, 0.995, 0.9999, 1.0):
            self.assertFalse(self.web_check.should_check(score), score)

    def test_no_key_means_no_call(self):
        self.web_check._read_key = lambda: None
        self.assertFalse(self.web_check.should_check(0.4))

    def test_no_measured_band_means_no_call(self):
        """A missing band is not licence to guess when a search is needed."""
        self.web_check.unsure_band = lambda: None
        self.assertFalse(self.web_check.should_check(0.4))

    def test_missing_score_means_no_call(self):
        self.assertFalse(self.web_check.should_check(None))


class WebCheckFailureTests(TestCase):
    """Every failure returns a renderable result. None of them raise."""

    def setUp(self):
        from . import web_check
        self.web_check = web_check
        self._key = web_check._read_key

    def tearDown(self):
        self.web_check._read_key = self._key

    def test_missing_key_is_unavailable(self):
        self.web_check._read_key = lambda: None
        result = self.web_check.check_online('Some headline')
        self.assertEqual(result['status'], 'unavailable')
        self.assertIsNone(result['verdict'])
        self.assertIn('GEMINI_API_KEY', result['error'])

    def test_empty_headline_is_unavailable(self):
        self.web_check._read_key = lambda: 'test-key'
        result = self.web_check.check_online('')
        self.assertEqual(result['status'], 'unavailable')

    def test_network_failure_is_unavailable_not_an_exception(self):
        self.web_check._read_key = lambda: 'test-key'
        original = self.web_check.urllib.request.urlopen

        def explode(*args, **kwargs):
            raise OSError('network is down')

        self.web_check.urllib.request.urlopen = explode
        try:
            result = self.web_check.check_online('Some headline')
        finally:
            self.web_check.urllib.request.urlopen = original

        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['error'], 'OSError')

    def test_garbled_reply_keeps_the_sources_it_did_get(self):
        """
        A reply that is not the JSON we asked for must not throw the grounding
        sources away. Those are the pages actually consulted, so they are the
        part of the answer that cannot have been invented.
        """
        reply = {'candidates': [{
            'content': {'parts': [{'text': 'Sorry, I cannot comply.'}]},
            'groundingMetadata': {
                'groundingChunks': [
                    {'web': {'uri': 'https://example.com/a',
                             'title': 'example.com'}}],
                'webSearchQueries': ['something'],
            },
        }]}
        result = self.web_check._read_reply(reply, 1.0)

        self.assertEqual(len(result['sources']), 1)
        self.assertEqual(result['verdict'], 'UNCLEAR')
        self.assertIn('JSON', result['error'])


class WebCheckDisplayTests(TestCase):
    """What the reader is actually told, in each of the five states."""

    def _page(self, label, web):
        analysis = NewsAnalysis.objects.create(
            headline='Display probe', article_text='body', score=50,
            result_label=label, explanation='probe', confidence=86.9,
            method='reading_model', web_check=web)
        return self.client.get(reverse('result', args=[analysis.id]))

    def test_agreement_shows_the_evidence(self):
        response = self._page('Likely Fake', FOUND_FAKE)
        self.assertContains(response, 'What We Found Online')
        self.assertContains(response, 'Snopes')
        self.assertNotContains(response, 'These two disagree')

    def test_model_says_fake_but_web_says_real_is_flagged(self):
        response = self._page('Likely Fake', FOUND_REAL)
        self.assertContains(response, 'These two disagree')
        self.assertContains(response, 'Reuters')

    def test_model_says_real_but_web_says_fake_is_flagged(self):
        response = self._page('Likely Real', FOUND_FAKE)
        self.assertContains(response, 'These two disagree')

    def test_found_nothing_reads_as_suspicion_not_proof(self):
        response = self._page('Likely Fake', FOUND_NOTHING)
        self.assertContains(response, 'No coverage found')
        self.assertContains(response, 'not proof')

    def test_a_failed_check_is_not_presented_as_a_finding(self):
        """
        The important one. 'We could not check' must never read like 'we checked
        and found nothing', because the second is evidence against the article
        and the first is evidence about nothing at all.
        """
        response = self._page('Likely Fake', UNAVAILABLE)
        self.assertContains(response, 'Could not check')
        self.assertNotContains(response, 'No coverage found')
        self.assertNotContains(response, 'These two disagree')

    def test_no_section_at_all_when_the_search_never_ran(self):
        response = self._page('Likely Fake', None)
        self.assertNotContains(response, 'What We Found Online')


# ============================================================================
# WHAT WE REFUSE TO JUDGE
# ============================================================================
#
# Measured on 2 Sep 2026, the reading model answers everything it is shown,
# including things it cannot read:
#
#   the same story in Urdu   Likely Fake   0.9866
#   Spanish                  Likely Fake   0.3498
#   random symbols           Likely Fake   0.9950
#   keyboard mash            Likely Fake   0.9998
#   the single word "news"   Likely Fake   0.9980
#
# Every one of those would have been printed next to "right 86.9% of the time",
# a figure measured on English news articles. Refusing is the honest answer, and
# these tests exist so a later change cannot quietly start guessing again.

class InputCheckTests(TestCase):
    """The rules themselves, without the form or the model in the way."""

    def setUp(self):
        from .input_check import check_input, check_long_enough
        self.check_input = check_input
        self.check_long_enough = check_long_enough

    def test_real_english_article_is_accepted(self):
        text = ("Central bank holds interest rates steady amid inflation concerns. "
                "The central bank left its benchmark interest rate unchanged on "
                "Wednesday, citing persistent inflation and a cooling labour market. "
                "Policymakers voted seven to two to maintain the current range.")
        self.assertIsNone(self.check_input(text))
        self.assertIsNone(self.check_long_enough(text))

    def test_non_latin_script_is_refused(self):
        urdu = ("مرکزی بینک نے شرح سود برقرار رکھی۔ مرکزی بینک نے بدھ کو اپنی بنیادی "
                "شرح سود میں کوئی تبدیلی نہیں کی، مسلسل مہنگائی اور سست ہوتی لیبر "
                "مارکیٹ کا حوالہ دیتے ہوئے۔ پالیسی سازوں نے سات کے مقابلے دو ووٹوں "
                "سے موجودہ حد برقرار رکھنے کا فیصلہ کیا جو توقعات کے مطابق تھا۔")
        problem = self.check_input(urdu)
        self.assertIsNotNone(problem)
        self.assertEqual(problem.code, 'not_english')

    def test_latin_script_but_not_english_is_refused(self):
        """Spanish passes the alphabet check, so the word check has to catch it."""
        spanish = ("El banco central mantuvo sin cambios su tasa de referencia el "
                   "miercoles citando la inflacion persistente y un mercado laboral "
                   "en proceso de enfriamiento. Los responsables de politica "
                   "monetaria votaron siete a dos para mantener el rango actual.")
        problem = self.check_input(spanish)
        self.assertIsNotNone(problem)
        self.assertEqual(problem.code, 'not_english_prose')

    def test_keyboard_mash_is_refused(self):
        mash = ("asdkjfh alskdjfh alskdjfh qwoieuryt zxcmvnb asldkfj qpwoeiru "
                "zmxncbv alsdkjfh qweruiop mnbvcxz lkjhgfds poiuytre wqasdfgh "
                "zxcvbnml plokijuh ygtfrdes wsxedcrf vgybhunj mikolp qazwsxed "
                "rfvtgbyh njmikolp qwertyui asdfghjk zxcvbnmq")
        problem = self.check_input(mash)
        self.assertIsNotNone(problem)
        self.assertEqual(problem.code, 'not_english_prose')

    def test_over_the_word_limit_is_refused(self):
        from .input_check import MAX_WORDS
        problem = self.check_input('the quick brown fox jumps over ' * 100)
        self.assertIsNotNone(problem)
        self.assertEqual(problem.code, 'too_long')
        self.assertIn(str(MAX_WORDS), problem.message)

    def test_a_bare_headline_is_still_accepted_by_the_form(self):
        """
        The important one. A four-word headline must NOT be turned away, because
        Step 1 looks headlines up in the fact-check table and answers from the
        record — the most reliable answer this app gives. Only the model needs a
        minimum, and that is checked later.
        """
        self.assertIsNone(self.check_input('Trump wins Iowa caucus'))
        self.assertIsNotNone(self.check_long_enough('Trump wins Iowa caucus'))


class InputCheckFormTests(TestCase):
    """The same rules as the user meets them."""

    def _post(self, text):
        return self.client.post(reverse('analyze'), {'news_content': text})

    def test_urdu_is_rejected_with_an_explanation(self):
        response = self._post("مرکزی بینک نے شرح سود برقرار رکھی۔ مرکزی بینک نے بدھ "
                              "کو اپنی بنیادی شرح سود میں کوئی تبدیلی نہیں کی۔")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'only read English')
        self.assertEqual(NewsAnalysis.objects.count(), 0)

    def test_over_the_limit_is_rejected_with_an_explanation(self):
        response = self._post('the quick brown fox jumps over ' * 100)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'the limit is')
        self.assertEqual(NewsAnalysis.objects.count(), 0)

    def test_nothing_is_saved_when_input_is_refused(self):
        """A refusal must not leave a row behind claiming a verdict."""
        self._post('😀🎉🚀💀🔥👍🌟⚡ ' * 8)
        self.assertEqual(NewsAnalysis.objects.count(), 0)


class TooShortForModelTests(KnownArticleMixin, TestCase):
    """Short unknown text gets 'Not Sure', never a confident guess."""

    def test_short_unknown_text_is_not_guessed_at(self):
        result = analyze_text('Some unseen headline nobody has indexed')

        self.assertEqual(result['label'], 'Not Sure')
        self.assertEqual(result['method'], 'too_short')
        self.assertIsNone(result['confidence'])
        self.assertIn('25 words', result['explanation'])

    def test_a_known_short_headline_still_answers_from_the_database(self):
        """
        The minimum must not block Step 1. This headline is four words long and
        in the fact-check table, so it gets a definitive answer with no model
        involved — that path is more reliable than anything the model does.
        """
        result = analyze_text(KNOWN_FAKE)

        self.assertEqual(result['method'], 'database')
        self.assertEqual(result['label'], 'Likely Fake')
