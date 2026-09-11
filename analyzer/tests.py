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

import re
import unittest
from unittest import mock

from django.test import TestCase
from django.urls import reverse

from . import analysis_engine
from . import views
from .analysis_engine import (analyze_text, best_match, best_match_by_body,
                              _ml_ready)
from .forms import NewsInputForm
from .input_check import MIN_WORDS
from .models import KnownArticle, NewsAnalysis
from .text_matching import BODY_KEY_WORDS, body_key, normalize_headline


# Applied to tests that need model.pkl / vectorizer.pkl to be present
requires_ml = unittest.skipUnless(
    _ml_ready, "model.pkl / vectorizer.pkl not present — run train_mega_model.py"
)

KNOWN_FAKE = "a known fake headline"
KNOWN_REAL = "a known real headline"

# The real headline that exposed the too-fussy matching: the apostrophe in
# "Year's" is the curly kind (U+2019), not the one on a keyboard.
CURLY = "Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing"

# The case that made body matching necessary. During testing the user pasted a
# headline AND its body, having trimmed the trailing "(VIDEO)" off the title.
# The headline lookup missed, the model was asked instead, and 100% certain
# became an 86.9% guess — while a perfect copy of the body sat in the database,
# unexamined.
VIDEO_HEADLINE = (
    "Sheriff David Clarke Becomes An Internet Joke For Threatening "
    "To Poke People In The Eye (VIDEO)"
)
VIDEO_BODY = (
    "Milwaukee Sheriff David Clarke is a Fox News regular who has a habit of "
    "saying outrageous things in order to stay relevant, and this week was no "
    "different. During an appearance on the network he threatened to poke "
    "people in the eye, which the internet immediately turned into a joke at "
    "his expense."
)


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


# ============================================================================
# MATCHING ON THE ARTICLE INSTEAD OF THE HEADLINE
# ============================================================================

class BodyKeyTests(TestCase):
    """
    The article's own lookup key: its opening words, tidied the same way a
    headline is. See text_matching.body_key.
    """

    def test_key_is_the_opening_words_tidied(self):
        self.assertEqual(
            body_key("WASHINGTON (Reuters) - The head of a faction", words=4),
            'washington reuters the head',
        )

    def test_too_short_to_key_gives_a_blank(self):
        self.assertEqual(body_key('only four words here'), '')

    def test_key_stops_at_the_configured_word_count(self):
        self.assertEqual(len(body_key(VIDEO_BODY).split()), BODY_KEY_WORDS)

    def test_a_longer_paste_gives_the_same_key_as_the_shorter_stored_copy(self):
        """
        The whole reason the key is built from the OPENING words. We store about
        100 words of each article; a user pastes the entire thing. Both have to
        arrive at the same key or the lookup is pointless.
        """
        pasted = VIDEO_BODY + ' ' + ('Further paragraphs continue here. ' * 40)

        self.assertEqual(body_key(pasted), body_key(VIDEO_BODY))

    def test_curly_quotes_and_punctuation_do_not_change_the_key(self):
        messy = VIDEO_BODY.replace("'", '’').replace('-', '—').upper()

        self.assertEqual(body_key(messy), body_key(VIDEO_BODY))


class BodyMatchTests(TestCase):
    """best_match_by_body — the second door into the corpus."""

    def setUp(self):
        self.article = KnownArticle.objects.create(
            headline=VIDEO_HEADLINE, article_text=VIDEO_BODY,
            label='FAKE', source='Kaggle ISOT',
        )

    def test_save_fills_in_the_body_key(self):
        self.assertEqual(self.article.body_key, body_key(VIDEO_BODY))

    def test_exact_body_matches(self):
        self.assertEqual(best_match_by_body(VIDEO_BODY), self.article)

    def test_a_paste_longer_than_we_store_still_matches(self):
        longer = VIDEO_BODY + ' Clarke later doubled down in a second interview.'

        self.assertEqual(best_match_by_body(longer), self.article)

    def test_edits_after_the_opening_words_do_not_cost_the_match(self):
        """Only the opening is keyed, so a trimmed final paragraph is harmless."""
        opening = ' '.join(VIDEO_BODY.split()[:BODY_KEY_WORDS])

        self.assertEqual(best_match_by_body(opening + ' something else entirely'), self.article)

    def test_a_different_opening_does_not_match(self):
        self.assertIsNone(best_match_by_body('Something else entirely happened today. ' * 10))

    def test_a_body_too_short_to_key_matches_nothing(self):
        """
        A blank key must never be matchable. If it were, every article too short
        to key on would collide with every other one — and report 100%.
        """
        short = 'Too short to key on'
        KnownArticle.objects.create(
            headline='a short entry', article_text=short, label='FAKE', source='Junk',
        )

        self.assertIsNone(best_match_by_body(short))
        self.assertIsNone(best_match_by_body(''))
        self.assertIsNone(best_match_by_body(None))

    def test_a_row_with_no_stored_body_is_not_reachable_this_way(self):
        KnownArticle.objects.create(headline='bodyless row', label='REAL', source='Junk')

        self.assertIsNone(best_match_by_body('bodyless row'))

    def test_fact_checked_row_beats_publisher_guess(self):
        """Same ranking as the headline door — evidence quality decides, not luck."""
        KnownArticle.objects.create(
            headline='A different headline for the same story', article_text=VIDEO_BODY,
            label='REAL', source='PolitiFact', label_basis=KnownArticle.BASIS_FACT_CHECK,
        )

        winner = best_match_by_body(VIDEO_BODY)

        self.assertEqual(winner.source, 'PolitiFact')

    def test_changing_the_stored_body_moves_the_key_with_it(self):
        """
        Regression guard on save(). A key left pointing at text the row no longer
        holds is worse than no key: it matches the wrong article.
        """
        replacement = 'A completely different story about the city council budget ' * 4
        self.article.article_text = replacement
        self.article.save()

        self.assertIsNone(best_match_by_body(VIDEO_BODY))
        self.assertEqual(best_match_by_body(replacement), self.article)


class BodyMatchInAnalysisTests(TestCase):
    """
    The reported bug, end to end: take one word out of a headline and the app
    stopped consulting the database — even with a perfect copy of the article
    pasted underneath it.
    """

    def setUp(self):
        self.article = KnownArticle.objects.create(
            headline=VIDEO_HEADLINE, article_text=VIDEO_BODY,
            label='FAKE', source='Kaggle ISOT',
        )
        self.trimmed = VIDEO_HEADLINE.replace(' (VIDEO)', '')

    def test_exact_headline_and_body_answers_from_the_database(self):
        result = analyze_text(VIDEO_HEADLINE, VIDEO_BODY)

        self.assertEqual(result['method'], 'database')
        self.assertEqual(result['matched_on'], 'headline')

    def test_headline_missing_a_word_still_answers_from_the_database(self):
        """This is the exact case that used to fall through to the model."""
        result = analyze_text(self.trimmed, VIDEO_BODY)

        self.assertEqual(result['method'], 'database')
        self.assertEqual(result['label'], 'Likely Fake')
        self.assertEqual(result['confidence'], 100.0)
        self.assertEqual(result['score'], 100)

    def test_a_body_match_admits_the_headline_did_not_match(self):
        """
        The user's headline is not ours, and saying so matters: they may be
        looking at a differently titled copy of the story.
        """
        result = analyze_text(self.trimmed, VIDEO_BODY)

        self.assertEqual(result['matched_on'], 'body')
        self.assertIn('not the one we have on file', result['explanation'])

    def test_a_headline_match_does_not_claim_the_headline_differed(self):
        result = analyze_text(VIDEO_HEADLINE, VIDEO_BODY)

        self.assertNotIn('not the one we have on file', result['explanation'])

    def test_a_body_match_still_offers_the_row_as_evidence(self):
        """
        Doubly important on this path: the headline shown on the result page is
        ours, not the one the user pasted, so they need to see it.
        """
        result = analyze_text(self.trimmed, VIDEO_BODY)

        self.assertEqual(result['matched_article'], self.article)

    def test_the_headline_decides_when_both_doors_could_answer(self):
        """
        A deliberately typed headline is better evidence of what the user means
        than the first twenty words of what they pasted below it.
        """
        other = KnownArticle.objects.create(
            headline='An unrelated headline about the same body text',
            article_text=VIDEO_BODY, label='REAL', source='Kaggle ISOT',
        )

        result = analyze_text(other.headline, VIDEO_BODY)

        self.assertEqual(result['matched_article'], other)
        self.assertEqual(result['label'], 'Likely Real')

    def test_a_body_pasted_on_its_own_over_several_lines_matches(self):
        """
        views.analyze() hands the first line over as the headline. Paste a body
        with no title and line one is its opening paragraph — so the words the
        key needs end up in `headline`, not in `article_text`.
        """
        first, _, rest = VIDEO_BODY.partition('. ')

        result = analyze_text(first + '.', rest)

        self.assertEqual(result['method'], 'database')
        self.assertEqual(result['matched_on'], 'body')

    def test_a_body_pasted_on_its_own_as_one_block_matches(self):
        """No newline anywhere, so the entire article arrives as the headline."""
        result = analyze_text(VIDEO_BODY, '')

        self.assertEqual(result['method'], 'database')

    def test_a_genuinely_unknown_article_does_not_come_back_as_a_match(self):
        """
        The point of two exact keys is reach, not looseness. Something we do not
        hold must still miss both doors.
        """
        result = analyze_text(
            'A headline nobody has ever indexed anywhere',
            'The city council voted on Tuesday to approve the revised budget for '
            'the coming financial year after four hours of public comment, with '
            'two members abstaining and one absent for medical reasons.',
        )

        self.assertNotEqual(result['method'], 'database')


class LabelBasisTests(TestCase):
    """
    The two grounds must stay tellable apart. They are now separated by one
    verb rather than a sentence each — 'labelled' for a publisher's say-so,
    'fact-checked' for an investigated claim — so these assert on the verb.
    """

    def test_publisher_basis_does_not_claim_it_was_checked(self):
        article = KnownArticle.objects.create(
            headline='x', label='REAL', source='Kaggle ISOT',
        )

        self.assertFalse(article.is_fact_checked)
        self.assertIn('labelled REAL by Kaggle ISOT', article.describe_basis())
        self.assertNotIn('fact-checked', article.describe_basis())

    def test_fact_check_basis_says_who_checked_it(self):
        article = KnownArticle.objects.create(
            headline='y', label='FAKE', source='PolitiFact',
            label_basis=KnownArticle.BASIS_FACT_CHECK,
        )

        self.assertTrue(article.is_fact_checked)
        self.assertIn('fact-checked FAKE by PolitiFact', article.describe_basis())

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

    def test_post_of_a_body_with_no_headline_matches_the_database(self):
        """
        End to end through the form, because this is where the split happens: the
        paste has no title, so its opening paragraph becomes the "headline" and
        the body key has to be found anyway.
        """
        KnownArticle.objects.create(
            headline=VIDEO_HEADLINE, article_text=VIDEO_BODY,
            label='FAKE', source='Kaggle ISOT',
        )
        first, _, rest = VIDEO_BODY.partition('. ')

        self.client.post(reverse('analyze'), {'news_content': f'{first}.\n{rest}'})

        analysis = NewsAnalysis.objects.get()
        self.assertEqual(analysis.method, 'database')
        self.assertEqual(analysis.result_label, 'Likely Fake')
        # Our headline, not the paragraph they pasted — that is the evidence
        self.assertEqual(analysis.matched_headline, VIDEO_HEADLINE)

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


class HowWasThisDeterminedTests(TestCase):
    """
    The "How was this determined?" card must never render as a bare heading.

    It did exactly that for method='too_short': the template had branches for
    database, reading_model and ml_prediction, and nothing else, so a refusal
    produced a card asking the question with no answer under it. Nothing crashed
    and no test failed — the page just quietly stopped explaining itself.
    """

    # Every value analyze_text() can put in 'method'. Keep this in step with the
    # engine; the last test in this class fails if it drifts.
    ALL_METHODS = ['database', 'too_short', 'reading_model', 'ml_prediction', 'unavailable']

    # Roughly the length the engine really produces, so the word count below is
    # measuring the template rather than the shortness of a fixture.
    EXPLANATION = ('Analysis system is not available. Please ensure the model '
                   'files are present before trying again.')

    def _card(self, method):
        """The How-was-this-determined card, as rendered for one method."""
        analysis = NewsAnalysis.objects.create(
            headline='Some headline', score=50, result_label='Not Sure',
            explanation=self.EXPLANATION, method=method,
        )
        body = self.client.get(reverse('result', args=[analysis.id])).content.decode()

        start = body.index('How was this determined?')
        return body[start:body.index('</section>', start)]

    def _sentence(self, method):
        """
        The same card with its line breaks collapsed, for asserting on wording.

        Whitespace is not significant in HTML, so a template author reflowing a
        paragraph must not break a test. Asserting on the raw markup did exactly
        that: "at least {{ min_words }} words" had a newline inside it.
        """
        return ' '.join(self._card(method).split())

    def test_every_method_explains_itself(self):
        for method in self.ALL_METHODS:
            with self.subTest(method=method):
                card = self._card(method)
                # Strip the tags and see whether any words are left under the heading
                text = re.sub(r'<[^>]+>', ' ', card.split('How was this determined?')[1])
                self.assertGreater(
                    len(text.split()), 8,
                    f"method={method!r} renders an empty card",
                )

    def test_a_refusal_says_the_model_was_never_run(self):
        """
        The point of the wording. "Not Sure" reads like the model looked and
        could not decide — it never looked at all.
        """
        card = self._sentence('too_short')

        self.assertIn('Not enough text', card)
        self.assertIn('nothing was checked', card.lower())

    def test_a_refusal_says_how_much_text_to_paste(self):
        """
        The only sentence on this card the reader can act on. It went missing
        once already, when the card was three sentences of why we refused and
        none of what to do about it.
        """
        self.assertIn(f'at least {MIN_WORDS} words', self._sentence('too_short'))

    def test_the_refusal_card_quotes_the_real_limit(self):
        """
        Guards against the number being typed into the template. Raise the limit
        and the page must follow, or it tells people to paste an amount that
        will be refused again.
        """
        with mock.patch.object(views, 'MIN_WORDS', 40):
            self.assertIn('at least 40 words', self._sentence('too_short'))

    def test_a_refusal_does_not_claim_a_model_read_it(self):
        card = self._card('too_short')

        self.assertNotIn('Judged on roughly the first 200 words', card)

    def test_an_unrecognised_method_falls_back_to_the_explanation(self):
        """A method nobody has written wording for must still say something."""
        card = self._card('some_future_step')

        self.assertIn('Analysis system is not available', card)

    def test_the_method_list_here_matches_the_engine(self):
        """
        Guards the list above. A new method added to the engine without wording
        here would otherwise slip through every test in this class.
        """
        import re as _re
        from pathlib import Path

        source = (Path(__file__).parent / 'analysis_engine.py').read_text(encoding='utf-8')
        found = set(_re.findall(r"'method':\s*'(\w+)'", source))

        self.assertEqual(found, set(self.ALL_METHODS))


# ============================================================================
# THE PAGE MUST NOT RANK THE MODEL'S TWO ANSWERS IN PROSE
# ============================================================================

class ReadingModelCardTests(TestCase):
    """
    The card used to name which verdict was the more reliable one. That claim was
    hardcoded once, went stale at the next retrain, and told users in bold to
    trust the WEAKER answer more: it called "Likely Fake" stronger while
    cutoff.json said fake 86.9% against real 90.1%.

    It was then computed from the measurements, which was correct but not worth
    the two branches — the reader cannot act on knowing which answer is weaker.
    So the card now says the same thing for both verdicts, and these tests keep
    the ranking from creeping back in.
    """

    def _card(self, label):
        analysis = NewsAnalysis.objects.create(
            headline='Some headline', score=50, result_label=label,
            explanation='x', method='reading_model', confidence=86.9)
        page = self.client.get(reverse('result', args=[analysis.id]))
        return ' '.join(page.content.decode().split())

    def test_the_card_says_why_there_is_a_prediction_at_all(self):
        for label in ('Likely Fake', 'Likely Real'):
            card = self._card(label)
            self.assertIn('not found in our dataset', card, label)
            self.assertIn('language and writing patterns', card, label)

    def test_neither_verdict_is_ranked_against_the_other(self):
        """The regression that started this. No retrain can make prose stale."""
        for label in ('Likely Fake', 'Likely Real'):
            card = self._card(label)
            for claim in ('weaker verdict', 'stronger verdict',
                          'less reliable', 'more reliable of its'):
                self.assertNotIn(claim, card, f'{label}: {claim}')

    def test_likely_real_is_not_presented_as_clearance(self):
        """
        "Likely Real" reads as a clean bill of health. The model never checked
        whether anything in the article happened, so it is not one.
        """
        self.assertIn('not as clearance', self._card('Likely Real'))

    def test_likely_fake_does_not_carry_the_clearance_line(self):
        self.assertNotIn('not as clearance', self._card('Likely Fake'))


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

        self.assertIn('labelled FAKE by Kaggle ISOT', analysis.explanation)

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

    def test_bulk_inserted_rows_still_get_a_body_key(self):
        """
        Same trap, second key. Miss this and a reload leaves every row with an
        empty body_key: the body lookup finds nothing, silently, with no error
        anywhere to point at the cause.
        """
        self._load([('New One', VIDEO_BODY, 'REAL', 'BBC', SOURCE)])

        saved = KnownArticle.objects.get(source='BBC')
        self.assertEqual(saved.body_key, body_key(VIDEO_BODY))
        self.assertEqual(best_match_by_body(VIDEO_BODY), saved)

    def test_the_body_key_is_built_from_the_extract_we_actually_store(self):
        """
        The key has to describe the stored text, not the full body it was cut
        from — otherwise the loader and save() would disagree about the same row.
        """
        from load_mega_data import EXTRACT_CHARS

        long_body = VIDEO_BODY + ' ' + ('Further reporting followed. ' * 60)
        self.assertGreater(len(long_body), EXTRACT_CHARS)

        self._load([('New One', long_body, 'REAL', 'BBC', SOURCE)])

        saved = KnownArticle.objects.get(source='BBC')
        self.assertEqual(saved.body_key, body_key(saved.article_text))
        # Re-saving must not move it, which is what proves the two agree
        saved.save()
        self.assertEqual(
            KnownArticle.objects.get(source='BBC').body_key, body_key(VIDEO_BODY)
        )


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

# Wire-service prose the reading model scores near 0, well outside the unsure
# band, and long enough to clear MIN_WORDS. The checkbox tests need an article
# the band would NEVER search on its own, so that a search happening at all
# proves the box did it. Written from scratch, so no KnownArticle can match it.
REAL_WIRE = (
    "Central bank holds interest rates steady at four percent",
    "The central bank left its benchmark interest rate unchanged at 4 percent "
    "on Thursday, citing easing inflation and a cooling labour market. In a "
    "statement following the two-day meeting, policymakers said they would "
    "continue to assess incoming data before adjusting policy further. Eight of "
    "the nine committee members voted to hold, with one favouring a "
    "quarter-point cut. The decision was in line with expectations from "
    "economists surveyed last week.",
)

# The same article as one textarea paste: headline on the first line, body under
# it, which is how analyze() splits what it receives.
WIRE_PASTE = '\n'.join(REAL_WIRE)


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

    def test_a_requested_search_does_not_claim_the_model_was_unsure(self):
        """
        Ticking the box on a confident article puts this card under a confident
        verdict. If it said "the model was unsure" it would contradict the
        percentage sitting directly above it.
        """
        response = self._page('Likely Fake', dict(FOUND_FAKE, reason='requested'))
        self.assertContains(response, 'You asked us to check')
        self.assertNotContains(response, 'The model was unsure')

    def test_an_automatic_search_says_the_model_was_unsure(self):
        response = self._page('Likely Fake', dict(FOUND_FAKE, reason='unsure'))
        self.assertContains(response, 'The model was unsure')
        self.assertNotContains(response, 'You asked us to check')

    def test_a_row_saved_before_reasons_existed_claims_neither(self):
        """Older analyses have no reason stored. Silence beats a guess."""
        response = self._page('Likely Fake', FOUND_FAKE)
        self.assertNotContains(response, 'You asked us to check')
        self.assertNotContains(response, 'The model was unsure')


class SearchWebCheckboxTests(KnownArticleMixin, TestCase):
    """
    The opt-in only ever ADDS searches.

    Every test here stubs check_online, so nothing reaches the network and no
    API allowance is spent. What is being tested is the gate, not the search.
    """

    def setUp(self):
        super().setUp()
        self.calls = []

        def fake_search(headline, article_text='', **kwargs):
            self.calls.append(headline)
            return dict(FOUND_REAL)

        self._real_search = analysis_engine.check_online
        analysis_engine.check_online = fake_search

        # A key must appear to exist or the form deletes the field and
        # should_check() refuses outright, which would hide real failures here.
        from . import web_check
        self.web_check = web_check
        self._read_key = web_check._read_key
        web_check._read_key = lambda: 'test-key'
        self._band = web_check.unsure_band
        web_check.unsure_band = lambda: (0.01, 0.99)

    def tearDown(self):
        analysis_engine.check_online = self._real_search
        self.web_check._read_key = self._read_key
        self.web_check.unsure_band = self._band
        super().tearDown()

    # ── the engine gate ──

    def test_ticking_the_box_searches_a_confident_article(self):
        state = analysis_engine._load_reading_model()
        if not state:
            self.skipTest('reading model not present')

        head, body = REAL_WIRE
        score = analysis_engine._fake_score(state, f'{head}\n{body}')
        self.assertFalse(self.web_check.should_check(score),
                         'fixture is meant to be outside the unsure band')

        analysis_engine.analyze_text(head, body, search_web=False)
        self.assertEqual(self.calls, [], 'unticked must not spend a call')

        result = analysis_engine.analyze_text(head, body, search_web=True)
        self.assertEqual(self.calls, [head])
        self.assertEqual(result['web']['reason'], 'requested')

    def test_the_band_still_searches_on_its_own_when_unticked(self):
        """
        The reason this is an 'or' and not a switch. Nobody sees the raw score,
        so nobody can tell which articles need the box.
        """
        analysis_engine.analyze_text('Unsure probe', 'body ' * 40,
                                     search_web=False)
        before = list(self.calls)

        # Force every score into the band, whatever the model says.
        self.web_check.unsure_band = lambda: (-1.0, 2.0)
        result = analysis_engine.analyze_text('Unsure probe', 'body ' * 40,
                                             search_web=False)

        if result.get('method') != 'reading_model':
            self.skipTest('reading model not present')
        self.assertEqual(len(self.calls), len(before) + 1)
        self.assertEqual(result['web']['reason'], 'unsure')

    def test_a_database_hit_ignores_the_box_entirely(self):
        result = analysis_engine.analyze_text(KNOWN_FAKE, '', search_web=True)

        self.assertEqual(result['method'], 'database')
        self.assertIsNone(result.get('web'))
        self.assertEqual(self.calls, [], 'a known article costs no API call')

    # ── the form and the view ──

    def test_the_checkbox_is_offered_when_a_key_exists(self):
        response = self.client.get(reverse('analyze'))
        self.assertContains(response, 'search-web')
        self.assertContains(response, 'Also check the web')

    def test_the_checkbox_is_absent_without_a_key(self):
        """A box that could only ever fail should not be on the page."""
        self.web_check._read_key = lambda: None
        response = self.client.get(reverse('analyze'))
        self.assertNotContains(response, 'search-web')

    def test_an_unticked_form_reports_no_request(self):
        form = NewsInputForm({'news_content': WIRE_PASTE})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertFalse(form.wants_web_search())

    def test_a_ticked_form_reports_a_request(self):
        form = NewsInputForm({'news_content': WIRE_PASTE,
                              'search_web': 'on'})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertTrue(form.wants_web_search())

    def test_wants_web_search_is_false_when_the_field_was_dropped(self):
        """A stale POST carrying search_web with no key must not sneak a call."""
        self.web_check._read_key = lambda: None
        form = NewsInputForm({'news_content': WIRE_PASTE,
                              'search_web': 'on'})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertFalse(form.wants_web_search())

    def test_the_view_passes_the_box_through_to_the_engine(self):
        seen = {}
        real = views.analyze_text

        def spy(headline, article_text='', **kwargs):
            seen.update(kwargs)
            return real(headline, article_text, **kwargs)

        views.analyze_text = spy
        try:
            self.client.post(reverse('analyze'),
                             {'news_content': KNOWN_REAL, 'search_web': 'on'})
        finally:
            views.analyze_text = real

        self.assertIs(seen.get('search_web'), True)

    def test_a_requested_search_is_saved_on_the_analysis(self):
        state = analysis_engine._load_reading_model()
        if not state:
            self.skipTest('reading model not present')

        head, body = REAL_WIRE
        self.client.post(reverse('analyze'),
                         {'news_content': f'{head}\n{body}', 'search_web': 'on'})

        analysis = NewsAnalysis.objects.latest('id')
        self.assertEqual(analysis.method, 'reading_model')
        self.assertEqual(analysis.web_check['reason'], 'requested')

        page = self.client.get(reverse('result', args=[analysis.id]))
        self.assertContains(page, 'You asked us to check')


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


class TemplateRenderTests(KnownArticleMixin, TestCase):
    """
    Every page must render to finished HTML, with no template source left in it.

    This class exists because of a bug nothing else caught. Django's {# #}
    comment is SINGLE-LINE ONLY: written across several lines it stops being a
    comment, and the whole block — design notes, section banners and all — is
    printed into the page for the reader to see. Four templates were doing it.

    The suite missed it because every other test asserts that something SHOULD
    be present. Nothing asserted on what should be absent, and leaked comments
    are invisible to a test looking for 'Likely Fake' in the response. So these
    check the opposite direction, once per page.
    """

    # Any of these surviving into the response means the template did not fully
    # render. Checked as raw text, since a real page has no reason to contain
    # them and Django escapes anything a user could paste that did.
    LEAKS = ('{#', '{%', '{{', 'endcomment')

    def _assert_clean(self, response):
        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        for leak in self.LEAKS:
            self.assertNotIn(
                leak, body,
                f"unrendered template syntax {leak!r} reached the page — a "
                f"multi-line {{# #}} comment is the usual cause; use "
                f"{{% comment %}} instead")

    def test_home_renders_clean(self):
        self._assert_clean(self.client.get(reverse('home')))

    def test_analyze_renders_clean(self):
        self._assert_clean(self.client.get(reverse('analyze')))

    def test_refused_input_renders_clean(self):
        """The validation-error path renders different markup, so check it too."""
        response = self.client.post(reverse('analyze'),
                                    {'news_content': 'یہ اردو میں لکھا گیا ہے'})
        self._assert_clean(response)
        self.assertContains(response, 'only read English')

    def test_empty_dashboard_renders_clean(self):
        self._assert_clean(self.client.get(reverse('dashboard')))

    def test_populated_dashboard_renders_clean(self):
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})
        self._assert_clean(self.client.get(reverse('dashboard')))

    def test_result_renders_clean(self):
        """A result with a web check, so the online block renders as well."""
        row = NewsAnalysis.objects.create(
            headline='Render probe',
            article_text='Body.',
            score=0.5,
            result_label='Likely Fake',
            explanation='Probe.',
            confidence=86.9,
            method='reading_model',
            web_check={
                'status': 'found', 'verdict': 'REAL', 'confidence': 'high',
                'summary': 'Widely reported.', 'reporting': ['BBC'],
                'debunking': [], 'factcheck_rating': None,
                'sources': [{'title': 'BBC News', 'url': 'https://bbc.co.uk'}],
                'queries': [], 'error': None, 'elapsed': 4.1,
            },
        )

        self._assert_clean(self.client.get(reverse('result', args=[row.id])))

    def test_database_result_renders_clean(self):
        """The Step 1 path shows the matched article, which no other case does."""
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})
        row = NewsAnalysis.objects.latest('id')

        self._assert_clean(self.client.get(reverse('result', args=[row.id])))


# ============================================================================
# DELETING SAVED ANALYSES
# ============================================================================

class DeleteAnalysisTests(KnownArticleMixin, TestCase):
    """
    The delete buttons on the dashboard.

    Most of these assert what must NOT happen. A delete feature is only as good
    as the things it refuses to do, and the expensive mistake here is not a
    button that fails to delete — it is one that deletes on a GET, or reaches
    past the history table into the fact-check corpus.
    """

    def _make_analysis(self):
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})
        return NewsAnalysis.objects.latest('id')

    def test_post_deletes_the_row(self):
        row = self._make_analysis()

        self.client.post(reverse('delete_analysis', args=[row.id]))

        self.assertFalse(NewsAnalysis.objects.filter(id=row.id).exists())

    def test_get_does_not_delete(self):
        """
        A delete reachable by GET is one a browser prefetch or a crawler can
        fire with nobody clicking anything, and this app has no login to fall
        back on. require_POST must answer 405 and leave the row alone.
        """
        row = self._make_analysis()

        response = self.client.get(reverse('delete_analysis', args=[row.id]))

        self.assertEqual(response.status_code, 405)
        self.assertTrue(NewsAnalysis.objects.filter(id=row.id).exists())

    def test_deleting_a_missing_row_is_404_not_silent_success(self):
        """A stale tab should be told the row is gone, not redirected as if it worked."""
        response = self.client.post(reverse('delete_analysis', args=[99999]))

        self.assertEqual(response.status_code, 404)

    def test_delete_leaves_other_rows_alone(self):
        keep = self._make_analysis()
        drop = self._make_analysis()

        self.client.post(reverse('delete_analysis', args=[drop.id]))

        self.assertTrue(NewsAnalysis.objects.filter(id=keep.id).exists())

    def test_clear_all_with_confirmation_empties_the_table(self):
        self._make_analysis()
        self._make_analysis()

        self.client.post(reverse('clear_analyses'), {'confirm': 'DELETE'})

        self.assertEqual(NewsAnalysis.objects.count(), 0)

    def test_clear_all_without_confirmation_does_nothing(self):
        """
        The typed confirmation is checked server-side. A JavaScript confirm()
        dialog is skipped by anything that is not a browser, so a replayed or
        hand-made POST must not be enough to empty the table.
        """
        self._make_analysis()

        self.client.post(reverse('clear_analyses'))

        self.assertEqual(NewsAnalysis.objects.count(), 1)

    def test_clear_all_by_get_does_nothing(self):
        self._make_analysis()

        response = self.client.get(reverse('clear_analyses'))

        self.assertEqual(response.status_code, 405)
        self.assertEqual(NewsAnalysis.objects.count(), 1)

    def test_clearing_history_does_not_touch_the_fact_check_corpus(self):
        """
        The corpus is what the app KNOWS; history is what it was asked. Wiping
        the corpus from a public button would silently turn every future
        database match into a model guess, with nothing on screen saying so.
        """
        self._make_analysis()
        before = KnownArticle.objects.count()

        self.client.post(reverse('clear_analyses'), {'confirm': 'DELETE'})

        self.assertEqual(KnownArticle.objects.count(), before)

    def test_dashboard_offers_delete_controls_when_rows_exist(self):
        row = self._make_analysis()

        html = self.client.get(reverse('dashboard')).content.decode()

        self.assertIn(reverse('delete_analysis', args=[row.id]), html)
        self.assertIn(reverse('clear_analyses'), html)
        # POST forms carry a CSRF token; a bare link would not.
        self.assertIn('csrfmiddlewaretoken', html)

    def test_empty_dashboard_offers_nothing_to_delete(self):
        html = self.client.get(reverse('dashboard')).content.decode()

        self.assertNotIn(reverse('clear_analyses'), html)

    def test_recent_table_does_not_number_rows_by_database_id(self):
        """
        The # column must not print the row's database id. SQLite never reuses
        one, so after Clear all the next analysis appeared as #26 and the table
        looked like the delete had silently failed.
        """
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})
        self.client.post(reverse('clear_analyses'), {'confirm': 'DELETE'})
        self.client.post(reverse('analyze'), {'news_content': KNOWN_REAL})

        row = NewsAnalysis.objects.latest('id')
        html = self.client.get(reverse('dashboard')).content.decode()

        self.assertGreater(row.id, 1, "test needs a row whose id is not 1")
        self.assertIn('<td class="muted">1</td>', html)
        self.assertNotIn(f'<td class="muted">{row.id}</td>', html)

    def test_newest_row_carries_the_highest_number(self):
        """
        Numbering runs bottom-to-top: oldest is 1, newest is highest. Numbering
        the newest row 1 means every new analysis takes the number off the row
        that had it, so nothing in the table keeps a stable label.
        """
        for i in range(3):
            self.client.post(reverse('analyze'), {'news_content': f'{KNOWN_FAKE} {i}'})

        html = self.client.get(reverse('dashboard')).content.decode()
        shown = re.findall(r'<td class="muted">(\d+)</td>', html)

        # Rendered newest-first, so the numbers count down to 1 at the bottom.
        self.assertEqual(shown, ['3', '2', '1'])

    def test_numbering_restarts_at_one_after_clear_all(self):
        """The complaint that started this: a cleared dashboard must count from 1."""
        self.client.post(reverse('analyze'), {'news_content': KNOWN_FAKE})
        self.client.post(reverse('clear_analyses'), {'confirm': 'DELETE'})
        self.client.post(reverse('analyze'), {'news_content': KNOWN_REAL})

        html = self.client.get(reverse('dashboard')).content.decode()

        self.assertEqual(re.findall(r'<td class="muted">(\d+)</td>', html), ['1'])
