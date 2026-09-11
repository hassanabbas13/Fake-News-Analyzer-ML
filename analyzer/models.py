"""Database models for the analyzer app."""

from django.db import models

# Imported as a module: there is a field below also called body_key.
from . import text_matching

class KnownArticle(models.Model):
    """
    The fact-check corpus, loaded by load_mega_data.py — one row per unique
    headline. Row count is deliberately not stated here; count() it instead.
    """

    # Lower rank = better evidence, so ordering by this puts fact-checks first.
    BASIS_FACT_CHECK = 'fact_check'   # a fact-checker investigated this claim
    BASIS_SOURCE = 'source'           # labelled by who published it, nothing more
    BASIS_CHOICES = [
        (BASIS_FACT_CHECK, 'Verified by a fact-checker'),
        (BASIS_SOURCE, 'Inferred from the publisher'),
    ]
    BASIS_RANK = {BASIS_FACT_CHECK: 0, BASIS_SOURCE: 1}

    # The headline as published, kept for display
    headline = models.CharField(max_length=1000)

    # The tidied version lookups actually compare against. Set by save();
    # load_mega_data.py sets it explicitly because bulk_create() skips save().
    headline_key = models.CharField(max_length=1000, db_index=True, default='')

    # A short opening extract, so a match can show what it found rather than only
    # asserting a verdict. Full bodies are not stored.
    article_text = models.TextField(blank=True, default='')

    # A second exact way in: the article's opening words, tidied the same way.
    # Headlines are brittle — drop "(VIDEO)" off the end and a story we hold a
    # fact-check for stops being recognisable. Blank means too short to key on,
    # and the lookup skips blanks rather than letting them collide.
    body_key = models.CharField(max_length=400, db_index=True, default='', blank=True)

    label = models.CharField(max_length=50) # 'FAKE' or 'REAL'
    source = models.CharField(max_length=100)

    # WHY this row carries its label: "Reuters published it" and "PolitiFact
    # debunked it" are not equally authoritative.
    label_basis = models.CharField(
        max_length=20, choices=BASIS_CHOICES, default=BASIS_SOURCE
    )

    class Meta:
        indexes = [
            # Serves best_match(): find by tidied headline, best evidence first.
            models.Index(fields=['headline_key', 'label_basis'], name='knownarticle_lookup_idx'),
        ]

    def __str__(self):
        return f"[{self.label}] {self.headline[:50]}"

    def save(self, *args, **kwargs):
        """Keep both lookup keys in step with their text, so no caller can create a
        row lookups will never find. bulk_create() skips this — load_mega_data.py
        sets both keys itself."""
        self.headline_key = text_matching.normalize_headline(self.headline)
        self.body_key = text_matching.body_key(self.article_text)
        super().save(*args, **kwargs)

    @property
    def is_fact_checked(self):
        """True when a fact-checker actually investigated this claim."""
        return self.label_basis == self.BASIS_FACT_CHECK

    def describe_basis(self):
        """A phrase that slots into "Found in our dataset of N articles, <this>."
        The verb carries the distinction: labelled by a publisher, fact-checked by
        a fact-checker."""
        verb = 'fact-checked' if self.is_fact_checked else 'labelled'
        return f"{verb} {self.label} by {self.source}"


class NewsAnalysis(models.Model):
    """One saved analysis: what the app was asked, and how it answered."""

    headline = models.CharField(max_length=500, blank=True, default='')
    article_text = models.TextField(blank=True, default='')
    score = models.IntegerField()
    result_label = models.CharField(max_length=20)
    explanation = models.TextField(blank=True, default='')
    analyzed_at = models.DateTimeField(auto_now_add=True)
    confidence = models.FloatField(null=True, blank=True)
    method = models.CharField(max_length=50, null=True, blank=True) # "database" or "ml_prediction"

    # A COPY of what Step 1 matched, not a link: load_mega_data.py can delete
    # every KnownArticle on a reload, which would break a real foreign key.
    matched_headline = models.CharField(max_length=1000, blank=True, default='')
    matched_extract = models.TextField(blank=True, default='')

    # What the web search found. JSON because it is a whole shape -- verdict,
    # outlets, links, why it failed -- that changes as web_check.py improves.
    #
    # NULL means the search never ran. Different from a stored status of 'nothing',
    # which means it ran and found no coverage: weak evidence against the article.
    # The template must not conflate the two.
    web_check = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ['-analyzed_at']
        verbose_name_plural = "News Analyses"

    def __str__(self):
        return f"{self.headline[:50]} → {self.result_label} (Score: {self.score})"
