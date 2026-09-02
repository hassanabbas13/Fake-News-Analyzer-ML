"""
models.py — Database Models for the Analyzer App
==================================================
"""

from django.db import models

from .text_matching import normalize_headline

class KnownArticle(models.Model):
    """
    The fact-check corpus, loaded by load_mega_data.py — one row per unique
    headline. Row count is deliberately not stated here; count() it instead.
    """

    # How much weight a label deserves. Lower number = more trustworthy, so
    # ordering by this field puts the best evidence first (see best_match).
    BASIS_FACT_CHECK = 'fact_check'   # a fact-checker investigated this claim
    BASIS_SOURCE = 'source'           # labelled by who published it, nothing more
    BASIS_CHOICES = [
        (BASIS_FACT_CHECK, 'Verified by a fact-checker'),
        (BASIS_SOURCE, 'Inferred from the publisher'),
    ]
    BASIS_RANK = {BASIS_FACT_CHECK: 0, BASIS_SOURCE: 1}

    # The headline as it was published, kept for display
    headline = models.CharField(max_length=1000)

    # The tidied-up version that lookups actually compare against. Filled in
    # automatically by save(); load_mega_data.py sets it explicitly because
    # bulk_create() bypasses save().
    headline_key = models.CharField(max_length=1000, db_index=True, default='')

    # A short opening extract, so a match can show the user what it found
    # instead of only asserting a verdict. Full bodies are not stored.
    article_text = models.TextField(blank=True, default='')

    label = models.CharField(max_length=50) # 'FAKE' or 'REAL'
    source = models.CharField(max_length=100)

    # WHY this row carries its label. Without this, "Reuters published it" and
    # "PolitiFact debunked it" look equally authoritative, which they are not.
    label_basis = models.CharField(
        max_length=20, choices=BASIS_CHOICES, default=BASIS_SOURCE
    )

    class Meta:
        indexes = [
            # Serves the lookup in analysis_engine.best_match(): find by tidied
            # headline, best evidence first.
            models.Index(fields=['headline_key', 'label_basis'], name='knownarticle_lookup_idx'),
        ]

    def __str__(self):
        return f"[{self.label}] {self.headline[:50]}"

    def save(self, *args, **kwargs):
        """
        Keep headline_key in step with headline automatically, so no caller can
        forget it and quietly create a row that lookups will never find.

        Note that bulk_create() does NOT call save() — load_mega_data.py sets
        headline_key itself for exactly that reason.
        """
        self.headline_key = normalize_headline(self.headline)
        super().save(*args, **kwargs)

    @property
    def is_fact_checked(self):
        """True when a fact-checker actually investigated this claim."""
        return self.label_basis == self.BASIS_FACT_CHECK

    def describe_basis(self):
        """
        Plain-English account of why this row carries the label it does, so the
        result page can be specific instead of implying every verdict is equal.
        """
        if self.is_fact_checked:
            return f"{self.source} fact-checked this claim and rated it {self.label}."
        return (
            f"This is labelled {self.label} because of where it was published "
            f"({self.source}) — nobody checked the individual claims in it."
        )


class NewsAnalysis(models.Model):
    """
    Stores one news analysis result.
    """
    headline = models.CharField(max_length=500, blank=True, default='')
    article_text = models.TextField(blank=True, default='')
    score = models.IntegerField()
    result_label = models.CharField(max_length=20)
    explanation = models.TextField(blank=True, default='')  # Human-readable verdict reasoning
    analyzed_at = models.DateTimeField(auto_now_add=True)

    # New Fields for Hybrid System
    confidence = models.FloatField(null=True, blank=True)
    method = models.CharField(max_length=50, null=True, blank=True) # "database" or "ml_prediction"

    # When Step 1 found a known article, a COPY of what it found — so the result
    # page can show the user the evidence rather than only asserting a verdict.
    # Copied rather than linked on purpose: load_mega_data.py deletes every
    # KnownArticle on each reload, which would break or erase a real link.
    matched_headline = models.CharField(max_length=1000, blank=True, default='')
    matched_extract = models.TextField(blank=True, default='')

    # What the web search found, when it ran. Stored as JSON rather than as a
    # dozen columns because it is a whole shape -- verdict, outlets reporting,
    # outlets debunking, links, why it failed -- and because that shape will
    # change as web_check.py improves, which columns would make painful.
    #
    # Null means the search never ran: either the model was confident enough not
    # to need it, or there was no API key. That is DIFFERENT from a stored result
    # with status 'nothing', which means the search ran and found no coverage --
    # weak evidence against the article. The template must not conflate them.
    web_check = models.JSONField(null=True, blank=True)
    
    class Meta:
        ordering = ['-analyzed_at']
        verbose_name_plural = "News Analyses"
    
    def __str__(self):
        return f"{self.headline[:50]} → {self.result_label} (Score: {self.score})"
