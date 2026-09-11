"""Django admin registration, for browsing the two tables at /admin/."""

from django.contrib import admin
from .models import KnownArticle, NewsAnalysis


@admin.register(KnownArticle)
class KnownArticleAdmin(admin.ModelAdmin):
    list_display = ('headline', 'label', 'label_basis', 'source')
    list_filter = ('label', 'label_basis', 'source')
    search_fields = ('headline_key',)     # indexed, so searching 45k rows stays fast
    readonly_fields = ('headline_key',)   # filled automatically by save()


@admin.register(NewsAnalysis)
class NewsAnalysisAdmin(admin.ModelAdmin):
    list_display = ('headline', 'result_label', 'confidence', 'method', 'analyzed_at')
    list_filter = ('result_label', 'method')
    search_fields = ('headline',)
    ordering = ('-analyzed_at',)
