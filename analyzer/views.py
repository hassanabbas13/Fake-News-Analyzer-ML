"""
The page logic. One function per page, wired up in urls.py.

    home            landing page
    analyze         the form, and the POST that runs an analysis
    result          one saved analysis
    dashboard       history table and summary counts
    dashboard_data  JSON for the Chart.js charts
    delete_analysis / clear_analyses   remove saved history, POST only

The two delete views are POST-only because a delete reachable by GET is one a
browser prefetch or a crawler can fire, and this app has no login.
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .forms import NewsInputForm
from .models import NewsAnalysis
from .analysis_engine import analyze_text
from .input_check import MIN_WORDS


def home(request):
    """Landing page, with a count of how many analyses have been run."""
    total_analyses = NewsAnalysis.objects.count()
    return render(request, 'analyzer/home.html', {
        'total_analyses': total_analyses,
    })


def analyze(request):
    """GET shows an empty form. POST runs the analysis, saves it, and redirects."""
    if request.method == 'POST':
        form = NewsInputForm(request.POST)

        if form.is_valid():
            # One textarea, split on the first newline: the headline is what the
            # database lookup searches on, the rest is the article body.
            content = form.cleaned_data['news_content'].strip()
            headline, _, body = content.partition('\n')
            headline = headline.strip()
            article_text = body.strip()

            # Only ADDS searches: the engine still runs one on its own inside the
            # unsure band, ticked or not.
            result = analyze_text(headline, article_text,
                                  search_web=form.wants_web_search())

            # A long single-line paste is an article body, not a headline — keep
            # the full text so nothing is lost when the headline is truncated.
            if not article_text and len(headline) > 500:
                article_text = headline

            display_headline = headline if len(headline) <= 500 else headline[:497] + "..."

            matched = result.get('matched_article')

            analysis = NewsAnalysis.objects.create(
                headline=display_headline,
                article_text=article_text,
                score=result['score'],
                result_label=result['label'],
                explanation=result['explanation'],
                confidence=result.get('confidence'),
                method=result.get('method'),
                # Copied across so the result page can show its evidence.
                matched_headline=matched.headline if matched else '',
                matched_extract=matched.article_text if matched else '',
                # None when the search did not run at all, which the result page
                # renders differently from "ran and found nothing".
                web_check=result.get('web'),
            )

            return redirect('result', analysis_id=analysis.id)
    else:
        form = NewsInputForm()

    return render(request, 'analyzer/analyze.html', {'form': form})


def result(request, analysis_id):
    """One saved analysis, with the wording that explains what its number means."""
    analysis = get_object_or_404(NewsAnalysis, id=analysis_id)

    # 'Not Sure' is kept for rows saved by an earlier version of the engine. The
    # reading model no longer produces it, but old entries must still render.
    if analysis.result_label == 'Likely Real':
        result_class = 'result-real'
        result_icon = '✅'
    elif analysis.result_label == 'Not Sure':
        result_class = 'result-unsure'
        result_icon = '🤷'
    else:
        result_class = 'result-fake'
        result_icon = '❌'

    # What the percentage next to the verdict actually means. For the reading model
    # it is how often a verdict like this is CORRECT, not how sure the model feels,
    # so it must never be labelled "confidence".
    if analysis.method == 'reading_model':
        confidence_label = 'of these verdicts are correct'
    elif analysis.method == 'database':
        # Not "headline match": a row can be found by its opening words too, so a
        # 100% hit is not always a headline hit.
        confidence_label = 'dataset match'
    else:
        confidence_label = 'Confidence'

    return render(request, 'analyzer/result.html', {
        'analysis': analysis,
        'result_class': result_class,
        'result_icon': result_icon,
        'confidence': analysis.confidence,
        'confidence_label': confidence_label,
        'method': analysis.method,
        # Sent from here so the number lives once, in input_check, rather than in a
        # template nobody would think to update.
        'min_words': MIN_WORDS,
    })


def dashboard(request):
    """Summary counts and the last 20 analyses. The charts fetch their own data."""
    analyses = NewsAnalysis.objects.all()
    total = analyses.count()

    # 'Not Sure' is counted explicitly rather than lumped in with anything else. A
    # dashboard whose three numbers do not add up to the total looks broken.
    real_count = analyses.filter(result_label='Likely Real').count()
    fake_count = analyses.filter(result_label='Likely Fake').count()
    unsure_count = analyses.filter(result_label='Not Sure').count()

    real_pct = round((real_count / total * 100), 1) if total > 0 else 0
    fake_pct = round((fake_count / total * 100), 1) if total > 0 else 0
    unsure_pct = round((unsure_count / total * 100), 1) if total > 0 else 0

    recent = list(analyses[:20])

    # Number them bottom-to-top, so a new analysis adds a number on the end rather
    # than renumbering everything.
    #
    # Not the database id: SQLite never reuses one, so after Clear all the next
    # analysis came back as 26 and the table looked like the delete had failed. Not
    # forloop.counter either, since the list is newest-first. Done here because
    # Django has an `add` filter but no subtract, and total - offset is the point.
    for offset, item in enumerate(recent):
        item.row_number = total - offset

    return render(request, 'analyzer/dashboard.html', {
        'total': total,
        'real_count': real_count,
        'fake_count': fake_count,
        'unsure_count': unsure_count,
        'real_pct': real_pct,
        'fake_pct': fake_pct,
        'unsure_pct': unsure_pct,
        'recent': recent,
    })


def dashboard_data(request):
    """JSON for the two Chart.js charts, fetched by the dashboard page."""
    analyses = NewsAnalysis.objects.all()

    pie_data = {
        'labels': ['Likely Real', 'Likely Fake', 'Not Sure'],
        'counts': [
            analyses.filter(result_label='Likely Real').count(),
            analyses.filter(result_label='Likely Fake').count(),
            analyses.filter(result_label='Not Sure').count(),
        ],
    }

    # 'Word-Counter' only appears when the reading model could not be loaded, so a
    # non-zero bar there is a useful warning that the app is running on the old
    # fallback rather than the model we measured.
    method_data = {
        'labels': ['Database Match', 'Reading Model', 'Word-Counter'],
        'counts': [
            analyses.filter(method='database').count(),
            analyses.filter(method='reading_model').count(),
            analyses.filter(method='ml_prediction').count(),
        ],
    }

    return JsonResponse({
        'pie_data': pie_data,
        'method_data': method_data,
    })


# Both delete views only ever touch NewsAnalysis, the record of what this app was
# asked. Neither can reach KnownArticle, the fact-check corpus that Step 1 looks
# up: wiping that from a button on a public page would quietly turn every future
# database match into a model guess, with nothing on screen saying so. Deleting
# history is cheap and reversible. Deleting the corpus is neither.

@require_POST
def delete_analysis(request, analysis_id):
    """
    Delete one saved analysis, then return to the dashboard.

    get_object_or_404 rather than filter().delete(): a stale tab whose row is
    already gone should say 404, not report success for deleting nothing.
    """
    analysis = get_object_or_404(NewsAnalysis, id=analysis_id)
    analysis.delete()
    return redirect('dashboard')


@require_POST
def clear_analyses(request):
    """
    Delete every saved analysis.

    Guarded by a typed confirmation rather than a JavaScript confirm(), which is
    trivially skipped by anything that is not a browser. Without confirm=DELETE
    this is a no-op redirect, so a replayed POST cannot wipe the table.
    """
    if request.POST.get('confirm') == 'DELETE':
        NewsAnalysis.objects.all().delete()
    return redirect('dashboard')
