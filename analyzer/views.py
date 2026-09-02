"""
views.py — All Views (Page Logic) for the Analyzer App
========================================================

Each view function handles one page of the website.
Views connect URLs to templates, process data, and return HTML.

Views in this file:
    1. home         → Landing page (GET)
    2. analyze      → Form page + analysis processing (GET/POST)
    3. result       → Display analysis result (GET)
    4. dashboard    → Analytics dashboard (GET)
    5. dashboard_data → JSON API for Chart.js charts (GET)
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse

from .forms import NewsInputForm
from .models import NewsAnalysis
from .analysis_engine import analyze_text


# ============================================================================
# VIEW 1: HOME PAGE
# ============================================================================

def home(request):
    """
    Landing page — shows project overview and a call-to-action button.
    Also shows a quick stat of total analyses performed.
    """
    total_analyses = NewsAnalysis.objects.count()
    return render(request, 'analyzer/home.html', {
        'total_analyses': total_analyses,
    })


# ============================================================================
# VIEW 2: ANALYZE PAGE (Form + Processing)
# ============================================================================

def analyze(request):
    """
    Handles the analysis form.
    
    GET request  → Show empty form
    POST request → Process form data, run analysis, save to database, redirect to result
    """
    if request.method == 'POST':
        form = NewsInputForm(request.POST)
        
        if form.is_valid():
            # Step 1: Get cleaned data from form
            # The form has a single textarea, so we split it: the first line is
            # the headline (what the database fact-check searches on) and
            # anything after it is the article body.
            content = form.cleaned_data['news_content'].strip()
            headline, _, body = content.partition('\n')
            headline = headline.strip()
            article_text = body.strip()

            # Step 2: Run the analysis engine
            result = analyze_text(headline, article_text)

            # Step 3: Save result to database
            # A long single-line paste is an article body, not a headline — keep
            # the full text so nothing is lost when the headline is truncated.
            if not article_text and len(headline) > 500:
                article_text = headline

            display_headline = headline if len(headline) <= 500 else headline[:497] + "..."

            # If Step 1 found a known article, copy across what it found so the
            # result page can show it as evidence
            matched = result.get('matched_article')

            analysis = NewsAnalysis.objects.create(
                headline=display_headline,
                article_text=article_text,
                score=result['score'],
                result_label=result['label'],
                explanation=result['explanation'],
                confidence=result.get('confidence'),
                method=result.get('method'),
                matched_headline=matched.headline if matched else '',
                matched_extract=matched.article_text if matched else '',
                # None when the search did not run at all, which the result page
                # renders differently from "ran and found nothing".
                web_check=result.get('web'),
            )

            # Step 4: Redirect to result page
            return redirect('result', analysis_id=analysis.id)
    else:
        form = NewsInputForm()
    
    return render(request, 'analyzer/analyze.html', {'form': form})


# ============================================================================
# VIEW 3: RESULT PAGE
# ============================================================================

def result(request, analysis_id):
    """
    Shows the result of a specific analysis.

    Retrieves the analysis from database using its ID.
    """
    # get_object_or_404: returns the object or shows 404 error if not found
    analysis = get_object_or_404(NewsAnalysis, id=analysis_id)

    # Colour and icon per verdict. 'Not Sure' is kept for rows saved by an
    # earlier version of the engine — the reading model no longer produces it,
    # but old dashboard entries still carry it and must still render.
    if analysis.result_label == 'Likely Real':
        result_class = 'result-real'
        result_icon = '✅'
    elif analysis.result_label == 'Not Sure':
        result_class = 'result-unsure'
        result_icon = '🤷'
    else:
        result_class = 'result-fake'
        result_icon = '❌'

    # What the percentage next to the verdict actually means. For the reading
    # model it is how often a verdict like this is CORRECT, not how sure the
    # model feels — so it must never be labelled "confidence".
    if analysis.method == 'reading_model':
        confidence_label = 'of these verdicts are correct'
    elif analysis.method == 'database':
        confidence_label = 'headline match'
    else:
        confidence_label = 'Confidence'

    return render(request, 'analyzer/result.html', {
        'analysis': analysis,
        'result_class': result_class,
        'result_icon': result_icon,
        'confidence': analysis.confidence,
        'confidence_label': confidence_label,
        'method': analysis.method,
    })


# ============================================================================
# VIEW 4: DASHBOARD PAGE
# ============================================================================

def dashboard(request):
    """
    Analytics dashboard showing:
    - Summary statistics
    - Recent analyses table
    - Charts are loaded via JavaScript using the dashboard_data API
    """
    # Get all analyses
    analyses = NewsAnalysis.objects.all()
    total = analyses.count()

    # Count by category. 'Not Sure' is counted explicitly rather than lumped in
    # with anything else — since Step 2 stopped guessing it is the most common
    # outcome, and a dashboard whose three numbers do not add up to the total
    # would just look broken.
    real_count = analyses.filter(result_label='Likely Real').count()
    fake_count = analyses.filter(result_label='Likely Fake').count()
    unsure_count = analyses.filter(result_label='Not Sure').count()

    # Calculate percentages (avoid division by zero)
    real_pct = round((real_count / total * 100), 1) if total > 0 else 0
    fake_pct = round((fake_count / total * 100), 1) if total > 0 else 0
    unsure_pct = round((unsure_count / total * 100), 1) if total > 0 else 0

    # Get last 20 analyses for the table
    recent = analyses[:20]

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


# ============================================================================
# VIEW 5: DASHBOARD DATA API (for Chart.js)
# ============================================================================

def dashboard_data(request):
    """
    Returns JSON data for the dashboard charts.

    Chart.js (running in the browser) calls this URL via JavaScript fetch(),
    receives JSON data, and renders the charts.

    Returns:
        - pie_data: counts for Real/Fake/Not Sure (for the doughnut chart)
        - method_data: how many verdicts came from the database vs. each model
    """
    analyses = NewsAnalysis.objects.all()

    # --- Pie Chart Data ---
    pie_data = {
        'labels': ['Likely Real', 'Likely Fake', 'Not Sure'],
        'counts': [
            analyses.filter(result_label='Likely Real').count(),
            analyses.filter(result_label='Likely Fake').count(),
            analyses.filter(result_label='Not Sure').count(),
        ],
    }

    # --- Bar Chart Data: how each verdict was reached ---
    # 'Word-Counter' only appears when the reading model could not be loaded, so
    # a non-zero bar there is a useful warning that the app is running on the
    # old fallback rather than the model we measured.
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
