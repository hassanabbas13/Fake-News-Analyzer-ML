"""URL routing for the analyzer app."""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze/', views.analyze, name='analyze'),
    path('result/<int:analysis_id>/', views.result, name='result'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('api/dashboard-data/', views.dashboard_data, name='dashboard_data'),

    # POST-only, and history only. See views.py.
    path('delete/<int:analysis_id>/', views.delete_analysis, name='delete_analysis'),
    path('clear/', views.clear_analyses, name='clear_analyses'),
]
