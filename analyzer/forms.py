"""
forms.py — Django Form for User Input
=======================================

Django forms handle:
1. Rendering HTML form fields
2. Validating user input
3. Cleaning/sanitizing data

We have ONE form: NewsInputForm
"""

from django import forms


class NewsInputForm(forms.Form):
    """
    Form for users to enter a news headline or full article text.
    """
    news_content = forms.CharField(
        label='News or Article Text',
        required=True,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': 'Paste your news headline or full article text here...',
            'rows': 6,
            'id': 'content-input',
        })
    )
