"""
forms.py — Django Form for User Input
=======================================

Django forms handle:
1. Rendering HTML form fields
2. Validating user input
3. Cleaning/sanitizing data

We have ONE form: NewsInputForm

Validation matters more here than in most forms. The reading model cannot say
"I cannot read this" — it was only ever taught to tell English fake news from
English real news, so Urdu, Spanish, emoji and keyboard mash all come out as a
confident "Likely Fake". Worse, the result page would quote the measured 86.9%
reliability alongside it, which was measured on English news and means nothing
here. Stopping bad input at the form is what keeps the app from inventing a
statistic. See analyzer/input_check.py for the measurements behind each rule.
"""

from django import forms

from .input_check import MAX_WORDS, MIN_WORDS, check_input


class NewsInputForm(forms.Form):
    """
    Form for users to enter a news headline or full article text.
    """
    news_content = forms.CharField(
        label='News or Article Text',
        required=True,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': (f'Paste an English news article here — '
                            f'{MIN_WORDS} to {MAX_WORDS} words...'),
            'rows': 6,
            'id': 'content-input',
            # The browser's own counter, so someone pasting a long article finds
            # out before they submit rather than after. Generous by design: this
            # counts characters, and the real rule is words, so the server still
            # has the final say.
            'maxlength': MAX_WORDS * 12,
        }),
        help_text=(f'English only, between {MIN_WORDS} and {MAX_WORDS} words. '
                   f'The model reads roughly the first 200 words.'),
    )

    def clean_news_content(self):
        """
        Refuse anything the model cannot honestly judge.

        Returning a verdict on unreadable input would be worse than refusing:
        the page states how often that verdict is right, and that figure was
        measured on English news articles only.
        """
        content = self.cleaned_data['news_content'].strip()

        problem = check_input(content)
        if problem:
            raise forms.ValidationError(problem.message, code=problem.code)

        return content
