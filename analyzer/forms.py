"""
The one input form: a textarea for the article, plus an opt-in web search.

Validation matters here because the model cannot say "I cannot read this" — Urdu,
emoji and keyboard mash all come out as a confident "Likely Fake" beside a
reliability figure measured on English news. See input_check.py for the rules.
"""

from django import forms

from .input_check import MAX_WORDS, MIN_WORDS, check_input
from .web_check import has_key


class NewsInputForm(forms.Form):
    """Headline on the first line, article body underneath. Split in views.py."""

    news_content = forms.CharField(
        label='News or Article Text',
        required=True,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'placeholder': (f'Paste an English news article here — '
                            f'{MIN_WORDS} to {MAX_WORDS} words...'),
            'rows': 6,
            'id': 'content-input',
            # Deliberately generous: this counts characters and the real rule is
            # words, so the server still has the final say.
            'maxlength': MAX_WORDS * 12,
            # How the limits reach the live counter in analyze.html. One source of
            # truth for the numbers — input_check.py — so the counter cannot drift
            # out of step with the rule the server enforces.
            'data-min-words': MIN_WORDS,
            'data-max-words': MAX_WORDS,
        }),
        # No help_text on purpose: the placeholder above the box and the counter
        # beside it already say the same thing.
    )

    # An ADDITION to the automatic search, not a replacement. should_check() still
    # searches on its own inside the unsure band, ticked or not, because the reader
    # never sees the raw score and cannot tell which articles are guesses. This box
    # only widens the net to the confident ones. Off by default: the free
    # allowance is 500 grounded searches a day.
    search_web = forms.BooleanField(
        label='Also check the web',
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'field-check-box',
            'id': 'search-web',
        }),
        help_text=('Looks the story up online even when the model is sure. '
                   'Skipped when the article is already in our dataset.'),
    )

    def __init__(self, *args, **kwargs):
        """Drop the checkbox when there is no API key, rather than offering a box
        that could only ever come back 'unavailable'. has_key() is a file and
        environment read, no network."""
        super().__init__(*args, **kwargs)
        if not has_key():
            del self.fields['search_web']

    def wants_web_search(self):
        """True only if the box is present, ticked and the form validated, so the
        view never has to know the field can be absent."""
        return bool(self.cleaned_data.get('search_web'))

    def clean_news_content(self):
        """Refuse anything the model cannot honestly judge. A verdict on unreadable
        input is worse than refusing, because the page quotes a reliability figure
        beside it that was measured on English news only."""
        content = self.cleaned_data['news_content'].strip()

        problem = check_input(content)
        if problem:
            raise forms.ValidationError(problem.message, code=problem.code)

        return content
