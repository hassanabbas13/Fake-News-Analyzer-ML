"""
Add KnownArticle.body_key and fill it in for every row already loaded.

The backfill matters as much as the column. An empty body_key on 59,660 existing
rows would mean the new lookup silently finds nothing — no crash, no warning,
just a feature that appears to be broken. Anyone running this on an already
loaded database needs the column populated, not merely created.

Batched, because SQLite will not take 59,660 rows in one statement.
"""

from django.db import migrations, models

# The backfill deliberately imports the real key function rather than copying
# its logic. A migration frozen against a stale copy of the rule would produce
# keys that the running app disagrees with — which is the exact silent-miss
# failure this project already hit once with headline_key.
from analyzer.text_matching import body_key

BATCH = 2000


def fill_body_keys(apps, schema_editor):
    KnownArticle = apps.get_model('analyzer', 'KnownArticle')

    pending = []
    # .iterator() so 59,660 rows are streamed rather than all held in memory.
    for row in KnownArticle.objects.exclude(article_text='').only('id', 'article_text').iterator():
        key = body_key(row.article_text)
        if not key:
            continue
        row.body_key = key
        pending.append(row)

        if len(pending) >= BATCH:
            KnownArticle.objects.bulk_update(pending, ['body_key'])
            pending.clear()

    if pending:
        KnownArticle.objects.bulk_update(pending, ['body_key'])


def clear_body_keys(apps, schema_editor):
    """
    Reverse step. Blanking the column is not strictly needed before dropping it,
    but a reversible migration that silently does nothing on the way back is
    worse than one that states what it undoes.
    """
    KnownArticle = apps.get_model('analyzer', 'KnownArticle')
    KnownArticle.objects.exclude(body_key='').update(body_key='')


class Migration(migrations.Migration):

    dependencies = [
        ('analyzer', '0007_newsanalysis_web_check'),
    ]

    operations = [
        migrations.AddField(
            model_name='knownarticle',
            name='body_key',
            field=models.CharField(blank=True, db_index=True, default='', max_length=400),
        ),
        migrations.RunPython(fill_body_keys, clear_body_keys),
    ]
