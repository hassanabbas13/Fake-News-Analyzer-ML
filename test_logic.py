import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fake_news_project.settings')
django.setup()

from analyzer.analysis_engine import analyze_text

print('\n--- TEST 1: DATABASE MATCH (Known Fake) ---')
res1 = analyze_text("Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing", "")
print(f"Label: {res1['label']}")
print(f"Confidence: {res1['confidence']}%")
print(f"Method: {res1['method']}")

print('\n--- TEST 2: AI PREDICTION (Unknown Text) ---')
res2 = analyze_text("Scientists discover that eating pizza every day cures cancer", "A new study from an unknown university claims that pizza is the ultimate cure.")
print(f"Label: {res2['label']}")
print(f"Confidence: {res2['confidence']}%")
print(f"Method: {res2['method']}")
