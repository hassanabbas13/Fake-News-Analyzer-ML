import os
import django
import json
import pandas as pd
import pickle
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fake_news_project.settings')
django.setup()

def run():
    print("Starting Mega-Model Training...")
    
    # 1. Load all three CSV files
    print("Reading CSV files...")
    df_fake = pd.read_csv('Fake.csv')
    df_fake['label'] = 'FAKE'
    
    df_true = pd.read_csv('True.csv')
    df_true['label'] = 'REAL'
    
    df_mixed = pd.read_csv('fake_or_real_news.csv')
    
    # Standardize columns
    df_fake = df_fake[['title', 'text', 'label']]
    df_true = df_true[['title', 'text', 'label']]
    df_mixed = df_mixed[['title', 'text', 'label']]
    
    # 2. Combine into one mega-dataset
    print("Combining datasets...")
    df = pd.concat([df_fake, df_true, df_mixed], ignore_index=True)
    
    # Drop rows with missing text
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    df['title'] = df['title'].astype(str)
    
    # Combine title + text for richer training data
    df['full_text'] = df['title'] + ' ' + df['text']
    
    # Convert labels to binary: FAKE = 1, REAL = 0
    df['label_binary'] = df['label'].apply(lambda x: 1 if x.upper() == 'FAKE' else 0)
    
    print(f"Total training samples: {len(df)}")
    print(f"  FAKE articles: {(df['label_binary'] == 1).sum()}")
    print(f"  REAL articles: {(df['label_binary'] == 0).sum()}")
    
    # 3. Split into training (80%) and testing (20%)
    X = df['full_text']
    y = df['label_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} articles")
    print(f"Testing set:  {len(X_test)} articles")
    
    # 4. Convert text to TF-IDF vectors
    print("\nConverting text to TF-IDF vectors (this may take a moment)...")
    vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} unique words")
    
    # 5. Train the Logistic Regression model
    print("\nTraining the AI model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # 6. Test the model and print accuracy
    print("\nTesting model accuracy...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"  MODEL ACCURACY: {accuracy * 100:.2f}%")
    print(f"{'='*50}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    
    # 7. Save the trained model and vectorizer
    model_path = os.path.join('analyzer', 'model.pkl')
    vectorizer_path = os.path.join('analyzer', 'vectorizer.pkl')
    meta_path = os.path.join('analyzer', 'model_meta.json')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to: {vectorizer_path}")

    # 8. Record what we just measured, next to the model it describes.
    # analysis_engine.py reads this instead of having numbers typed into it, so
    # every retrain updates what the app reports to users.
    metadata = {
        'test_accuracy': round(accuracy * 100, 2),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'total_samples': len(df),
        'vocabulary_size': len(vectorizer.vocabulary_),
        'trained_at': date.today().isoformat(),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Measurements saved to: {meta_path}")

    print("\nSUCCESS! AI model has been trained and saved.")
    print("You can now use the Hybrid System in Django.")

if __name__ == '__main__':
    run()
