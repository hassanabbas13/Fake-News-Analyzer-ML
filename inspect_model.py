import pickle
import numpy as np
import os

def inspect():
    with open('analyzer/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('analyzer/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients
    # Logistic regression has a single coefficient array for binary classification
    coefs = model.coef_[0]
    
    # Sort coefficients
    # Positive coefficients push towards class 1 (FAKE)
    # Negative coefficients push towards class 0 (REAL)
    sorted_indices = np.argsort(coefs)
    
    # Top 20 REAL words
    top_real_indices = sorted_indices[:20]
    top_real_words = [(feature_names[i], coefs[i]) for i in top_real_indices]
    
    # Top 20 FAKE words
    top_fake_indices = sorted_indices[-20:][::-1]
    top_fake_words = [(feature_names[i], coefs[i]) for i in top_fake_indices]
    
    print("--- TOP INDICATORS OF REAL NEWS ---")
    for word, weight in top_real_words:
        print(f"{word}: {weight:.4f}")
        
    print("\n--- TOP INDICATORS OF FAKE NEWS ---")
    for word, weight in top_fake_words:
        print(f"{word}: {weight:.4f}")

if __name__ == '__main__':
    inspect()
