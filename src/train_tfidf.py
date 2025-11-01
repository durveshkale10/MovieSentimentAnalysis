# src/train_tfidf.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("📘 Loading training and test data...")
train_df = pd.read_csv(DATA_DIR / "clean_train.csv")
test_df = pd.read_csv(DATA_DIR / "clean_test.csv")

print("🔠 Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_df["review"])
y_train = train_df["sentiment"]

X_test = vectorizer.transform(test_df["review"])
y_test = test_df["sentiment"]

print("🤖 Training Logistic Regression classifier...")
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

print("✅ Evaluating model...")
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"🎯 Accuracy: {acc * 100:.2f}%")

print("💾 Saving vectorizer and model to 'models/'...")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
joblib.dump(clf, MODEL_DIR / "tfidf_clf.pkl")

print("✅ Model and vectorizer saved successfully!")
