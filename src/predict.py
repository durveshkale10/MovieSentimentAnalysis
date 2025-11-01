# src/predict.py
import joblib
from pathlib import Path

MODEL_DIR = Path("models")

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
clf = joblib.load(MODEL_DIR / "tfidf_clf.pkl")

def predict_sentiment(text: str) -> str:
    text_tfidf = vectorizer.transform([text])
    prediction = clf.predict(text_tfidf)[0]
    return "Positive 😀" if prediction == 1 else "Negative 😞"

if __name__ == "__main__":
    sample = input("Enter a review: ")
    print(predict_sentiment(sample))
