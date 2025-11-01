# app.py
import streamlit as st
import joblib
from pathlib import Path
import re

# Load model and vectorizer
MODEL_DIR = Path("models")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
clf = joblib.load(MODEL_DIR / "tfidf_clf.pkl")

# Function to clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.set_page_config(page_title="🎬 Movie Sentiment Analyzer", layout="centered")

st.title("🎬 Movie Sentiment Analysis")
st.write("Enter a movie review below to analyze its sentiment!")

user_input = st.text_area("💬 Enter review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])
        prediction = clf.predict(features)[0]
        sentiment = "Positive 😀" if prediction == 1 else "Negative 😞"

        st.subheader("Result:")
        st.success(sentiment)
    else:
        st.warning("Please enter some text to analyze.")
