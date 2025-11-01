# src/preprocess.py

import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path("data")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("🧹 Cleaning text data...")

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

train_df["review"] = train_df["review"].apply(clean_text)
test_df["review"] = test_df["review"].apply(clean_text)

train_df.to_csv(DATA_DIR / "clean_train.csv", index=False)
test_df.to_csv(DATA_DIR / "clean_test.csv", index=False)

print("✅ Cleaned files saved as:")
print(" - data/clean_train.csv")
print(" - data/clean_test.csv")
