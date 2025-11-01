import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/aclImdb")

def load_data(folder):
    data = []
    for label in ["pos", "neg"]:
        labeled_dir = folder / label
        print(f"📂 Reading {label} reviews from {labeled_dir}")
        txt_files = list(labeled_dir.glob("*.txt"))
        print(f"Found {len(txt_files)} {label} reviews.")
        for file in txt_files:
            try:
                with open(file, encoding="utf-8") as f:
                    review = f.read().strip()
                    data.append({
                        "review": review,
                        "sentiment": 1 if label == "pos" else 0
                    })
            except Exception as e:
                print(f"⚠️ Error reading {file.name}: {e}")
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("🚀 Preparing IMDB dataset...")

    train_df = load_data(DATA_DIR / "train")
    test_df = load_data(DATA_DIR / "test")

    print(f"✅ Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("💾 Saved: data/train.csv and data/test.csv")
