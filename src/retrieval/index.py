import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "data/processed/apple_support_cases.csv"

def build_index():
    df = pd.read_csv(DATA_PATH)

    texts = df["initial_customer_message"].fillna("")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000
    )

    matrix = vectorizer.fit_transform(texts)

    print("=== Retrieval Index ===")
    print(f"Cases indexed: {len(df)}")
    print(f"TF-IDF matrix shape: {matrix.shape}")

    return df, vectorizer, matrix

if __name__ == "__main__":
    build_index()
