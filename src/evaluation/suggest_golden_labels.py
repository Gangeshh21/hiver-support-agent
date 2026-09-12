import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


REVIEW_PATH = "data/golden/intent_review_100.csv"
CANDIDATE_PATH = "data/golden/golden_candidates_100.csv"


def main():
    reviewed = pd.read_csv(REVIEW_PATH, dtype=str, keep_default_na=False)
    candidates = pd.read_csv(CANDIDATE_PATH, dtype=str, keep_default_na=False)

    X_train = reviewed["initial_customer_message"].fillna("")
    y_train = reviewed["intent"]

    X_new = candidates["initial_customer_message"].fillna("")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_new_tfidf = vectorizer.transform(X_new)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_new_tfidf)
    probabilities = model.predict_proba(X_new_tfidf)

    candidates["suggested_intent"] = predictions
    candidates["suggested_confidence"] = probabilities.max(axis=1)

    candidates.to_csv(CANDIDATE_PATH, index=False)

    print("Generated suggestions for:", len(candidates))
    print("\nSuggested intent distribution:")
    print(candidates["suggested_intent"].value_counts())

    print("\nConfidence:")
    print(candidates["suggested_confidence"].describe())


if __name__ == "__main__":
    main()
