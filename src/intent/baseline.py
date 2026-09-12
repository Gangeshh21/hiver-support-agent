import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = "data/golden/intent_review_100.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df["initial_customer_message"].fillna("")
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    classifier.fit(X_train_tfidf, y_train)

    predictions = classifier.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    print("=== Simple TF-IDF + Logistic Regression Baseline ===")
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy (%): {accuracy * 100:.2f}%")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()
