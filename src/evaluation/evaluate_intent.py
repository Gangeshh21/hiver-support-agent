import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


INPUT_PATH = "evaluation/golden_set.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    X = df["initial_customer_message"].fillna("")
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    classifier.fit(X_train_vec, y_train)

    predictions = classifier.predict(X_test_vec)

    print("=" * 60)
    print("INTENT CLASSIFIER EVALUATION")
    print("=" * 60)

    print(f"Total examples : {len(df)}")
    print(f"Train examples : {len(X_train)}")
    print(f"Test examples  : {len(X_test)}")

    print(f"Accuracy       : {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Macro Precision: "
        f"{precision_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Macro Recall   : "
        f"{recall_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Macro F1       : "
        f"{f1_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )

    print("\nPER-INTENT RESULTS")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("CONFUSION MATRIX")
    print("-" * 60)

    labels = sorted(y.unique())
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    print(matrix_df.to_string())


if __name__ == "__main__":
    main()
