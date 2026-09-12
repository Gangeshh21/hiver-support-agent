import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/processed/apple_support_cases.csv"


class Retriever:

    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)

        texts = self.df["initial_customer_message"].fillna("")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000
        )

        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=3):
        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []

        for index in top_indices:
            row = self.df.iloc[index]

            results.append({
                "case_id": row["case_id"],
                "customer_message": row["initial_customer_message"],
                "historical_reply": row["final_apple_reply"],
                "similarity": float(scores[index])
            })

        return results


if __name__ == "__main__":

    retriever = Retriever()

    query = "My iPhone battery is draining very quickly after the latest iOS update."

    results = retriever.search(query, top_k=3)

    print("\n=== Retrieval Results ===")

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Case ID: {result['case_id']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Customer: {result['customer_message']}")
        print(f"AppleSupport: {result['historical_reply']}")
