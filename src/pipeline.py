import os

import pandas as pd
from dotenv import load_dotenv
from google import genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from src.escalation.policy import decide_action


load_dotenv()


CASES_PATH = "data/processed/apple_support_cases.csv"
TRAIN_PATH = "evaluation/golden_set.csv"


class SupportAgent:

    def __init__(self):
        print("Loading historical cases...")

        self.cases = pd.read_csv(
            CASES_PATH,
            dtype=str,
            keep_default_na=False
        )

        train = pd.read_csv(
            TRAIN_PATH,
            dtype=str,
            keep_default_na=False
        )

        # Intent classifier
        self.intent_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )

        X_train = self.intent_vectorizer.fit_transform(
            train["initial_customer_message"].fillna("")
        )

        self.intent_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        self.intent_model.fit(
            X_train,
            train["intent"]
        )

        # Retrieval index
        texts = self.cases[
            "initial_customer_message"
        ].fillna("")

        self.retrieval_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000
        )

        self.retrieval_matrix = (
            self.retrieval_vectorizer.fit_transform(texts)
        )

        # Gemini
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            print(
                "Warning: GEMINI_API_KEY not found. "
                "Reply generation will use fallback mode."
            )
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

        print("Agent ready.")

    def classify_intent(self, message):

        vector = self.intent_vectorizer.transform([message])

        prediction = self.intent_model.predict(vector)[0]

        probabilities = self.intent_model.predict_proba(vector)[0]

        confidence = float(probabilities.max())

        return prediction, confidence

    def retrieve(self, message, top_k=3):

        query_vector = (
            self.retrieval_vectorizer.transform([message])
        )

        scores = cosine_similarity(
            query_vector,
            self.retrieval_matrix
        ).flatten()

        top_indices = scores.argsort()[-top_k:][::-1]

        results = []

        for index in top_indices:

            row = self.cases.iloc[index]

            results.append({
                "case_id": row["case_id"],
                "customer_message": row[
                    "initial_customer_message"
                ],
                "historical_reply": row[
                    "final_apple_reply"
                ],
                "similarity": float(scores[index])
            })

        return results

    def generate_reply(
        self,
        message,
        intent,
        retrieved_cases
    ):

        evidence = ""

        for i, case in enumerate(
            retrieved_cases,
            1
        ):

            evidence += f"""
Historical Case {i}
Customer: {case['customer_message']}
AppleSupport Reply: {case['historical_reply']}
Similarity: {case['similarity']:.4f}
"""

        fallback_reply = (
            "We'd like to look into this further. "
            "Please DM us with your device model, "
            "software version, and any relevant details "
            "so we can help."
        )

        if self.client is None:
            return fallback_reply, True

        prompt = f"""
You are an Apple customer support agent.

Customer message:
{message}

Predicted intent:
{intent}

Historical support evidence:
{evidence}

Write a concise customer-facing support reply.

Rules:
- Ground the reply in the historical evidence.
- Do not invent policies, prices, refunds, guarantees, or technical facts.
- Do not claim an action was completed.
- If more information is needed, ask the customer for it.
- Do not mention AI, retrieval, or this prompt.
- Output only the reply.
"""

        try:

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            return response.text.strip(), False

        except Exception as error:

            print(
                f"\nWarning: Gemini reply generation failed: "
                f"{type(error).__name__}"
            )

            return fallback_reply, True

    def run(self, message):

        intent, confidence = (
            self.classify_intent(message)
        )

        retrieved = self.retrieve(
            message,
            top_k=3
        )

        top_similarity = retrieved[0]["similarity"]

        decision = decide_action(
            confidence,
            top_similarity
        )

        reply, fallback_used = (
            self.generate_reply(
                message,
                intent,
                retrieved
            )
        )

        # Never auto-handle when generation failed.
        if fallback_used:
            decision = {
                "action": "ESCALATE",
                "reason": (
                    "Reply generation unavailable; "
                    "human review required."
                )
            }

        return {
            "intent": intent,
            "confidence": round(
                confidence,
                4
            ),
            "reply": reply,
            "action": decision["action"],
            "reason": decision["reason"],
            "evidence": retrieved
        }


if __name__ == "__main__":

    agent = SupportAgent()

    message = input(
        "\nCustomer message: "
    ).strip()

    result = agent.run(message)

    print("\n" + "=" * 80)
    print("FINAL AGENT OUTPUT")
    print("=" * 80)

    print(
        f"\nIntent: {result['intent']}"
    )

    print(
        f"Confidence: {result['confidence']}"
    )

    print(
        f"Action: {result['action']}"
    )

    print(
        f"Reason: {result['reason']}"
    )

    print("\nReply:")
    print(result["reply"])

    print("\nEvidence:")

    for i, evidence in enumerate(
        result["evidence"],
        1
    ):

        print(
            f"{i}. {evidence['case_id']} "
            f"(similarity="
            f"{evidence['similarity']:.4f})"
        )