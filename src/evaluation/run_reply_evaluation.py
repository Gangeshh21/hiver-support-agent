import os
import time
import pandas as pd

from src.pipeline import SupportAgent
from src.evaluation.judge import judge_reply

INPUT_PATH = "src/evaluation/reply_eval_cases.csv"
OUTPUT_PATH = "evaluation/results/reply_quality_results.csv"


def main():
    os.makedirs("evaluation/results", exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    agent = SupportAgent()

    # Resume from cases not already evaluated.
    if os.path.exists(OUTPUT_PATH):
        previous = pd.read_csv(OUTPUT_PATH)
        completed_ids = set(previous["case_id"].astype(str))
        df = df[~df["case_id"].astype(str).isin(completed_ids)]
        results = previous.to_dict("records")
        print(f"Already evaluated: {len(completed_ids)} cases.")
        print(f"Remaining cases: {len(df)}")
    else:
        results = []

    for _, row in df.iterrows():
        print(f"\nEvaluating case {row['case_id']}...")

        customer_message = row["customer_message"]

        try:
            output = agent.run(customer_message)

            judge = judge_reply(
                customer_message,
                output["reply"],
                output["evidence"],
            )

            results.append({
                "case_id": row["case_id"],
                "customer_message": customer_message,
                "intent": output["intent"],
                "confidence": output["confidence"],
                "retrieval_similarity": output["evidence"][0]["similarity"],
                "action": output["action"],
                "reply": output["reply"],
                "correctness": judge["correctness"],
                "grounding": judge["grounding"],
                "relevance": judge["relevance"],
                "helpfulness": judge["helpfulness"],
                "no_unsupported_claims": judge["no_unsupported_claims"],
                "overall": judge["overall"],
                "judge_reason": judge["reason"],
            })

            # Save after EVERY successful case.
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print(f"Saved {len(results)} successful evaluations.")

            # Small delay to reduce quota pressure.
            time.sleep(15)

        except Exception as e:
            error_text = str(e)

            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                print("\nGemini quota exhausted.")
                print("Stopping safely and preserving successful evaluations.")
                break

            print(f"Case {row['case_id']} failed: {e}")

    if results:
        results_df = pd.DataFrame(results)

        print("\n" + "=" * 60)
        print("REPLY QUALITY EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Cases successfully evaluated: {len(results_df)}")

        print("\nAverage scores:")
        for column in [
            "correctness",
            "grounding",
            "relevance",
            "helpfulness",
            "no_unsupported_claims",
            "overall",
        ]:
            print(f"{column}: {results_df[column].mean():.2f}/5")

        print(f"\nSaved to: {OUTPUT_PATH}")
    else:
        print("\nNo LLM-judge evaluations were completed.")
        print("Gemini quota is currently exhausted.")


if __name__ == "__main__":
    main()
