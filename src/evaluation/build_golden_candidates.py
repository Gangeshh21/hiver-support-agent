import pandas as pd

CASES_PATH = "data/processed/apple_support_cases.csv"
REVIEW_PATH = "data/golden/intent_review_100.csv"
OUTPUT_PATH = "data/golden/golden_candidates_100.csv"

cases = pd.read_csv(CASES_PATH, dtype=str, keep_default_na=False)
reviewed = pd.read_csv(REVIEW_PATH, dtype=str, keep_default_na=False)

# Remove the 100 cases already used for taxonomy validation
used_ids = set(reviewed["case_id"])
candidates = cases[~cases["case_id"].isin(used_ids)].copy()

# Reproducible random sample
sample = candidates.sample(n=100, random_state=42)

golden = sample[
    [
        "case_id",
        "initial_customer_message",
        "final_apple_reply",
        "conversation",
    ]
].copy()

golden["intent"] = ""
golden["label_notes"] = ""
golden["review_status"] = ""

golden.to_csv(OUTPUT_PATH, index=False)

print("Created:", OUTPUT_PATH)
print("New candidates:", len(golden))
print("\nSample:")
print(
    golden[
        ["case_id", "initial_customer_message"]
    ].head(10).to_string(index=False)
)
