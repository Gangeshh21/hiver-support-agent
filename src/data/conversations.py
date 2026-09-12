from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/processed/apple_support_conversations.csv")

BRAND = "AppleSupport"
CHUNK_SIZE = 50_000


def main():
    print("=" * 60)
    print("APPLE SUPPORT - CONVERSATION EXTRACTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: Find all AppleSupport tweets
    # ---------------------------------------------------------
    print("\n[1/3] Finding AppleSupport tweets...")

    apple_tweets = {}

    for chunk_no, chunk in enumerate(
        pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
        start=1
    ):
        brand_rows = chunk[chunk["author_id"] == BRAND]

        for _, row in brand_rows.iterrows():
            tweet_id = str(int(row["tweet_id"]))

            apple_tweets[tweet_id] = {
                "tweet_id": tweet_id,
                "author_id": str(row["author_id"]),
                "inbound": bool(row["inbound"]),
                "created_at": row["created_at"],
                "text": str(row["text"]),
                "response_tweet_id": row["response_tweet_id"],
                "in_response_to_tweet_id": row["in_response_to_tweet_id"],
            }

        if chunk_no % 10 == 0:
            print(
                f"  Processed chunks: {chunk_no} | "
                f"AppleSupport tweets: {len(apple_tweets)}"
            )

    print(f"\nFound {len(apple_tweets):,} AppleSupport tweets.")

    # ---------------------------------------------------------
    # STEP 2: Collect customer tweets directly replied to
    # ---------------------------------------------------------
    print("\n[2/3] Finding customer tweets replied to by AppleSupport...")

    parent_ids = set()

    for tweet in apple_tweets.values():
        parent_id = tweet["in_response_to_tweet_id"]

        if pd.notna(parent_id):
            parent_ids.add(str(int(float(parent_id))))

    print(f"Customer/parent tweet IDs needed: {len(parent_ids):,}")

    # ---------------------------------------------------------
    # STEP 3: Extract those tweets from the original dataset
    # ---------------------------------------------------------
    print("\n[3/3] Extracting conversation tweets...")

    needed_ids = set(apple_tweets.keys()) | parent_ids
    conversations = {}

    for chunk_no, chunk in enumerate(
        pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
        start=1
    ):
        chunk["tweet_id_str"] = chunk["tweet_id"].astype("int64").astype(str)

        matched = chunk[chunk["tweet_id_str"].isin(needed_ids)]

        for _, row in matched.iterrows():
            tweet_id = str(int(row["tweet_id"]))

            conversations[tweet_id] = {
                "tweet_id": tweet_id,
                "author_id": str(row["author_id"]),
                "inbound": bool(row["inbound"]),
                "created_at": row["created_at"],
                "text": str(row["text"]),
                "response_tweet_id": row["response_tweet_id"],
                "in_response_to_tweet_id": row[
                    "in_response_to_tweet_id"
                ],
            }

        if chunk_no % 10 == 0:
            print(
                f"  Processed chunks: {chunk_no} | "
                f"Tweets collected: {len(conversations):,}"
            )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    output_df = pd.DataFrame(conversations.values())

    output_df["created_at"] = pd.to_datetime(
        output_df["created_at"],
        errors="coerce"
    )

    output_df = output_df.sort_values("created_at")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"\nSaved to:")
    print(f"  {OUTPUT_PATH}")

    print(f"\nTotal tweets extracted:")
    print(f"  {len(output_df):,}")

    print("\nSpeaker distribution:")
    print(output_df["inbound"].value_counts().to_string())


if __name__ == "__main__":
    main()