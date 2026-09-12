from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/processed/apple_support_threads.csv")

BRAND = "AppleSupport"
CHUNK_SIZE = 50_000


def clean_id(value):
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return None


def main():
    print("=" * 60)
    print("APPLE SUPPORT - FULL THREAD EXTRACTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # PASS 1
    # Find every AppleSupport tweet and its parent
    # ---------------------------------------------------------
    print("\n[1/3] Finding AppleSupport tweets...")

    apple_ids = set()
    parent_ids = set()

    for chunk_no, chunk in enumerate(
        pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
        start=1
    ):
        brand_rows = chunk[chunk["author_id"] == BRAND]

        for _, row in brand_rows.iterrows():
            tweet_id = clean_id(row["tweet_id"])
            parent_id = clean_id(row["in_response_to_tweet_id"])

            if tweet_id:
                apple_ids.add(tweet_id)

            if parent_id:
                parent_ids.add(parent_id)

        if chunk_no % 10 == 0:
            print(
                f"  Chunks: {chunk_no} | "
                f"AppleSupport: {len(apple_ids):,}"
            )

    print(f"\nAppleSupport tweets: {len(apple_ids):,}")
    print(f"Direct parent tweets: {len(parent_ids):,}")

    # ---------------------------------------------------------
    # PASS 2
    # Extract AppleSupport + direct parents
    # ---------------------------------------------------------
    print("\n[2/3] Extracting linked tweets...")

    needed_ids = apple_ids | parent_ids
    tweets = {}

    for chunk_no, chunk in enumerate(
        pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
        start=1
    ):
        chunk["tweet_id_str"] = (
            pd.to_numeric(chunk["tweet_id"], errors="coerce")
            .astype("Int64")
            .astype(str)
        )

        matched = chunk[
            chunk["tweet_id_str"].isin(needed_ids)
        ]

        for _, row in matched.iterrows():
            tweet_id = clean_id(row["tweet_id"])

            if tweet_id:
                tweets[tweet_id] = {
                    "tweet_id": tweet_id,
                    "author_id": str(row["author_id"]),
                    "inbound": bool(row["inbound"]),
                    "created_at": row["created_at"],
                    "text": str(row["text"]),
                    "response_tweet_id": row[
                        "response_tweet_id"
                    ],
                    "in_response_to_tweet_id": row[
                        "in_response_to_tweet_id"
                    ],
                }

        if chunk_no % 10 == 0:
            print(
                f"  Chunks: {chunk_no} | "
                f"Tweets collected: {len(tweets):,}"
            )

    # ---------------------------------------------------------
    # PASS 3
    # Expand backwards through parent relationships
    # ---------------------------------------------------------
    print("\n[3/3] Expanding conversation threads...")

    all_ids = set(tweets.keys())

    # We repeatedly search for parents of already discovered tweets.
    # Usually only a small number of iterations are required.
    for iteration in range(1, 6):
        new_parent_ids = set()

        for tweet_id in list(all_ids):
            tweet = tweets.get(tweet_id)

            if tweet is None:
                continue

            parent_id = clean_id(
                tweet["in_response_to_tweet_id"]
            )

            if parent_id and parent_id not in all_ids:
                new_parent_ids.add(parent_id)

        if not new_parent_ids:
            print(
                f"  No new parents found at iteration {iteration}."
            )
            break

        print(
            f"  Iteration {iteration}: "
            f"{len(new_parent_ids):,} new parent IDs"
        )

        # Scan original dataset for newly discovered parents.
        for chunk in pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE
        ):
            chunk["tweet_id_str"] = (
                pd.to_numeric(chunk["tweet_id"], errors="coerce")
                .astype("Int64")
                .astype(str)
            )

            matched = chunk[
                chunk["tweet_id_str"].isin(new_parent_ids)
            ]

            for _, row in matched.iterrows():
                tweet_id = clean_id(row["tweet_id"])

                if tweet_id:
                    tweets[tweet_id] = {
                        "tweet_id": tweet_id,
                        "author_id": str(row["author_id"]),
                        "inbound": bool(row["inbound"]),
                        "created_at": row["created_at"],
                        "text": str(row["text"]),
                        "response_tweet_id": row[
                            "response_tweet_id"
                        ],
                        "in_response_to_tweet_id": row[
                            "in_response_to_tweet_id"
                        ],
                    }

        all_ids.update(new_parent_ids)

    # ---------------------------------------------------------
    # Create dataframe
    # ---------------------------------------------------------
    df = pd.DataFrame(tweets.values())

    if df.empty:
        print("ERROR: No tweets found.")
        return

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce"
    )

    df = df.sort_values(
        ["created_at", "tweet_id"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"\nOutput:")
    print(f"  {OUTPUT_PATH}")

    print(f"\nTotal tweets:")
    print(f"  {len(df):,}")

    print("\nSpeaker distribution:")
    print(
        df["inbound"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()