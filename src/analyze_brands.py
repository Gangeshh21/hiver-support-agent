from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/twcs.csv")
CHUNK_SIZE = 50_000


def main():
    print("=" * 70)
    print("HIVER SUPPORT AGENT - CORRECT BRAND ANALYSIS")
    print("=" * 70)

    brand_stats = {}

    total_rows = 0

    # ---------------------------------------------------------
    # Pass 1:
    # Count outbound/support tweets for every author.
    # ---------------------------------------------------------
    print("\nPass 1: Finding support/brand accounts...")

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "in_response_to_tweet_id",
            ],
        ),
        start=1,
    ):
        total_rows += len(df)

        print(
            f"\rProcessing chunk {chunk_number} | "
            f"Rows: {total_rows:,}",
            end="",
        )

        outbound = df[df["inbound"] == False]

        counts = outbound.groupby("author_id").agg(
            brand_replies=("tweet_id", "count"),
            replies_to_customer=("in_response_to_tweet_id", "count"),
        )

        for author_id, row in counts.iterrows():
            if author_id not in brand_stats:
                brand_stats[author_id] = {
                    "brand_replies": 0,
                    "replies_to_customer": 0,
                }

            brand_stats[author_id]["brand_replies"] += int(
                row["brand_replies"]
            )

            brand_stats[author_id]["replies_to_customer"] += int(
                row["replies_to_customer"]
            )

    print("\n\nPass 1 complete.")

    # ---------------------------------------------------------
    # Candidate brands
    # ---------------------------------------------------------
    candidates = []

    for author_id, stats in brand_stats.items():
        if stats["brand_replies"] >= 100:
            candidates.append(
                {
                    "brand": author_id,
                    "brand_replies": stats["brand_replies"],
                    "replies_to_customer": stats["replies_to_customer"],
                }
            )

    candidates_df = pd.DataFrame(candidates)

    candidates_df = candidates_df.sort_values(
        by="brand_replies",
        ascending=False,
    )

    print("\nTop 30 candidate support accounts:")
    print("-" * 70)

    print(
        candidates_df.head(30).to_string(index=False)
    )


if __name__ == "__main__":
    main()