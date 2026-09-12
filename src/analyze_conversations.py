from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/twcs.csv")
CHUNK_SIZE = 50_000

TOP_BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
]


def main():
    print("=" * 70)
    print("HIVER SUPPORT AGENT - CONVERSATION ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------
    # We first identify all tweets belonging to conversations
    # involving our candidate brands.
    # ---------------------------------------------------------

    brand_tweets = {
        brand: []
        for brand in TOP_BRANDS
    }

    print("\nReading dataset...")

    total_rows = 0

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "response_tweet_id",
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

        for brand in TOP_BRANDS:
            mask = df["author_id"] == brand

            if mask.any():
                brand_rows = df.loc[mask]

                brand_tweets[brand].extend(
                    brand_rows[
                        [
                            "tweet_id",
                            "author_id",
                            "inbound",
                            "response_tweet_id",
                            "in_response_to_tweet_id",
                        ]
                    ].to_dict("records")
                )

    print("\n\nDataset scan complete.")

    # ---------------------------------------------------------
    # Basic statistics
    # ---------------------------------------------------------

    print("\nBrand conversation statistics:")
    print("-" * 70)

    results = []

    for brand, tweets in brand_tweets.items():

        brand_df = pd.DataFrame(tweets)

        if brand_df.empty:
            continue

        brand_replies = len(brand_df)

        # Every brand reply usually points to a customer tweet.
        replied_to = (
            brand_df["in_response_to_tweet_id"]
            .notna()
            .sum()
        )

        results.append(
            {
                "brand": brand,
                "brand_replies": brand_replies,
                "replied_to_tweets": replied_to,
            }
        )

    results_df = pd.DataFrame(results)

    print(
        results_df.sort_values(
            "brand_replies",
            ascending=False,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()