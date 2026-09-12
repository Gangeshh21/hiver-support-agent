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
    print("HIVER SUPPORT AGENT - CONVERSATION STATISTICS")
    print("=" * 70)

    # Store all relevant tweets for candidate brands.
    brand_tweets = {
        brand: []
        for brand in TOP_BRANDS
    }

    # ---------------------------------------------------------
    # PASS 1
    # Read the dataset and collect brand tweets.
    # ---------------------------------------------------------

    total_rows = 0

    print("\nReading dataset...")

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATA_PATH,
            chunksize=CHUNK_SIZE,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
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

                rows = df.loc[mask]

                brand_tweets[brand].extend(
                    rows.to_dict("records")
                )

    print("\n\nDataset scan complete.")

    # ---------------------------------------------------------
    # PASS 2
    # Analyze conversation relationships.
    # ---------------------------------------------------------

    results = []

    for brand, tweets in brand_tweets.items():

        brand_df = pd.DataFrame(tweets)

        if brand_df.empty:
            continue

        # Number of brand replies
        brand_replies = len(brand_df)

        # Unique customer tweets that the brand replied to
        customer_tweet_ids = (
            brand_df["in_response_to_tweet_id"]
            .dropna()
            .astype(int)
            .unique()
        )

        unique_customer_messages = len(
            customer_tweet_ids
        )

        results.append(
            {
                "brand": brand,
                "brand_replies": brand_replies,
                "customer_messages_replied_to":
                    unique_customer_messages,
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "brand_replies",
        ascending=False,
    )

    print("\nConversation statistics:")
    print("-" * 70)

    print(
        results_df.to_string(index=False)
    )


if __name__ == "__main__":
    main()