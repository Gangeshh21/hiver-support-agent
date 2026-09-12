from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/twcs.csv")

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "TMobileHelp",
]

SAMPLE_PER_BRAND = 10


def main():
    print("=" * 70)
    print("HIVER SUPPORT AGENT - CONVERSATION SAMPLE")
    print("=" * 70)

    # ---------------------------------------------------------
    # Read only the columns we need.
    # ---------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    )

    # Create lookup by tweet ID.
    tweet_lookup = df.set_index("tweet_id").to_dict("index")

    for brand in BRANDS:

        print("\n")
        print("=" * 70)
        print(f"BRAND: {brand}")
        print("=" * 70)

        brand_df = df[
            (df["author_id"] == brand)
            & (df["inbound"] == False)
            & (df["in_response_to_tweet_id"].notna())
        ]

        samples = brand_df.sample(
            min(SAMPLE_PER_BRAND, len(brand_df)),
            random_state=42,
        )

        for _, brand_tweet in samples.iterrows():

            customer_id = int(
                brand_tweet["in_response_to_tweet_id"]
            )

            customer_tweet = tweet_lookup.get(
                customer_id
            )

            print("\n" + "-" * 70)

            if customer_tweet:
                print(
                    f"Customer: "
                    f"{customer_tweet['text']}"
                )

            print(
                f"{brand}: "
                f"{brand_tweet['text']}"
            )


if __name__ == "__main__":
    main()
    