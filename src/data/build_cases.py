from pathlib import Path
import pandas as pd
import re

INPUT_PATH = Path("data/processed/apple_support_threads.csv")
OUTPUT_PATH = Path("data/processed/apple_support_cases.csv")

BRAND = "AppleSupport"


def clean_text(text):
    """Clean whitespace from tweet text."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_id(value):
    """Convert tweet IDs to consistent string format."""
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return None


def main():

    print("=" * 60)
    print("APPLE SUPPORT - BUILD SUPPORT CASES")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    print("\n[1/5] Loading threads...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df):,} tweets.")

    # ---------------------------------------------------------
    # 2. CLEAN DATA
    # ---------------------------------------------------------

    print("\n[2/5] Cleaning data...")

    df["tweet_id"] = df["tweet_id"].apply(clean_id)

    df["parent_id"] = df[
        "in_response_to_tweet_id"
    ].apply(clean_id)

    df["text"] = df["text"].apply(clean_text)

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True
    )

    # Remove tweets without text
    df = df[
        df["text"].str.len() > 0
    ].copy()

    # Sort chronologically
    df = df.sort_values(
        ["created_at", "tweet_id"]
    ).reset_index(drop=True)

    print(f"Tweets after cleaning: {len(df):,}")

    # ---------------------------------------------------------
    # 3. BUILD LOOKUP
    # ---------------------------------------------------------

    print("\n[3/5] Building tweet lookup...")

    # IMPORTANT:
    # Keep tweet_id INSIDE each dictionary record.
    # This avoids the previous KeyError.
    tweets = {}

    for _, row in df.iterrows():

        tweet_id = row["tweet_id"]

        if tweet_id is None:
            continue

        tweets[tweet_id] = {
            "tweet_id": tweet_id,
            "author_id": str(row["author_id"]),
            "inbound": bool(row["inbound"]),
            "created_at": row["created_at"],
            "text": row["text"],
            "parent_id": row["parent_id"],
        }

    print(f"Tweets in lookup: {len(tweets):,}")

    # ---------------------------------------------------------
    # 4. BUILD SUPPORT CASES
    # ---------------------------------------------------------

    print("\n[4/5] Building support cases...")

    apple_rows = df[
        df["author_id"] == BRAND
    ]

    print(
        f"AppleSupport tweets: "
        f"{len(apple_rows):,}"
    )

    cases = []

    for count, (_, apple_row) in enumerate(
        apple_rows.iterrows(),
        start=1
    ):

        apple_id = apple_row["tweet_id"]

        parent_id = apple_row["parent_id"]

        # Standalone AppleSupport tweet
        if parent_id is None:
            continue

        # Find direct parent
        parent = tweets.get(parent_id)

        if parent is None:
            continue

        # Ignore AppleSupport -> AppleSupport replies
        if parent["author_id"] == BRAND:
            continue

        # -----------------------------------------------------
        # Walk backwards through parent relationships
        # -----------------------------------------------------

        chain_ids = [apple_id]

        current_id = parent_id

        visited = {apple_id}

        while current_id is not None:

            # Prevent circular relationships
            if current_id in visited:
                break

            visited.add(current_id)

            current = tweets.get(current_id)

            if current is None:
                break

            chain_ids.append(current_id)

            current_id = current["parent_id"]

        # -----------------------------------------------------
        # Convert IDs to tweet records
        # -----------------------------------------------------

        chain_rows = []

        for tweet_id in chain_ids:

            tweet = tweets.get(tweet_id)

            if tweet is not None:
                chain_rows.append(tweet)

        if not chain_rows:
            continue

        # -----------------------------------------------------
        # Sort conversation chronologically
        # -----------------------------------------------------

        chain_rows = sorted(
            chain_rows,
            key=lambda x: (
                x["created_at"],
                x["tweet_id"]
            )
        )

        # -----------------------------------------------------
        # Get customer messages
        # -----------------------------------------------------

        customer_messages = [
            row
            for row in chain_rows
            if row["author_id"] != BRAND
        ]

        if not customer_messages:
            continue

        # First customer message in this thread
        first_customer = customer_messages[0]

        first_customer_id = first_customer[
            "tweet_id"
        ]

        # -----------------------------------------------------
        # Build readable conversation
        # -----------------------------------------------------

        conversation_parts = []

        for row in chain_rows:

            if row["author_id"] == BRAND:
                speaker = "AppleSupport"
            else:
                speaker = "Customer"

            conversation_parts.append(
                f"{speaker}: {row['text']}"
            )

        conversation = "\n".join(
            conversation_parts
        )

        # -----------------------------------------------------
        # Create case
        # -----------------------------------------------------

        case_id = (
            f"case_{len(cases) + 1:06d}"
        )

        cases.append({
            "case_id": case_id,

            "customer_id": str(
                first_customer["author_id"]
            ),

            "apple_support_tweet_id": apple_id,

            "customer_tweet_id": first_customer_id,

            "turn_count": len(chain_rows),

            "initial_customer_message":
                first_customer["text"],

            "final_apple_reply":
                apple_row["text"],

            "conversation":
                conversation,

            "created_at":
                first_customer["created_at"],
        })

        # Progress indicator
        if count % 10_000 == 0:
            print(
                f"  Processed AppleSupport tweets: "
                f"{count:,} | "
                f"Cases: {len(cases):,}"
            )

    # ---------------------------------------------------------
    # 5. SAVE RESULTS
    # ---------------------------------------------------------

    print("\n[5/5] Saving cases...")

    cases_df = pd.DataFrame(cases)

    if cases_df.empty:

        print("\nERROR: No support cases were created.")

        return

    cases_df["created_at"] = pd.to_datetime(
        cases_df["created_at"],
        errors="coerce",
        utc=True
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cases_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("CASE BUILDING COMPLETE")
    print("=" * 60)

    print(
        f"\nCases created:"
        f" {len(cases_df):,}"
    )

    single_turn = (
        cases_df["turn_count"] == 2
    ).sum()

    multi_turn = (
        cases_df["turn_count"] > 2
    ).sum()

    print(
        f"\nSingle-turn cases:"
        f" {single_turn:,}"
    )

    print(
        f"\nMulti-turn cases:"
        f" {multi_turn:,}"
    )

    print(
        f"\nMaximum turns:"
        f" {cases_df['turn_count'].max()}"
    )

    print("\nTurn-count distribution:")

    print(
        cases_df[
            "turn_count"
        ]
        .value_counts()
        .sort_index()
        .head(15)
        .to_string()
    )

    print("\nUnique customers:")

    print(
        cases_df[
            "customer_id"
        ].nunique()
    )

    print("\nSaved to:")

    print(
        f"  {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()