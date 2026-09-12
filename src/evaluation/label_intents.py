from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/golden/intent_review_100.csv")

INTENTS = {
    "1": "software_update_bug",
    "2": "battery_charging",
    "3": "device_hardware",
    "4": "connectivity",
    "5": "app_functionality",
    "6": "media_music",
    "7": "account_authentication",
    "8": "icloud_backup_restore",
    "9": "billing_subscription",
    "10": "how_to_settings",
    "11": "purchase_service_support",
    "12": "other_unclear",
}


def show_intents():
    print("\n" + "=" * 60)
    print("INTENT LABELS")
    print("=" * 60)

    for number, intent in INTENTS.items():
        print(f"{number:>2}. {intent}")

    print("\nCommands:")
    print("  s = skip")
    print("  q = quit and save")


def main():

    print("=" * 60)
    print("APPLE SUPPORT - INTENT LABELING TOOL")
    print("=" * 60)

    # Read everything as strings
    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False
    )

    # Make sure required columns exist
    required_columns = [
        "intent",
        "label_notes",
        "suggested_intent",
        "review_status",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

        # Force column to string
        df[column] = df[column].astype(str)

    show_intents()

    # Only show unlabeled cases
    unlabeled = df[
        df["intent"].str.strip() == ""
    ]

    print(f"\nUnlabeled cases: {len(unlabeled)}")

    for index in unlabeled.index:

        row = df.loc[index]

        print("\n" + "-" * 70)
        print(f"CASE {index + 1}/{len(df)}")
        print(f"Case ID: {row['case_id']}")

        print("\nCustomer message:")
        print(row["initial_customer_message"])

        print("\nSelect intent:")

        for number, intent in INTENTS.items():
            print(f"  {number}. {intent}")

        choice = input("\nYour choice: ").strip()

        # Quit
        if choice.lower() == "q":

            df.to_csv(
                INPUT_PATH,
                index=False
            )

            print("\nProgress saved.")
            return

        # Skip
        if choice.lower() == "s":

            df.at[index, "review_status"] = "skipped"

            df.to_csv(
                INPUT_PATH,
                index=False
            )

            print("Skipped.")
            continue

        # Invalid choice
        if choice not in INTENTS:

            print(
                "\nInvalid choice. "
                "Please select 1-12, s, or q."
            )

            continue

        selected_intent = INTENTS[choice]

        # Save intent
        df.at[index, "intent"] = selected_intent
        df.at[index, "review_status"] = "reviewed"

        note = input(
            "Optional note (press Enter to skip): "
        ).strip()

        df.at[index, "label_notes"] = note

        # Save after every case
        df.to_csv(
            INPUT_PATH,
            index=False
        )

        print(
            f"Saved: {selected_intent}"
        )

    print("\n" + "=" * 60)
    print("LABELING COMPLETE")
    print("=" * 60)

    labeled_count = (
        df["intent"]
        .str.strip()
        .ne("")
        .sum()
    )

    print(
        f"Labeled cases: {labeled_count}/{len(df)}"
    )


if __name__ == "__main__":
    main()