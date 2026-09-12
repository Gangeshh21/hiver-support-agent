from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/golden/intent_review_100.csv")

INTENTS = [
    "software_update_bug",
    "battery_charging",
    "device_hardware",
    "connectivity",
    "app_functionality",
    "media_music",
    "account_authentication",
    "icloud_backup_restore",
    "billing_subscription",
    "how_to_settings",
    "purchase_service_support",
    "other_unclear",
]


def show_intents():
    print("\nAvailable intents:")

    for i, intent in enumerate(INTENTS, start=1):
        print(f"{i:>2}. {intent}")

    print("\nCommands:")
    print("  a = accept AI suggestion")
    print("  c = choose/correct label")
    print("  s = skip")
    print("  q = quit and save")


def main():

    print("=" * 65)
    print("HUMAN VERIFICATION - INTENT LABELS")
    print("=" * 65)

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    for column in [
        "intent",
        "suggested_intent",
        "review_status",
        "label_notes",
    ]:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].astype(str)

    # Cases that still need human verification
    pending = df[
        df["review_status"].isin(
            ["ai_suggested", ""]
        )
    ]

    print(
        f"\nCases needing human verification: "
        f"{len(pending)}"
    )

    for count, index in enumerate(
        pending.index,
        start=1
    ):

        case_id = df.at[index, "case_id"]
        message = df.at[
            index,
            "initial_customer_message"
        ]

        suggestion = df.at[
            index,
            "suggested_intent"
        ].strip()

        print("\n" + "-" * 70)
        print(
            f"CASE {count}/{len(pending)}"
        )
        print(
            f"Case ID: {case_id}"
        )

        print("\nCustomer message:")
        print(message)

        if suggestion:
            print(
                f"\nGemini suggestion: "
                f"{suggestion}"
            )
        else:
            print(
                "\nGemini suggestion: "
                "NOT AVAILABLE"
            )

        choice = input(
            "\nAction [a/c/s/q]: "
        ).strip().lower()

        # Quit
        if choice == "q":

            df.to_csv(
                INPUT_PATH,
                index=False
            )

            print(
                "\nProgress saved."
            )

            return

        # Skip
        if choice == "s":

            df.at[
                index,
                "review_status"
            ] = "skipped"

            df.to_csv(
                INPUT_PATH,
                index=False
            )

            print("Skipped.")

            continue

        # Accept AI suggestion
        if choice == "a":

            if suggestion not in INTENTS:

                print(
                    "\nNo valid AI suggestion "
                    "available."
                )

                continue

            df.at[
                index,
                "intent"
            ] = suggestion

            df.at[
                index,
                "review_status"
            ] = "human_verified"

            df.to_csv(
                INPUT_PATH,
                index=False
            )

            print(
                f"Human verified: {suggestion}"
            )

            continue

        # Correct / manually choose
        if choice == "c":

            show_intents()

            label = input(
                "\nEnter intent number: "
            ).strip()

            if not label.isdigit():

                print(
                    "\nInvalid choice."
                )

                continue

            label_number = int(label)

            if (
                label_number < 1
                or label_number > len(INTENTS)
            ):

                print(
                    "\nInvalid intent number."
                )

                continue

            selected_intent = INTENTS[
                label_number - 1
            ]

            note = input(
                "Optional note: "
            ).strip()

            df.at[
                index,
                "intent"
            ] = selected_intent

            df.at[
                index,
                "label_notes"
            ] = note

            df.at[
                index,
                "review_status"
            ] = "human_verified"

            df.to_csv(
                INPUT_PATH,
                index=False
            )

            print(
                f"Human verified: "
                f"{selected_intent}"
            )

            continue

        print(
            "\nInvalid action. "
            "Use a, c, s or q."
        )

    print("\n" + "=" * 65)
    print("VERIFICATION COMPLETE")
    print("=" * 65)

    verified = (
        df["review_status"]
        == "human_verified"
    ).sum()

    print(
        f"Human-verified cases: "
        f"{verified}/{len(df)}"
    )


if __name__ == "__main__":
    main()