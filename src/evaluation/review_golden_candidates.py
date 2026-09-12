import pandas as pd

PATH = "data/golden/golden_candidates_100.csv"

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

df = pd.read_csv(PATH, dtype=str, keep_default_na=False)

pending = df[df["review_status"] == ""].index.tolist()

print(f"Pending cases: {len(pending)}")
print("")

for index in pending:
    row = df.loc[index]

    print("=" * 80)
    print(f"Case {index + 1}/{len(df)}")
    print(f"ID: {row['case_id']}")
    print(f"\nCustomer:\n{row['initial_customer_message']}")
    print(f"\nAppleSupport reply:\n{row['final_apple_reply']}")

    print("\nChoose intent:")
    for i, intent in enumerate(INTENTS, 1):
        print(f"{i:2}. {intent}")

    while True:
        choice = input("\nEnter number, s=skip, q=quit: ").strip().lower()

        if choice == "q":
            df.to_csv(PATH, index=False)
            print("\nSaved. Exiting.")
            raise SystemExit

        if choice == "s":
            break

        if choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            intent = INTENTS[int(choice) - 1]

            note = input("Label note (optional): ").strip()

            df.loc[index, "intent"] = intent
            df.loc[index, "label_notes"] = note
            df.loc[index, "review_status"] = "human_verified"

            df.to_csv(PATH, index=False)

            print(f"Saved: {intent}")
            break

        print("Invalid choice. Enter 1-12, s, or q.")

print("\nAll candidate cases reviewed!")

verified = (df["review_status"] == "human_verified").sum()
print(f"Human-verified: {verified}/{len(df)}")
