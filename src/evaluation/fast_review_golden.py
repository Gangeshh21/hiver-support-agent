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

pending = df[df["review_status"] != "human_verified"].index.tolist()

print(f"Pending cases: {len(pending)}")

for index in pending:
    row = df.loc[index]

    print("\n" + "=" * 80)
    print(f"Case {index + 1}/{len(df)}")
    print(f"ID: {row['case_id']}")
    print(f"\nCustomer:\n{row['initial_customer_message']}")
    print(f"\nAppleSupport reply:\n{row['final_apple_reply']}")

    suggestion = row.get("suggested_intent", "")
    confidence = row.get("suggested_confidence", "")

    print(f"\nSuggested: {suggestion} (confidence: {confidence})")

    print("\nIntents:")
    for i, intent in enumerate(INTENTS, 1):
        marker = " <-- suggested" if intent == suggestion else ""
        print(f"{i:2}. {intent}{marker}")

    while True:
        choice = input(
            "\nEnter = accept suggestion | 1-12 = correct | s = skip | q = quit: "
        ).strip().lower()

        if choice == "":
            if suggestion in INTENTS:
                df.loc[index, "intent"] = suggestion
                df.loc[index, "label_notes"] = "Human verified AI-assisted suggestion"
                df.loc[index, "review_status"] = "human_verified"
                df.to_csv(PATH, index=False)
                print(f"Accepted: {suggestion}")
                break
            else:
                print("No valid suggestion. Choose 1-12.")

        elif choice == "q":
            df.to_csv(PATH, index=False)
            print("Saved. Exiting.")
            raise SystemExit

        elif choice == "s":
            print("Skipped.")
            break

        elif choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            intent = INTENTS[int(choice) - 1]
            note = input("Short label note (optional): ").strip()

            df.loc[index, "intent"] = intent
            df.loc[index, "label_notes"] = note
            df.loc[index, "review_status"] = "human_verified"

            df.to_csv(PATH, index=False)
            print(f"Saved: {intent}")
            break

        else:
            print("Invalid input.")

verified = (df["review_status"] == "human_verified").sum()

print(f"\nHuman-verified: {verified}/{len(df)}")
