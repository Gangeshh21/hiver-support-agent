from pathlib import Path
import os
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai


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


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=api_key)


SYSTEM_PROMPT = """
You are an intent classification assistant for an Apple Support customer service dataset.

Classify the customer's message into exactly ONE of these intents:

software_update_bug
battery_charging
device_hardware
connectivity
app_functionality
media_music
account_authentication
icloud_backup_restore
billing_subscription
how_to_settings
purchase_service_support
other_unclear

Definitions:

software_update_bug:
Problems caused by or related to iOS, macOS, watchOS or software updates.
Examples: update broke phone, lag after update, bugs after iOS update.

battery_charging:
Battery drain, battery percentage problems, charging problems or battery life.

device_hardware:
Physical/device-level problems such as microphone, speaker, screen, buttons or hardware malfunction.

connectivity:
Wi-Fi, Bluetooth, cellular, network or connection problems.

app_functionality:
An application is malfunctioning or behaving incorrectly.

media_music:
Apple Music, Podcasts, music playback, music downloads or purchased media.

account_authentication:
Apple ID, passwords, authentication, activation, Family Sharing or account access.

icloud_backup_restore:
iCloud, backups, restoring data or syncing personal data.

billing_subscription:
Charges, invoices, subscriptions, refunds or payment-related problems.

how_to_settings:
The customer is asking how to perform an action or change/use a setting.

purchase_service_support:
Apple Store/service questions, appointments, repairs, product/service availability or service-location issues.

other_unclear:
Insufficient context, unrelated content, or a message that does not fit the other categories.

Return ONLY the exact intent name.
"""


def classify_message(message, max_retries=5):

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    SYSTEM_PROMPT,
                    f"\nCustomer message:\n{message}",
                ],
            )

            result = response.text.strip()
            result = result.replace("`", "").strip()

            if result not in INTENTS:
                print(
                    f"Unexpected model output: {result}"
                )
                return "other_unclear"

            return result

        except Exception as error:

            error_text = str(error)

            # Rate-limit / quota error
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:

                wait_time = 35

                print(
                    f"Rate limit reached. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue

            # Other errors
            raise error

    raise RuntimeError(
        "Maximum retry attempts reached because of API rate limits."
    )


def main():

    print("=" * 60)
    print("GEMINI AI-ASSISTED INTENT SUGGESTION")
    print("=" * 60)

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    # Make sure required columns exist
    for column in [
        "intent",
        "suggested_intent",
        "review_status",
        "label_notes",
    ]:

        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].astype(str)

    # Only process cases without AI suggestions
    pending = df[
        df["suggested_intent"].str.strip() == ""
    ]

    print(
        f"\nCases needing AI suggestions: {len(pending)}"
    )

    if len(pending) == 0:

        print(
            "\nAll cases already have AI suggestions."
        )

        return

    for count, index in enumerate(
        pending.index,
        start=1
    ):

        message = df.at[
            index,
            "initial_customer_message"
        ]

        print("\n" + "-" * 60)
        print(
            f"Processing {count}/{len(pending)}"
        )
        print(
            f"Case ID: {df.at[index, 'case_id']}"
        )

        try:

            suggestion = classify_message(message)

            df.at[
                index,
                "suggested_intent"
            ] = suggestion

            df.at[
                index,
                "review_status"
            ] = "ai_suggested"

            # Save immediately
            df.to_csv(
                INPUT_PATH,
                index=False,
            )

            print(
                f"AI suggestion: {suggestion}"
            )

            # Wait between requests to stay under
            # the free-tier per-minute quota.
            if count < len(pending):

                print(
                    "Waiting 15 seconds before next request..."
                )

                time.sleep(15)

        except Exception as error:

            print(
                f"\nERROR: {error}"
            )

            print(
                "\nSaving progress before stopping..."
            )

            df.to_csv(
                INPUT_PATH,
                index=False,
            )

            return

    print("\n" + "=" * 60)
    print("AI SUGGESTIONS COMPLETE")
    print("=" * 60)

    print(
        f"Suggestions generated: {len(pending)}"
    )


if __name__ == "__main__":
    main()