import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)


def generate_reply(customer_message, retrieved_cases, intent):
    evidence = ""

    for i, case in enumerate(retrieved_cases, 1):
        evidence += f"""
Historical Case {i}
Customer: {case['customer_message']}
AppleSupport Reply: {case['historical_reply']}
Similarity: {case['similarity']:.4f}
"""

    prompt = f"""
You are an Apple customer support agent.

Customer message:
{customer_message}

Predicted intent:
{intent}

Relevant historical AppleSupport cases:
{evidence}

Write a helpful draft reply for the customer.

Rules:
- Use the historical cases as evidence.
- Do not invent policies, refunds, prices, guarantees, or technical facts.
- Do not claim an action was completed if it was not.
- Keep the reply concise and professional.
- If more information is needed, politely ask the customer for it.
- Do not mention that an AI or retrieval system was used.
- Output only the customer-facing reply.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text.strip()


if __name__ == "__main__":
    customer_message = (
        "My iPhone battery is draining very quickly "
        "after the latest iOS update."
    )

    retrieved_cases = [
        {
            "customer_message": "My battery is dying very quickly after the 11 upgrade",
            "historical_reply": (
                "Let's take a look at this together. "
                "DM us your iOS version to get started."
            ),
            "similarity": 0.48,
        },
        {
            "customer_message": (
                "With new update my iphone 7 battery draining very quickly."
            ),
            "historical_reply": (
                "We'd like to look into this further. "
                "Join us in DM here."
            ),
            "similarity": 0.40,
        },
    ]

    reply = generate_reply(
        customer_message,
        retrieved_cases,
        "battery_charging"
    )

    print("=== Generated Reply ===")
    print(reply)
