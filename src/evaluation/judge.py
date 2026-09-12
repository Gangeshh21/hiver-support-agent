import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def judge_reply(customer_message, reply, evidence):
    evidence_text = "\n".join(
        f"- Customer: {x['customer_message']}\n"
        f"  Historical reply: {x['historical_reply']}"
        for x in evidence
    )

    prompt = f"""
You are evaluating an AI customer-support reply.

CUSTOMER:
{customer_message}

AI REPLY:
{reply}

HISTORICAL EVIDENCE:
{evidence_text}

Score the reply from 1 to 5 on:

1. correctness
2. grounding in historical evidence
3. relevance
4. helpfulness
5. avoidance of unsupported claims

Return ONLY valid JSON:

{{
  "correctness": <1-5>,
  "grounding": <1-5>,
  "relevance": <1-5>,
  "helpfulness": <1-5>,
  "no_unsupported_claims": <1-5>,
  "overall": <1-5>,
  "reason": "<short explanation>"
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


if __name__ == "__main__":
    print("LLM judge ready.")
