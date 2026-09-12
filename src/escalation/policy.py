def decide_action(intent_confidence, retrieval_similarity):
    """
    Conservative escalation policy.

    AUTO_HANDLE only when both:
    - intent confidence is sufficiently high
    - relevant historical evidence is available
    """

    INTENT_THRESHOLD = 0.60
    RETRIEVAL_THRESHOLD = 0.35

    if (
        intent_confidence >= INTENT_THRESHOLD
        and retrieval_similarity >= RETRIEVAL_THRESHOLD
    ):
        return {
            "action": "AUTO_HANDLE",
            "reason": (
                "High intent confidence and sufficiently similar "
                "historical support evidence were found."
            ),
        }

    reasons = []

    if intent_confidence < INTENT_THRESHOLD:
        reasons.append("low intent confidence")

    if retrieval_similarity < RETRIEVAL_THRESHOLD:
        reasons.append("insufficient historical evidence")

    return {
        "action": "ESCALATE",
        "reason": "Escalated because " + " and ".join(reasons) + ".",
    }


if __name__ == "__main__":
    examples = [
        (0.91, 0.48),
        (0.55, 0.48),
        (0.91, 0.20),
    ]

    for intent_confidence, retrieval_similarity in examples:
        result = decide_action(
            intent_confidence,
            retrieval_similarity
        )

        print(
            f"Intent confidence={intent_confidence}, "
            f"retrieval={retrieval_similarity}"
        )
        print(result)
        print()
