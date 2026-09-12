# AI Support Agent for AppleSupport

Hiver SDE Intern Take-Home Assignment — 2027 Batch

## Overview

This project implements an AI-powered customer support agent for **AppleSupport** using the Customer Support on Twitter dataset.

Given a customer message, the system:

1. Classifies the customer's intent.
2. Retrieves similar historical AppleSupport cases.
3. Generates a grounded customer-facing reply.
4. Decides whether to `AUTO_HANDLE` or `ESCALATE`.
5. Provides evidence and an explanation for the decision.

Example output:

```json
{
  "intent": "battery_charging",
  "confidence": 0.1457,
  "reply": "We'd like to look into this further. Please DM us with your current iOS version so we can help.",
  "action": "ESCALATE",
  "reason": "Low intent confidence: 0.1457 < 0.60"
}
Architecture
                    Customer Message
                           |
                           v
                +---------------------+
                |   Intent Classifier |
                | TF-IDF + Logistic   |
                |     Regression      |
                +---------------------+
                           |
                    Intent + Confidence
                           |
                           v
                +---------------------+
                | Historical Retrieval|
                | TF-IDF + Cosine     |
                |    Similarity       |
                +---------------------+
                           |
                    Historical Evidence
                           |
                           v
                +---------------------+
                | Gemini 2.5 Flash    |
                | Grounded Reply      |
                | Generation          |
                +---------------------+
                           |
                           v
                +---------------------+
                | Escalation Policy   |
                +---------------------+
                    /             \
                   v               v
             AUTO_HANDLE        ESCALATE
Dataset

Dataset:

Customer Support on Twitter

The full dataset contains approximately 2.8 million tweets.

For this project, the selected brand is:

AppleSupport

Data statistics
Item	Count
AppleSupport tweets	106,860
Reconstructed/extracted tweets	222,948
Support cases	106,646
Single-turn cases	74,847
Multi-turn cases	31,799

Conversation context was reconstructed using the Twitter reply relationships:

in_response_to_tweet_id
response_tweet_id

Raw dataset files are intentionally excluded from Git because of their size.

Intent Taxonomy

The project uses 12 intents derived from observed AppleSupport conversations:

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

other_unclear is used when the message does not contain enough information to confidently assign a specific intent.

Project Structure

hiver-support-agent/
```text
hiver-support-agent/
│
├── README.md
├── requirements.txt
├── .gitignore
├── decision_log.md
│
├── evaluation/
│   ├── golden_set.csv
│   └── results/
│       └── reply_quality_results.csv
│
├── report/
│   └── report.md
│
└── src/
    ├── inspect_dataset.py
    ├── analyze_brands.py
    ├── analyze_conversations.py
    ├── conversation_stats.py
    ├── sample_conversations.py
    ├── pipeline.py
    │
    ├── data/
    │   ├── preprocess.py
    │   ├── conversations.py
    │   ├── build_apple_threads.py
    │   └── build_cases.py
    │
    ├── intent/
    │   └── baseline.py
    │
    ├── retrieval/
    │   ├── index.py
    │   └── retriever.py
    │
    ├── generation/
    │   └── reply_generator.py
    │
    ├── escalation/
    │   └── policy.py
    │
    └── evaluation/
        ├── ai_suggest_intents.py
        ├── build_golden_candidates.py
        ├── evaluate_intent.py
        ├── fast_review_golden.py
        ├── judge.py
        ├── label_intents.py
        ├── reply_eval_cases.csv
        ├── review_golden_candidates.py
        ├── review_intents.py
        ├── run_reply_evaluation.py
        └── suggest_golden_labels.py
Setup
1. Clone the repository
git clone https://github.com/Gangeshh21/hiver-support-agent.git
cd hiver-support-agent
2. Create a virtual environment

Python 3.12 is recommended.

python3.12 -m venv .venv
source .venv/bin/activate

Verify:

python --version

Expected:

Python 3.12.x
3. Install dependencies
pip install -r requirements.txt
4. Configure Gemini API

The reply generation and LLM-as-judge components use Gemini.

Create .env:

GEMINI_API_KEY=your_api_key_here

Do not commit .env.

The repository already ignores .env through .gitignore.

Dataset Setup

Download the Customer Support on Twitter dataset from Kaggle:

thoughtvector/customer-support-on-twitter

Place the main CSV at:

data/raw/twcs.csv

The expected columns include:

tweet_id
author_id
inbound
created_at
text
response_tweet_id
in_response_to_tweet_id
Reproducing the Data Pipeline

Run:

python src/inspect_dataset.py

Inspect brand distribution:

python src/analyze_brands.py

Analyze conversations:

python src/analyze_conversations.py

Build AppleSupport conversation threads:

python src/data/build_apple_threads.py

Build support cases:

python src/data/build_cases.py

The processed AppleSupport case data is written under:

data/processed/
Running the Intent Evaluation

The final golden evaluation set contains 155 reviewed examples.

Run:

PYTHONPATH=. python src/evaluation/evaluate_intent.py

The evaluation compares the TF-IDF + Logistic Regression model against the majority baseline.

Headline results on the current benchmark:

Model	Accuracy	Macro F1
Majority baseline	33.55%	0.0419
TF-IDF + Logistic Regression	51.61%	0.3343

The test split contains 31 examples, so these numbers should be treated as preliminary rather than production-level estimates.

Building the Retrieval Index

Run:

python src/retrieval/index.py

The current retrieval system indexes approximately:

106,646 AppleSupport cases

Retrieval uses:

TF-IDF
unigram + bigram features
cosine similarity

Example query:

My iPhone battery is draining very quickly after the latest iOS update.

Relevant historical examples are retrieved before reply generation.

Running the End-to-End Agent

Run:

PYTHONPATH=. python src/pipeline.py

The pipeline returns:

{
  "intent": "...",
  "confidence": 0.0,
  "reply": "...",
  "action": "AUTO_HANDLE",
  "reason": "...",
  "evidence": []
}

The pipeline consists of:

Message
  ↓
Intent Classification
  ↓
Historical Retrieval
  ↓
Grounded Reply Generation
  ↓
Escalation Decision
Escalation Policy

The current prototype uses conservative thresholds:

Intent confidence >= 0.60
AND
Retrieval similarity >= 0.35

If both conditions are satisfied:

AUTO_HANDLE

Otherwise:

ESCALATE

The system also records the reason for escalation, such as low intent confidence or insufficient historical evidence.

The conservative policy is intentional: an uncertain automated answer is worse than asking a human support agent to review the case.

Reply Generation

Gemini 2.5 Flash is used to draft replies.

The generation prompt provides:

customer message
retrieved historical cases
historical replies

The model is instructed to:

stay grounded in historical evidence
avoid inventing policies
avoid inventing prices or refunds
avoid unsupported guarantees
avoid claiming actions were completed
ask for additional information when necessary
produce a concise customer-facing response
Reply Evaluation

Run:

PYTHONPATH=. python src/evaluation/run_reply_evaluation.py

The LLM judge evaluates:

correctness
grounding
relevance
helpfulness
unsupported claims
overall quality

Three cases were successfully evaluated before the Gemini API free-tier quota was exhausted.

Current observed averages:

Metric	Score
Correctness	5.00 / 5
Grounding	5.00 / 5
Relevance	5.00 / 5
Helpfulness	4.33 / 5
No unsupported claims	5.00 / 5
Overall	5.00 / 5

Important: These scores are not presented as representative system-level performance because the sample size is only three.

Golden Evaluation Set

The final golden set is:

evaluation/golden_set.csv

It contains:

155 examples
12 intents

The set was constructed from AppleSupport cases and reviewed for intent assignment.

The intent distribution is intentionally retained rather than artificially forcing equal class sizes, so that the evaluation reflects the observed support distribution.

Baselines

Two baselines are included.

Trivial baseline

Always predict the majority class:

software_update_bug

Results:

Accuracy: 33.55%
Macro F1: 0.0419
Simple ML baseline

TF-IDF + Logistic Regression:

Accuracy: 51.61%
Macro F1: 0.3343

The simple ML model improves accuracy by:

18.06 percentage points

over the majority baseline.

Known Limitations

The current system is intentionally a prototype.

Important limitations include:

The labelled benchmark contains only 155 examples.
The held-out test set contains only 31 examples.
Several intents have very small evaluation support.
A single stratified train/test split is currently used.
LLM reply evaluation was limited by API quota.
Human-vs-LLM judge agreement at 30–50 examples was not completed.
Historical support conversations contain repetitive and noisy examples.
Some intents overlap semantically.
Retrieval should be evaluated with explicit exclusion of evaluation cases in a stricter benchmark.

These limitations are documented rather than hidden because headline metrics can otherwise be misleading.

Failure Modes

The main observed failure modes are:

Class imbalance

The dataset is dominated by software-update-related issues.

Low classifier confidence

The labelled training set is relatively small, causing weak probability separation.

Intent overlap

Battery, hardware, application, and software-update issues can share similar language.

Ambiguous messages

Some customer tweets do not provide enough context to determine a precise intent.

Repetitive retrieval evidence

The historical dataset contains near-duplicate conversations.

What I Would Improve Next

With another week, I would:

Expand the golden set beyond 200 examples.
Use repeated stratified cross-validation.
Add confidence calibration.
Compare TF-IDF retrieval with dense embeddings.
Deduplicate historical cases.
Tune escalation thresholds using labelled outcomes.
Run a larger LLM-as-judge evaluation.
Complete human-vs-LLM agreement testing.
Investigate hierarchical intent classification.
Add monitoring for retrieval quality and unsupported claims.
Decision Log

Non-obvious project decisions are documented in:

decision_log.md

The log covers:

brand selection
conversation reconstruction
case definition
intent taxonomy
unclear intent handling
baseline selection
retrieval approach
grounded generation
escalation policy
evaluation methodology
known limitations
Reproducibility Summary

The shortest path to reproduce the headline intent result is:

source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python src/evaluation/evaluate_intent.py

For the full pipeline, first ensure:

data/raw/twcs.csv

is available and then run the data preparation scripts followed by retrieval and evaluation.

Project Philosophy

The primary goal of this project is not to maximize a single metric.

The design prioritizes:

grounded responses
conservative automation
explicit uncertainty
historical evidence
reproducible evaluation
honest reporting of limitations

A support agent should prefer escalation over confidently giving an unsupported answer.
