# Hiver SDE Intern Take-Home: AI Support Agent

## 1. Problem Framing

The goal is to build an AI support agent for one brand using the Customer Support on Twitter dataset. The agent should:

1. Classify an incoming customer message into a compact intent taxonomy.
2. Retrieve relevant historical support cases.
3. Draft a grounded customer-facing response.
4. Decide whether the case can be auto-handled or should be escalated.

I selected **AppleSupport** because it has substantial representation in the dataset and contains diverse technical support interactions.

The system produces intent, confidence, a grounded reply, an action, and an escalation reason.

---

## 2. Data Preparation

The Customer Support on Twitter dataset contains approximately 2.8 million tweets.

For this project:

* Brand: AppleSupport
* AppleSupport tweets: 106,860
* Reconstructed/extracted tweets: 222,948
* Support cases: 106,646
* Single-turn cases: 74,847
* Multi-turn cases: 31,799

Tweet reply relationships were used to reconstruct conversation context.

Raw dataset files are excluded from Git because of their size.

---

## 3. Intent Taxonomy

A compact 12-intent taxonomy was created from observed AppleSupport conversations:

1. `software_update_bug`
2. `battery_charging`
3. `device_hardware`
4. `connectivity`
5. `app_functionality`
6. `media_music`
7. `account_authentication`
8. `icloud_backup_restore`
9. `billing_subscription`
10. `how_to_settings`
11. `purchase_service_support`
12. `other_unclear`

The final golden evaluation set contains **155 reviewed examples**.

### Golden-set distribution

| Intent                   | Count |
| ------------------------ | ----: |
| software_update_bug      |    52 |
| app_functionality        |    21 |
| device_hardware          |    15 |
| other_unclear            |    12 |
| battery_charging         |    11 |
| media_music              |    11 |
| connectivity             |     8 |
| how_to_settings          |     6 |
| account_authentication   |     6 |
| icloud_backup_restore    |     5 |
| purchase_service_support |     5 |
| billing_subscription     |     3 |

---

## 4. Baselines

### Trivial baseline

The majority classifier always predicts `software_update_bug`.

On the final 155-example golden set:

* Accuracy: **33.55%**
* Macro Precision: **0.0280**
* Macro Recall: **0.0833**
* Macro F1: **0.0419**

### Simple ML baseline

The non-trivial baseline uses TF-IDF word and bigram features with Logistic Regression, balanced class weights, and an 80/20 stratified split.

| Metric          |     Result |
| --------------- | ---------: |
| Total examples  |        155 |
| Train           |        124 |
| Test            |         31 |
| Accuracy        | **51.61%** |
| Macro Precision | **0.3935** |
| Macro Recall    | **0.3306** |
| Macro F1        | **0.3343** |
| Weighted F1     |   **0.49** |

The simple ML model improves accuracy by **18.06 percentage points** over the majority baseline and substantially improves macro F1.

The result remains preliminary because the held-out test set contains only 31 examples and several minority intents have very limited test support.

---

## 5. System Architecture

```text
Customer Message
       |
       v
Intent Classifier
(TF-IDF + Logistic Regression)
       |
       +------> Intent + Confidence
       |
       v
Historical Case Retrieval
(TF-IDF cosine similarity)
       |
       v
Gemini 3.6 Flash
(Grounded Reply Generation)
       |
       v
Escalation Policy
       |
       +----> AUTO_HANDLE
       |
       +----> ESCALATE
```

The classifier predicts the intent and confidence. A TF-IDF retrieval index finds similar historical AppleSupport cases. The retrieved cases are supplied as evidence to Gemini 3.6 Flash for reply drafting. A conservative policy then decides whether the result should be automatically handled or escalated.

---

## 6. Retrieval

A TF-IDF retrieval index was built over **106,646 AppleSupport support cases**.

Configuration:

* lowercase text
* word unigrams and bigrams
* minimum document frequency: 2
* maximum features: 100,000
* cosine similarity for ranking

Example query:

> My iPhone battery is draining very quickly after the latest iOS update.

Top historical matches included cases about battery drain after the iOS 11 upgrade, with cosine similarities of approximately **0.48**.

This provides concrete historical evidence for the reply generator rather than relying only on the model's general knowledge.

### Retrieval limitation

The current retrieval index is built over the full AppleSupport case corpus. For a stricter offline benchmark, evaluation cases should be excluded from the retrieval index to eliminate any possibility of retrieval leakage.

---

## 7. Grounded Reply Generation

Gemini 3.6 Flash is used to draft customer-facing responses.

The generation prompt instructs the model to:

* use retrieved historical cases as evidence
* avoid inventing policies, prices, refunds, guarantees, or technical facts
* avoid claiming that an action has already been completed
* ask for additional information when the evidence is insufficient
* remain concise and professional
* output only the customer-facing reply

Example:

```text
Customer:
My iPhone battery is draining very quickly after the latest iOS update.

Historical evidence:
"My battery is dying very quickly after the 11 upgrade."

Generated response:
Let's take a look at your iPhone battery draining quickly after
the latest iOS update. Please DM us your iOS version to get started.
```

The generator is intentionally constrained because unsupported support claims are more damaging than a cautious request for additional information.

---

## 8. Auto-Handle vs Escalate

The system uses a conservative confidence policy.

```text
AUTO_HANDLE if:

intent confidence >= 0.60
AND
retrieval similarity >= 0.35

Otherwise:

ESCALATE
```

Escalation reasons distinguish between:

* low intent confidence
* insufficient historical evidence
* unavailable reply generation
* multiple uncertainty signals

For example:

```text
Confidence = 0.91
Retrieval similarity = 0.48

=> AUTO_HANDLE
```

while:

```text
Confidence = 0.55
Retrieval similarity = 0.48

=> ESCALATE
```

The conservative threshold is intentional: an uncertain automated answer is worse than asking a human support agent to review the case.

---

## 9. End-to-End Example

Input:

> My iPhone battery is draining very quickly after the latest iOS update.

Example system output:

```text
Intent: battery_charging
Confidence: 0.1457
Action: ESCALATE

Reason:
Low intent confidence / reply-generation uncertainty requires human review.

Evidence:
case_001738  similarity=0.4800
case_001828  similarity=0.4800
case_077106  similarity=0.4037
```

The low classifier confidence correctly results in escalation under the conservative policy.

When reply generation is unavailable, the system also forces escalation rather than silently presenting a generated-looking answer.

---

## 10. Golden Evaluation Set

The final evaluation set contains **155 reviewed examples**.

The first 100 examples were manually reviewed for taxonomy development and verification. An additional 55 examples were selected from held-out cases and reviewed against suggested labels.

The final set covers all 12 intents, although the distribution is intentionally imbalanced because it reflects the observed support workload.

The golden set is stored in:

```text
evaluation/golden_set.csv
```

### Sampling note

Examples were sampled from AppleSupport cases after conversation reconstruction. The initial 100 examples were used for taxonomy review and manual verification. Additional examples were sampled from cases outside the initial review set to expand the benchmark.

This is a small benchmark relative to the full 106,646-case AppleSupport corpus, so results should not be interpreted as production-scale estimates.

---

## 11. Reply Quality Evaluation

A 30-case reply evaluation set was used to assess generated customer-facing responses.

Gemini 3.6 Flash acted as the LLM judge. Each response was scored from 1–5 on:

* correctness
* grounding
* relevance
* helpfulness
* absence of unsupported claims
* overall quality

### LLM-as-Judge Results

| Dimension             |     Mean |
| --------------------- | -------: |
| Correctness           | **5.00** |
| Grounding             | **4.93** |
| Relevance             | **5.00** |
| Helpfulness           | **4.67** |
| No unsupported claims | **5.00** |
| Overall               | **4.93** |

The overall LLM-judge score was **4.93/5**.

The strongest dimensions were correctness, relevance, and absence of unsupported claims. Helpfulness was lower because several responses were technically appropriate but primarily requested additional information rather than providing a concrete troubleshooting procedure.

---

## 12. Human-vs-LLM Agreement

The same 30 replies were reviewed using the same 1–5 dimensions.

| Dimension             | LLM Mean | Human Mean | Exact Agreement |
| --------------------- | -------: | ---------: | --------------: |
| Correctness           |     5.00 |       5.00 |          100.0% |
| Grounding             |     4.93 |       4.87 |           80.0% |
| Relevance             |     5.00 |       5.00 |          100.0% |
| Helpfulness           |     4.67 |       4.13 |           33.3% |
| No unsupported claims |     5.00 |       5.00 |          100.0% |
| Overall               | **4.93** |   **4.83** |       **83.3%** |

For overall ratings, weighted Cohen's kappa was **0.211**.

For helpfulness, weighted Cohen's kappa was **0.118**.

Kappa was not reported for correctness, relevance, and no-unsupported-claims because both raters assigned the same score to all examples, making Cohen's kappa undefined for those dimensions.

The largest disagreement was on helpfulness: the LLM judge averaged **4.67**, while the human reviewer averaged **4.13**. This suggests that an LLM judge can be more generous when a response is correct and relevant but provides limited actionable guidance.

---

## 13. Top 5 Failure Modes

### 1. Generic responses

Some support situations receive a safe but generic response such as asking the customer to DM.

**Hypothesis:** The generator is intentionally conservative, but the prompt could encourage more evidence-backed troubleshooting before escalation.

### 2. Low classifier confidence

The classifier can produce low confidence even when the intent appears obvious to a human.

**Hypothesis:** The golden set is small and imbalanced, so minority intents have limited training examples.

### 3. Intent overlap

Some cases can reasonably fit multiple categories, particularly:

* software update vs app functionality
* hardware vs battery
* settings vs account authentication

**Hypothesis:** Some support intents are naturally overlapping rather than mutually exclusive.

### 4. Limited historical evidence

Some customer messages do not have sufficiently similar historical cases.

**Hypothesis:** TF-IDF retrieval relies heavily on lexical similarity and may fail when customers describe the same issue using different vocabulary.

### 5. Judge helpfulness inflation

The LLM judge gives higher helpfulness scores than the human reviewer.

**Evidence:** LLM helpfulness = **4.67/5** versus human helpfulness = **4.13/5**.

**Hypothesis:** The judge rewards correctness and relevance even when the response does not provide enough immediate actionable guidance.

---

## 14. What Is Misleading About My Headline Number?

The most attractive headline number is the **51.61% intent accuracy**.

However, it should not be interpreted as production-level intent classification accuracy.

Important caveats:

1. The benchmark contains only 155 examples.
2. The held-out test set contains only 31 examples.
3. The dataset is highly imbalanced.
4. Some intents have very few test examples.
5. Only a single stratified train/test split was used.
6. The retrieval benchmark can contain overlapping historical conversations.
7. The reply-quality evaluation contains only 30 cases.

Therefore, the 51.61% accuracy demonstrates improvement over the trivial baseline, but it is not a reliable estimate of real-world performance.

The more meaningful conclusion is that the simple TF-IDF + Logistic Regression baseline improves substantially over majority prediction while exposing the limitations of a small and imbalanced evaluation set.

---

## 15. Limitations

### Data limitations

The Customer Support on Twitter dataset represents public Twitter support interactions from a historical period. It may not represent modern customer-support traffic.

### Evaluation limitations

The intent benchmark is small and has uneven class distribution.

The reply evaluation is also small at 30 cases.

### Retrieval limitation

The current retrieval index uses the full case corpus. A stricter evaluation should remove benchmark cases and potentially their near-duplicates from the retrieval index.

### Model limitation

The generation model can produce safe but generic responses when evidence is weak.

### Taxonomy limitation

The 12-intent taxonomy is derived from AppleSupport data and should not be assumed to generalize directly to other brands.

---

## 16. What I Would Do Next Week

### 1. Improve intent classification

* Expand the golden set to 500–1,000 examples.
* Use repeated stratified cross-validation.
* Compare Logistic Regression with a small transformer or embedding classifier.
* Analyze confusion matrices for overlapping intents.

### 2. Improve retrieval

* Move from TF-IDF retrieval to embedding-based retrieval.
* Add conversation-level metadata.
* Remove evaluation cases and near-duplicates from the retrieval index.

### 3. Improve reply quality

* Require the generator to cite which historical evidence supports its response.
* Add stronger checks for unsupported claims.
* Evaluate whether replies actually resolve the customer's issue rather than merely requesting a DM.

### 4. Improve escalation

Calibrate escalation thresholds using a validation set rather than choosing thresholds manually.

A useful objective would be to optimize:

```text
Expected Support Cost =
False Auto-Handle Cost
+
Human Escalation Cost
```

### 5. Expand evaluation

* Increase reply evaluation beyond 30 cases.
* Use multiple human reviewers.
* Measure inter-rater agreement.
* Test robustness on adversarial and out-of-domain support messages.

---

## 17. Decision Log

1. **Selected AppleSupport** — chosen because it has high volume and diverse support interactions.
2. **Single-brand scope** — keeps the system focused and makes historical evidence brand-specific.
3. **Reconstructed conversations** — reply relationships were used instead of treating every tweet independently.
4. **Case definition** — each AppleSupport response was associated with a customer-support case.
5. **12-intent taxonomy** — chosen to balance coverage and manageable classification complexity.
6. **Explicit `other_unclear` class** — prevents forcing ambiguous examples into incorrect intents.
7. **Human review** — used to validate the taxonomy and golden-set labels.
8. **Majority baseline** — establishes a trivial reference point.
9. **TF-IDF + Logistic Regression** — selected as a strong, simple and interpretable baseline.
10. **TF-IDF retrieval** — selected for simplicity, speed, and transparent similarity scores.
11. **Grounded generation** — historical cases are supplied to the LLM to reduce unsupported responses.
12. **Conservative escalation** — uncertain cases should reach a human rather than receive an unreliable automated response.
13. **Gemini 3.6 Flash** — selected for reply generation and LLM-based evaluation.
14. **Evaluation honesty** — headline metrics are reported together with dataset size and sampling limitations.
15. **Known retrieval limitation** — the current index may contain evaluation cases; future evaluation should enforce retrieval isolation.

---

## 18. Reproducibility

Clone the repository:

```bash
git clone https://github.com/Gangeshh21/hiver-support-agent.git
cd hiver-support-agent
```

Create the environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the Gemini API key in `.env`:

```text
GEMINI_API_KEY=your_key_here
```

Do not commit `.env` or the dataset.

Run the data pipeline scripts in `src/data/` to reproduce the AppleSupport conversation and case datasets.

Run the intent baseline:

```bash
python src/intent/baseline.py
```

Run retrieval:

```bash
python src/retrieval/index.py
```

Run the end-to-end agent:

```bash
python src/pipeline.py
```

Run reply evaluation:

```bash
python src/evaluation/run_reply_evaluation.py
```

The repository contains the code and evaluation artifacts required to reproduce the headline experiments without committing the large raw dataset or API credentials.
