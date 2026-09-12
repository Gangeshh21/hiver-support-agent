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

- Brand: AppleSupport
- AppleSupport tweets: 106,860
- Reconstructed/extracted tweets: 222,948
- Support cases: 106,646
- Single-turn cases: 74,847
- Multi-turn cases: 31,799

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

| Intent | Count |
|---|---:|
| software_update_bug | 52 |
| app_functionality | 21 |
| device_hardware | 15 |
| other_unclear | 12 |
| battery_charging | 11 |
| media_music | 11 |
| connectivity | 8 |
| how_to_settings | 6 |
| account_authentication | 6 |
| icloud_backup_restore | 5 |
| purchase_service_support | 5 |
| billing_subscription | 3 |

---

## 4. Baselines

### Trivial baseline

The majority classifier always predicts `software_update_bug`.

On the final 155-example golden set:

- Accuracy: **33.55%**
- Macro Precision: **0.0280**
- Macro Recall: **0.0833**
- Macro F1: **0.0419**

### Simple ML baseline

The non-trivial baseline uses TF-IDF word and bigram features with Logistic Regression, balanced class weights, and an 80/20 stratified split.

| Metric | Result |
|---|---:|
| Total examples | 155 |
| Train | 124 |
| Test | 31 |
| Accuracy | **51.61%** |
| Macro Precision | **0.3935** |
| Macro Recall | **0.3306** |
| Macro F1 | **0.3343** |
| Weighted F1 | **0.49** |

The simple ML model improves accuracy by **18.06 percentage points** over the majority baseline and substantially improves macro F1.

The result remains preliminary because the held-out test set contains only 31 examples.

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
Gemini 2.5 Flash
(Grounded Reply Generation)
       |
       v
Escalation Policy
       |
       +----> AUTO_HANDLE
       |
       +----> ESCALATE