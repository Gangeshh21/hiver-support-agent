# Decision Log

## 1. Brand selection: AppleSupport
Selected AppleSupport because it has high volume in the Twitter Support dataset and contains diverse technical support interactions suitable for intent classification and retrieval.

## 2. Use a single-brand system
Restricted the system to AppleSupport rather than building a multi-brand agent. This keeps the retrieval corpus and response style consistent and matches the assignment requirement to choose one brand.

## 3. Conversation reconstruction
Used tweet reply relationships to reconstruct customer-support conversations rather than treating every tweet as an isolated example.

## 4. Case definition
Defined a support case around an AppleSupport response, pairing customer context with the historical AppleSupport reply. This makes each case useful as a retrieval example.

## 5. Intent taxonomy
Created a compact 12-intent taxonomy from observed AppleSupport conversations instead of using hundreds of raw topics. The taxonomy balances coverage with classification feasibility.

## 6. Explicit unclear class
Added `other_unclear` for ambiguous, incomplete, or insufficient-context messages. This avoids forcing every noisy tweet into an incorrect specific intent.

## 7. Human verification
Created a manually reviewed 100-example intent set and used it as the initial labelled dataset for classifier evaluation.

## 8. Majority baseline
Implemented a majority-class baseline before the ML system. This provides a deliberately simple reference point and exposes class imbalance.

## 9. TF-IDF + Logistic Regression baseline
Used TF-IDF word/phrase features with Logistic Regression as a lightweight, interpretable baseline before introducing an LLM.

## 10. Retrieval approach
Used TF-IDF cosine similarity over historical AppleSupport cases. This was chosen for simplicity, speed, reproducibility, and low infrastructure cost.

## 11. Grounded generation
The reply generator receives retrieved historical cases as evidence and is instructed not to invent policies, prices, refunds, guarantees, or completed actions.

## 12. Conservative escalation
The first escalation policy requires both sufficient intent confidence and sufficient retrieval similarity. If either signal is weak, the case is escalated rather than automatically handled.

## 13. LLM choice
Used Gemini 2.5 Flash for response generation and LLM-based reply judging because it was available through the development environment and provided sufficient quality for prototyping.

## 14. Evaluation honesty
The LLM-as-judge evaluation was limited by the free-tier API quota. Only successfully completed judge evaluations are reported; failed or unavailable evaluations are not fabricated.

## 15. Known evaluation limitation
The current intent benchmark contains 100 manually verified examples, while the assignment target is 150–250. The additional candidate examples were not counted as hand-labelled golden data.
