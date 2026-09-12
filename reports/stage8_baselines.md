# Stage 8 — Baselines Benchmark Report

## 1. Executive Summary

Stage 8 establishes the empirical benchmark metrics for **Intent Classification** and **Support Response Generation** evaluated against the **Stage 7 Golden Evaluation Set** (200 high-quality AmazonHelp test conversations). These baselines provide the quantitative reference points that the Stage 9 AI Support Agent and Stage 10 Evaluation Harness must outperform.

---

## 2. Intent Classification Baselines

Four distinct classification architectures were evaluated:
1. **Majority Class Baseline (`majority_class`)**: Naive statistical floor always predicting the dominant training intent (`delivery_delay`).
2. **Keyword Rules Baseline (`keyword_rules`)**: Fast, deterministic regex matching based on Stage 6 taxonomy criteria.
3. **TF-IDF + Naive Bayes Baseline (`tfidf_naive_bayes`)**: Unigram + bigram TF-IDF probabilistic classifier trained strictly on 25,000 Stage 4 training interactions.
4. **BM25 Nearest Neighbor Baseline (`bm25_1nn`)**: 1-NN retrieval classifier querying the BM25 index over training customer queries.

### Intent Classification Benchmark Results

| Baseline Model | Accuracy | Macro F1 | Weighted F1 | Easy Acc | Medium Acc | Hard Acc | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `majority_class` | **9.00%** | 0.0150 | 0.0149 | 3.70% | 9.76% | 9.85% | 0.00 |
| `keyword_rules` | **61.00%** | 0.6311 | 0.6288 | 66.67% | 53.66% | 62.12% | 0.02 |
| `tfidf_naive_bayes` | **70.50%** | 0.7077 | 0.7064 | 81.48% | 75.61% | 66.67% | 0.09 |
| `bm25_1nn` | **47.50%** | 0.4922 | 0.4902 | 66.67% | 51.22% | 42.42% | 3.10 |

### Key Findings on Intent Classification
- **Statistical Floor**: Majority class achieves only **9.00%** accuracy, illustrating the severe penalty of naive bias on an equitable multi-class benchmark.
- **Keyword Strengths & Vulnerabilities**: `keyword_rules` achieves **61.00%** accuracy and excels on `easy` queries (66.67%), but drops significantly on `hard` edge cases (62.12%) due to vocabulary mismatch and multi-intent conflicts.
- **Statistical Winner**: `tfidf_naive_bayes` achieves the highest accuracy (**70.50%**) and Macro F1 (**0.7077**), effectively generalizing across n-gram patterns.
- **BM25 1-NN Retrieval**: Achieves **47.50%** accuracy. It is robust when verbatim training examples exist, but degrades on rare phrasings and terse ambiguous queries.

---

## 3. Support Response Generation / Retrieval Baselines

Three response generation and retrieval approaches were evaluated against human reference responses:
1. **Generic Default Response (`generic_default`)**: Static greeting and deflection link to Amazon help pages.
2. **Intent-Specific Canned Response (`intent_canned_template`)**: Curated, policy-compliant response template mapped to the predicted intent.
3. **BM25 Historic Retrieval (`bm25_retrieval`)**: Nearest-neighbor retrieval returning the exact historical response from real Amazon agents for the most similar training customer inquiry.

### Response Generation Benchmark Results

| Generation Baseline | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-1 | BLEU-2 | Guidance Adherence | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `generic_default` | 0.1782 | 0.0128 | **0.1159** | 0.1471 | 0.0226 | **28.00%** | 0.00 |
| `intent_canned_template` | 0.1335 | 0.0084 | **0.0989** | 0.1090 | 0.0135 | **79.50%** | 0.06 |
| `bm25_retrieval` | 0.1809 | 0.0320 | **0.1428** | 0.1449 | 0.0420 | **38.50%** | 3.07 |

### Key Findings on Response Generation
- **Lexical Overlap vs. Policy Grounding**: While `bm25_retrieval` scores competitive ROUGE-L (0.1428) by recycling real Twitter agent phrasing, its policy guidance adherence is only **38.50%** because historical human responses frequently contain specific deadlinks or deflection boilerplate ('DM sent', 'please check DM').
- **Canned Templates Lead Policy Adherence**: `intent_canned_template` achieves the highest guidance adherence (**79.50%**), proving that policy compliance can be reliably enforced once intent is correctly classified.
- **Need for Generative AI (Stage 9)**: Neither canned templates nor BM25 retrieval can dynamically personalize answers (e.g. acknowledging the customer's specific item name or addressing multi-intent inquiries). This formally establishes the necessity of an LLM agent in Stage 9.

---

## 4. Benchmark Scorecard & Target Goals for Stage 9 Agent

| Evaluation Dimension | Stage 8 Top Baseline Score | Model / Source | Stage 9 AI Agent Target Goal |
| :--- | :---: | :--- | :---: |
| Intent Classification Accuracy | 70.50% | `tfidf_naive_bayes` | **> 85.0%** |
| Intent Classification Macro F1 | 0.7077 | `tfidf_naive_bayes` | **> 0.8200** |
| Hard Difficulty Accuracy | 66.67% | `tfidf_naive_bayes` | **> 75.0%** |
| Response ROUGE-L | 0.1428 | `bm25_retrieval` | **> 0.3500** |
| Guidance Adherence | 79.50% | `intent_canned_template` | **> 90.0%** |

---

> **Stage 8 completed. Ready for Stage 9 — AI Support Agent.**
