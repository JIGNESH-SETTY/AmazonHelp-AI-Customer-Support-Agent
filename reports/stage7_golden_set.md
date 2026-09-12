# Stage 7 — Golden Evaluation Set Report

## 1. What Was Implemented

Stage 7 constructed the **Golden Evaluation Set**, a trusted, protected, and fully reproducible benchmark of 200 customer-support interactions for `AmazonHelp`. The benchmark serves as the grounded evaluation standard for downstream stages:
- **Stage 8 — Baselines**: Evaluating zero-shot, few-shot, and keyword retrieval models.
- **Stage 9 — AI Support Agent**: Evaluating conversational problem-solving, policy adherence, and tool use.
- **Stage 10 — Evaluation Harness**: Automated multi-metric scoring (intent accuracy, response relevance, hallucination resistance).
- **Stage 11 — Failure Analysis**: Granular qualitative diagnosis across error categories.
- **Stage 12 — Final Report**: Objective benchmark scorecards and trade-off analyses.

Key capabilities delivered in Stage 7:
1. **Stratified Sampling Pipeline**: Deterministic sampling (`RANDOM_SEED = 42`) across all 10 canonical intents from Stage 6 + controlled ambiguous/edge cases.
2. **Rigorous Quality & De-duplication Filter**: Eliminated boilerplate noise (< 25 chars, 'DM sent', 'check DM'), duplicate conversations, and near-duplicate complaints (token Jaccard similarity >= 0.80).
3. **Objective Difficulty Grading**: Deterministic assignment of `easy`, `medium`, and `hard` tiers reflecting conversational ambiguity, customer frustration, multi-intent friction, and hallucination risk.
4. **Actionable Evaluation Metadata**: Augmented each record with explicit `expected_behavior` and `gold_response_guidance` to assess semantic response validity beyond superficial lexical matching.
5. **Split Integrity Guarantee**: Sourced 100% of candidate pairs from the Stage 4 test split (`data/processed/splits/test/`), proving zero leakage from training and validation pools.
6. **Validation Runner & Integrity Suite**: Standalone validator (`validate_golden_set.py`) and automated unit tests (`tests/test_stage7.py`).

---

## 2. Input Dataset Used

The evaluation benchmark is compiled strictly from the following protected Stage 4 test partitions:
- `data/processed/splits/test/amazon_resolution_pairs_test.jsonl` (2,381 test pairs)
- `data/processed/splits/test/amazon_clarification_pairs_test.jsonl` (1,611 test pairs)
- `data/processed/splits/test/amazon_escalation_pairs_test.jsonl` (4,785 test pairs)
- `data/processed/amazonhelp_intent_labels.jsonl` (Stage 6 test labels; 8,777 test pairs cross-referenced)
- `data/intent_taxonomy.json` (Stage 6 canonical taxonomy v1.0 defining 10 support intents + fallback category)
- `data/selected_brand.json` (Stage 5 brand selection artifact designating `AmazonHelp`)

---

## 3. Golden-Set Construction Strategy

Rather than taking a naive random sample that would result in over 70% delivery tracking complaints and neglect critical edge cases, Stage 7 adopted a **stratified, quota-driven strategy**:
- **Target Population**: 200 examples total.
- **Canonical Intent Quota**: Exactly 18 examples for each of the 10 canonical intents (180 examples).
- **Ambiguous / Edge-Case Quota**: Exactly 20 examples classified as `unknown_or_ambiguous` or multi-intent conflicts to test agent restraint, clarification prompting, and hallucination resistance.
- **Reproducibility**: Governed by deterministic pseudorandom seed `RANDOM_SEED = 42`.
- **Contextual Sufficiency**: Selected pairs contain full customer inquiries and verified human support responses from real Amazon customer service agents.

---

## 4. Quality Filters

Before entering the candidate pool, every candidate was passed through a multi-stage filter:

| Filter Stage | Threshold / Condition | Candidates Rejected |
| :--- | :--- | :--- |
| Raw Test Pairs Ingested | Single-turn test split pairs | 8,777 total |
| Minimum Text Length | Customer text length < 25 characters | 174 |
| Boilerplate Noise | Matches noise patterns ('DM sent', 'check DM', handle-only) | 155 |
| Missing Support Response | Empty or whitespace-only agent reference response | 0 |
| Intent Label Match | Successfully matched with Stage 6 taxonomy test label | 8,448 passed |

---

## 5. Intent Coverage

The benchmark achieves **100% coverage of all 10 canonical intents** (10/10) plus the controlled ambiguous category:

| Intent ID | Intent Name | Golden Set Count | Percentage |
| :--- | :--- | :---: | :---: |
| `delivery_delay` | Delivery Delay & Tracking | 18 | 9.0% |
| `missing_delivered_package` | Missing Delivered Package | 18 | 9.0% |
| `returns_and_refunds` | Returns & Refunds | 18 | 9.0% |
| `damaged_defective_item` | Damaged or Defective Item | 18 | 9.0% |
| `order_cancellation` | Order Cancellation & Modification | 18 | 9.0% |
| `payment_and_billing` | Payment, Charges & Billing | 18 | 9.0% |
| `prime_membership` | Amazon Prime & Subscriptions | 18 | 9.0% |
| `account_access_security` | Account Access & Security | 18 | 9.0% |
| `digital_services_technical` | Digital Devices & Services | 18 | 9.0% |
| `service_complaint_escalation` | Customer Service Escalation | 18 | 9.0% |
| `unknown_or_ambiguous` | Unknown or Ambiguous | 20 | 10.0% |

**Multi-Intent / Secondary Intent Indicators**: 18 examples (9.0%) contain explicit secondary problem signals (e.g., late shipment combined with refund request, or broken delivery combined with cancellation).

---

## 6. Brand Distribution

- **Brand**: `AmazonHelp`
- **Representation**: 100% (200/200 examples).
- **Consistency**: Verified against `data/selected_brand.json`.

---

## 7. Difficulty Distribution

Difficulty was graded deterministically based on customer emotional intensity, query ambiguity, turn complexity, and hallucination risk:
- **Easy**: Clear, focused customer issue, single distinct intent keyword, high confidence, resolution response type, low sentiment friction.
- **Medium**: Moderate length (> 180 chars), multiple order attributes, clarification required, or moderate confidence.
- **Hard**: High emotional intensity (customer anger/frustration), escalation response, conflicting multi-intent signals, or ambiguous requests posing acute hallucination danger if the agent assumes unstated facts.

| Difficulty Tier | Count | Percentage | Primary Characteristics |
| :--- | :---: | :---: | :--- |
| `hard` | 132 | 66.0% | Frustrated customers, escalation demands, multi-intent conflicts, ambiguous queries |
| `medium` | 41 | 20.5% | Multi-clause inquiries, clarification required, product setup questions |
| `easy` | 27 | 13.5% | Straightforward inquiries with explicit keywords and clear single resolutions |

---

## 8. Duplicate / Leakage Prevention

To guarantee the benchmark's trustworthiness as a test standard:
1. **Conversation ID Collisions**: Enforced exact uniqueness on `conversation_id` across all 200 examples (4 collisions rejected).
2. **Exact Message Duplication**: Enforced exact text uniqueness (0 duplicates rejected).
3. **Near-Duplicate Protection**: Evaluated token Jaccard similarity between candidate messages within the same intent pool with a cutoff of >= 0.80 (0 near-duplicates rejected).
4. **Zero Split Leakage**: Checked all 200 golden conversation IDs against the 61,843 training conversations and 13,205 validation conversations.
   - **Train Leakage**: 0 (0.00%)
   - **Validation Leakage**: 0 (0.00%)

---

## 9. Validation Results

Automated validation executed via `src/evaluation/validate_golden_set.py`:

```text
========================================
GOLDEN SET VALIDATION
========================================
Total examples: 200
Unique conversations: 200
Intents covered: 10/10 canonical (11 total)
Duplicate examples: 0
Duplicate conversations: 0
Missing required fields: 0
Invalid intents: 0
Invalid difficulty labels: 0
Train/Val leakage: 0
----------------------------------------
STATUS: PASS
========================================
```

All 12 unit tests in `tests/test_stage7.py` passed with 0 failures.

---

## 10. Files Created / Modified

### Files Created
- `src/evaluation/__init__.py`: Package initialization documenting dependencies on Stage 1-6 outputs.
- `src/evaluation/build_golden_set.py`: Stratified sampling pipeline, quality filtering, difficulty labeling, and guidance generator.
- `src/evaluation/validate_golden_set.py`: Standalone CLI validation suite producing structured PASS/FAIL audits.
- `src/evaluation/generate_report.py`: Automated documentation generator.
- `data/golden/golden_evaluation_set.jsonl`: Primary JSONL evaluation dataset (200 records).
- `data/golden/golden_evaluation_set.csv`: Tabular evaluation dataset export (200 rows).
- `data/golden/golden_set_manifest.json`: Machine-readable metadata and distribution audit.
- `reports/stage7_golden_set.md`: Comprehensive Stage 7 documentation report.
- `tests/test_stage7.py`: Unit test suite verifying schema, uniqueness, coverage, and split integrity.

### Files Modified
- None (Stage 1-6 artifacts and source datasets were preserved unmodified).

---

## 11. How to Run the Pipeline

To rebuild, validate, and test the Golden Evaluation Set from scratch:

```bash
# 1. Generate the Golden Evaluation Set artifacts
.venv\Scripts\python -m src.evaluation.build_golden_set

# 2. Run the dedicated validation script
.venv\Scripts\python -m src.evaluation.validate_golden_set

# 3. Run the automated test suite
.venv\Scripts\python -m unittest tests/test_stage7.py -v

# 4. Run full project regression tests
.venv\Scripts\python -m unittest tests/test_stage5.py tests/test_stage6.py tests/test_stage7.py -v
```

---

## 12. Limitations

1. **Public Social Media Context**: Tweets naturally skew toward concise, informal wording and customer frustration compared to long-form email tickets.
2. **Heuristic Difficulty Grading**: Difficulty tiers are derived deterministically from text features, sentiment markers, and response behavior rather than human consensus panels.
3. **Reference Response Brevity**: Reference agent tweets often include Twitter handles, shortened links, or requests to move to direct messages due to Twitter's 280-character limit.
4. **Automated Intent Prioritization**: In multi-intent sentences, single-label primary classification follows Stage 6 priority rules, though secondary intent is captured explicitly for nuanced evaluation.

---

## 13. Why This Golden Set Is Suitable for Later Support Agent Evaluation

The Stage 7 Golden Evaluation Set fulfills all prerequisite criteria for rigorous, unbiased agent evaluation in Stages 8–10:
1. **Zero Contamination**: Strictly drawn from the Stage 4 test split with confirmed zero leakage into training data.
2. **Equitable Representation**: Avoids delivery bias by giving equal representation (18 examples each) across all 10 canonical support problem types.
3. **Grounded Anti-Hallucination Criteria**: Provides clear semantic response guidance for every intent, defining what the agent must NOT fabricate (e.g., promising unverified tracking dates or claiming refunds were already issued).
4. **Graded Complexity**: Evaluates the agent across simple inquiries (easy), detailed technical questions (medium), and hostile, ambiguous multi-issue complaints (hard).
5. **Edge-Case Stress Testing**: 20 ambiguous queries specifically measure whether the agent asks clarifying questions rather than generating hallucinated responses.

> **Stage 7 completed. Ready for Stage 8 — Baselines.**
