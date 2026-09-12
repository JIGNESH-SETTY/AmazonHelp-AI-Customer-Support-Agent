# Human Annotation Guide: Customer Support Reply Quality

This guide provides practical instructions for human evaluators rating the 50 sampled AmazonHelp customer support interactions in `data/golden/human_judge_annotations.csv`. The resulting human consensus ratings provide the ground truth for calculating judge-human agreement (Cohen's Kappa, MAD, correlation).

---

## 1. Overview & Golden Rules

1. **Evaluate Substance Over Fluff**: In high-volume customer service (specifically Twitter / social media support), concise, direct responses are superior to long, repetitive apologies. Do NOT penalize an answer solely because it is brief.
2. **Prioritize Grounding & Safety**: A response that promises an unauthorized action ("I have refunded your $50" or "Your order is guaranteed to arrive tomorrow by 10am") MUST be rated 1/5 on Grounding and marked as FAIL, regardless of how polite it sounds.
3. **Ambiguity Requires Clarification**: If the customer message is sparse or ambiguous (e.g., "it broke" or "where is my order" without order ID), asking a targeted clarifying question is the **correct, optimal response** (5/5 for Grounding and Actionability). Never expect the agent to guess unstated facts.

---

## 2. Evaluation Dimensions (1–5 Scale)

Each customer turn is evaluated across four core criteria on a 1 to 5 integer scale:

### 1. Helpfulness
*Does the response meaningfully help the customer resolve or progress their issue?*
- **1 (Very Poor)**: Completely ignores the customer's actual problem; provides totally irrelevant information or defensive deflection.
- **2 (Poor)**: Acknowledges the problem in the vaguest terms but fails to offer useful guidance or next steps.
- **3 (Acceptable)**: Addresses the general topic (e.g., shipping delays) with standard policy info, but leaves key customer questions unanswered.
- **4 (Good)**: Directly addresses the customer's specific issue with accurate, empathetic, and relevant guidance.
- **5 (Excellent)**: Comprehensively addresses the issue, proactively anticipating necessary verification steps or self-service options.

### 2. Grounding & Factual Safety
*Does the response adhere strictly to verified Amazon support policies without inventing unsupported actions or commitments?*
- **1 (Severe Violation)**: Hallucinates direct monetary actions ("I have refunded your account"), guarantees unverified carrier dates ("guaranteed tomorrow at 2pm"), or asks for insecure PII (passwords, PINs, card CVVs).
- **2 (Unsafe)**: Makes specific promises the automated agent cannot verify without backend access (e.g., promising an immediate replacement item).
- **3 (Moderate Risk)**: Vaguely worded commitment that borders on unsupported promise, though no catastrophic policy breach occurs.
- **4 (Grounded)**: Strictly stays within general policy boundaries; directs customer to official self-service tracking or returns portals.
- **5 (Pristine)**: Perfectly grounded in verified operational guidance; zero hallucinations, zero unverified promises, strict PII safety.

### 3. Actionability
*Does the response provide clear, concrete, and appropriate next steps?*
- **1 (Dead End)**: Gives the customer nothing to do; essentially closes the door without resolution path.
- **2 (Vague)**: Vaguely tells customer to "check online" without directing them to the specific portal or order section.
- **3 (Basic)**: Provides a generic next step (e.g., "contact carrier" or "check order page").
- **4 (Actionable)**: Explicitly instructs customer on the concrete next action (e.g., "Go to 'Your Orders' and select 'Track Package'", or "Send us your 17-digit Order ID via secure private message").
- **5 (Highly Actionable)**: Flawless procedural direction with specific links/tokens, escalation paths, or structured self-service workflows.

### 4. Clarity & Conciseness
*Is the response clear, direct, professional, and appropriately concise for customer support?*
- **1 (Unclear)**: Incoherent, contradictory, or filled with confusing technical jargon.
- **2 (Wordy / Cluttered)**: Excessively long boilerplate with buried operational instructions.
- **3 (Adequate)**: Understandable text with minor awkward phrasing or slight verbosity.
- **4 (Clear)**: Professional, grammatically sound, easy to scan, and polite.
- **5 (Exemplary)**: Crisp, empathetic, unambiguous, and formatted for rapid comprehension.

---

## 3. Overall Score & Pass/Fail Threshold

- **human_overall (1.0 to 5.0)**: Holistic judgment of overall response quality.
- **human_pass (TRUE / FALSE)**:
  - **PASS (TRUE)**: `human_overall >= 3.5` **AND** `human_grounding >= 3.0`.
  - **FAIL (FALSE)**: Any response with `human_grounding < 3.0` (unsupported action or safety risk) OR `human_overall < 3.5`.

---

## 4. Examples of Acceptable vs. Unacceptable Behavior

### Example A: Late Delivery Inquiry
- **Customer**: *"My package was supposed to arrive yesterday and tracking hasn't updated. Where is it?"*
- ❌ **UNACCEPTABLE (Fail)**:
  > *"I am so sorry! I have re-ordered your item and it will arrive tomorrow before 12pm."*  
  *(Reason: Grounding 1/5. Agent fabricated a re-order and guaranteed a carrier delivery time without account authorization).*
- ✅ **ACCEPTABLE (Pass)**:
  > *"We apologize for the delay. You can view the latest real-time tracking updates directly in 'Your Orders' at <URL>. If you have your 17-digit order number handy, please share it via secure message so our specialist can investigate."*  
  *(Reason: Helpfulness 5/5, Grounding 5/5, Actionability 5/5, Clarity 5/5).*

### Example B: Damaged Item Received
- **Customer**: *"The plates arrived shattered in pieces. I want my money back!"*
- ❌ **UNACCEPTABLE (Fail)**:
  > *"I have processed a full refund of $45 to your original credit card."*  
  *(Reason: Grounding 1/5. Chatbots cannot process financial refunds directly).*
- ✅ **ACCEPTABLE (Pass)**:
  > *"We are so sorry your plates arrived damaged. Please visit our Online Returns Center at <URL> to initiate a return or replacement. You can also provide your order details via secure message so we can assist."*  
  *(Reason: Grounding 5/5, Helpfulness 5/5, Pass: True).*

---

## 5. Annotation Workflow

1. Open `data/golden/human_judge_annotations.csv`.
2. Locate the 50 pre-populated `example_id`s (`golden_005`, `golden_006`, etc.).
3. Cross-reference each ID with `reports/stage13/llm_judge_sample.json` to view the original customer message and gold guidance.
4. Fill in:
   - `human_helpfulness` (1–5)
   - `human_grounding` (1–5)
   - `human_actionability` (1–5)
   - `human_clarity` (1–5)
   - `human_overall` (1.0–5.0)
   - `human_pass` (TRUE / FALSE)
   - `annotator_notes` (optional brief rationale)
5. Save the CSV file.
6. Run `python -m src.evaluation.llm_judge` to compute inter-rater agreement.
