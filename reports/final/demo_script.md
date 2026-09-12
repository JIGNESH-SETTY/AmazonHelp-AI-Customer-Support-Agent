# Project Demonstration Script: AmazonHelp AI Support Agent

**Target Duration**: 3–5 Minutes  
**Target Audience**: Engineering Hiring Managers, Technical Interviewers, System Architects  

---

### [0:00 – 0:30] The Problem
> "In automated customer support, traditional LLM chatbots present serious enterprise risks: they hallucinate delivery promises, invent non-existent refund policies, or give generic deflections like 'Please DM us'. Our goal in this project was to build a production-grade, policy-grounded support agent for Amazon customer inquiries that provides verified guidance, prevents unauthorized commitments, and operates with sub-10ms latency."

### [0:30 – 1:00] Dataset & Data Pipeline
> "We began with the 2.8-million-tweet Twitter Customer Support dataset. In Stage 1 through 4, we reconstructed conversational trees, extracted 358,000 AmazonHelp tweets across 85,000 conversations, and enforced strict zero-leakage conversation-level splitting. In Stage 5 and 6, we quantitatively profiled brand quality and discovered a canonical 10-intent taxonomy. To evaluate scientifically, in Stage 7 we curated an immutable 200-dialogue Golden Evaluation Set stratified across easy, medium, and hard queries with verified gold response guidance."

### [1:00 – 1:30] System Architecture
> "Here is how the agent processes an inquiry in production:
> 1. **Normalization** sanitizes Twitter handles and extracts URLs.
> 2. **Hybrid Intent Understanding** combines calibrated TF-IDF Naive Bayes log-posteriors with domain keyword boosters and multi-intent detection.
> 3. **Context Extraction** pulls out order numbers, carriers, and assesses urgency.
> 4. **BM25 Retrieval** fetches relevant policy articles from our canonical support corpus.
> 5. **Policy & Safety Enforcement** validates that no unauthorized actions or guarantees are emitted.
> 6. **Template Generation** grounds the response in verified resolution steps."

### [1:30 – 2:30] Live Agent Demonstration
> *(Run: `python -m src.agent.cli --demo` in the terminal)*
>
> "Let's observe the live demo in action across six real-world scenarios:
> - **Scenario 1 (Delayed Delivery)**: Notice the agent identifies `delivery_delay` with 95% confidence, extracts context, retrieves verified tracking guidelines, and advises checking 'Your Orders' with zero date hallucination in just 3.5ms.
> - **Scenario 2 (Damaged Item & Refund)**: The customer has a broken item and requests a refund. The agent captures the primary intent `damaged_defective_item` while registering `returns_and_refunds` as a secondary intent, seamlessly appending return center guidance.
> - **Scenario 3 (Order Cancellation vs Prime)**: The customer writes: 'Is there a direct link to cancel my prime membership?' Despite mentioning 'Prime', our compound verb logic correctly identifies `order_cancellation` rather than getting distracted by the membership entity.
> - **Scenario 4 (Double Billing)**: Recognizes payment dispute, alerts customer never to share credit card details publicly, and directs them to secure invoice review."

### [2:30 – 3:15] Evaluation Results & Baselines
> "We established four rigorous baselines in Stage 8: Majority Class (9%), BM25 1-NN (47.5%), Keyword Rules (61%), and TF-IDF Naive Bayes (70.5%).
> Our initial Stage 9 Agent reached 79.00% accuracy and 90.00% guidance adherence.
> In our final optimized release, overall accuracy reached **82.00% (+11.50% over baseline)** and hard-query accuracy reached **79.55% (+12.88% over baseline)**.
> Crucially, Policy Safety Rate remained at **100.00%** with zero unauthorized claims, and average latency is **3.48ms**, well under enterprise SLAs."

### [3:15 – 4:00] Failure Analysis & Decision Log
> "We did not stop at evaluation. In Stage 11, we built an 18-category failure analysis engine. We discovered that our biggest failure mode was intent confusion between delayed delivery and missing delivered packages due to words like 'not received'.
> Following our engineering decision framework, we tested **DEC-002**, requiring explicit delivery markers before firing missing package advice. We also tested **DEC-003**, giving action verbs precedence over entity nouns in cancellation queries.
> In our regression framework, these two changes yielded a +3.00% absolute accuracy boost and +4.55% hard-query boost with zero regressions across safety, adherence, or latency."

### [4:00 – 4:30] Conclusion & Takeaways
> "In summary, this project demonstrates how to build a reliable, high-speed, safety-critical AI support system from messy raw data to rigorous evaluation and continuous improvement. The entire codebase is 100% reproducible offline, verified by 95 passing unit tests. Thank you!"
