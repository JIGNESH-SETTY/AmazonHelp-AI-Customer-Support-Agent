# Technical Interview Cheat Sheet: AmazonHelp AI Support Agent

This guide provides rigorous, mathematically grounded, and technically defensible answers to the 25 most critical interview questions regarding this project.

---

### 1. What problem does this project solve?
In automated customer support, naive generative bots suffer from hallucinations, fabricated actions (e.g. claiming a refund was issued), and brittle intent understanding on complex multi-part queries. This project builds a production-grade, policy-grounded customer support agent for Amazon customer inquiries that accurately identifies intent, retrieves canonical policy/procedural guidance, validates safety constraints, and escalates risky or highly emotional queries—all under a strict sub-10ms latency SLA.

### 2. Why did you choose this problem?
Customer support is one of the highest-value, highest-liability enterprise AI use cases. High deflection rates are useless if the agent hallucinates policies or incurs operational liabilities by promising unauthorized credits. Building an end-to-end system from raw Twitter/TWCS data to baselines, production agent, evaluation harness, failure analysis, and decision logging mirrors real-world senior SDE/AI engineering.

### 3. What is the architecture?
The runtime architecture operates as a deterministic pipeline:
1. **Normalization**: Regex stripping of Twitter handles, URL extraction, audit logging.
2. **Hybrid Intent Understanding**: Calibrated TF-IDF Naive Bayes log-posteriors fused with taxonomy-aligned discriminative keyword boosters and multi-intent detection.
3. **Context Extraction**: Regex entity recognition for Order IDs, product references, carrier tags, and rule-based sentiment/urgency scoring.
4. **Knowledge Retrieval**: BM25 scoring across canonical Amazon policy and support documentation chunks.
5. **Policy & Safety Validation**: Zero-tolerance deterministic regex guardrails that prevent unauthorized financial commitments, PII collection, or date hallucinations.
6. **Response Generation**: Canonical template slot-filling grounded strictly in retrieved policy guidance and extracted context.
7. **Escalation Layer**: Automated routing to human tier for high-severity sentiment, legal/executive threats, or low classification confidence.

### 4. How did you process the dataset?
We started with the Twitter Customer Support (TWCS) dataset (2.8M tweets). We filtered for AmazonHelp (358,973 tweets across 85,087 conversations), cleaned broken references, extracted customer-agent turns into structured JSONL pairs, and conducted multi-criteria profiling (resolution rate, deflection rate, conversation depth) to confirm AmazonHelp as the highest-quality candidate brand.

### 5. How did you identify intents?
Rather than inventing arbitrary categories, we performed automated lexical and semantic clustering over high-quality AmazonHelp resolution pairs, discovering a canonical 10-intent taxonomy (e.g., `delivery_delay`, `missing_delivered_package`, `returns_and_refunds`, `order_cancellation`, `prime_membership`, etc.) plus an explicit `unknown_or_ambiguous` fallback.

### 6. Why did you create a Golden Evaluation Set?
Evaluating against massive raw Twitter data is deeply flawed: raw tweets contain noise, deflection loops ("Please DM us"), and customer sarcasm. We curated an immutable, stratified 200-conversation Golden Evaluation Set with verified ground truth, gold response guidance, and difficulty tiers (Easy, Medium, Hard).

### 7. Why not evaluate directly on the full dataset?
Evaluating directly on raw datasets introduces three major issues: (1) Label noise—over 60% of raw tweets are generic deflections ("DM us your account number"); (2) Evaluation leakage—evaluating on un-sanitized data rewards repeating historical Twitter deflection phrasing rather than actual operational resolution; (3) Lack of standardized criteria—the Golden Set provides strict response guidance against which guidance adherence can be objectively scored.

### 8. What baselines did you build?
To ensure scientific rigor, we established four baselines in Stage 8:
- **Majority Class Classifier**: 9.00% accuracy.
- **BM25 1-Nearest Neighbor**: 47.50% accuracy.
- **Keyword Rule-Based Baseline**: 61.00% accuracy.
- **TF-IDF + Naive Bayes Classifier**: 70.50% accuracy (Macro F1: 0.7077).

### 9. Why TF-IDF + Naive Bayes?
Multinomial Naive Bayes with sublinear TF-IDF weighting ($\text{tf} = 1 + \ln(\text{count})$) provides strong log-posterior calibration, handles out-of-vocabulary terms gracefully via Laplace smoothing, trains in under 2 seconds on 25,000 interactions, and infers in 0.09ms with zero GPU/API costs.

### 10. Why BM25?
BM25 (Best Matching 25) with $k_1=1.5$ and $b=0.75$ is the enterprise standard for lexical retrieval. It provides term-frequency saturation (preventing keyword-stuffed messages from dominating) and document length normalization, making it superior to cosine similarity over sparse TF-IDF vectors.

### 11. Why does BM25 retrieval sometimes fail?
BM25 fails on vocabulary mismatch (synonymy) and secondary entity distraction. If a customer writes "My Echo Dot hasn't arrived," BM25 may match chunks about Echo Dot configuration rather than shipping delays if "Echo Dot" has a higher inverse document frequency (IDF) than "arrived".

### 12. How does the AI agent work?
The agent combines statistical machine learning with deterministic operational rules. It feeds preprocessed customer text into the hybrid intent classifier, extracts entities, retrieves relevant policy chunks via BM25, selects a verified support template, validates the candidate output against deterministic policy guardrails, and evaluates whether escalation is required.

### 13. How does multi-intent handling work?
The classifier computes normalized probabilities across all taxonomy classes. If a secondary class exceeds 35% probability or contains explicit secondary triggers (e.g. "cancel" + "refund"), both intents are recorded. The response generator prioritizes the primary intent and appends modular guidance snippets addressing the secondary intent.

### 14. How do you prevent hallucinations?
We employ three strict layers:
1. **Template Grounding**: Responses are built from verified support response templates rather than free-form generative LLM decoding.
2. **Context Binding**: Variables (URLs, order IDs) are inserted strictly from verified metadata.
3. **Deterministic Regex Guardrails**: Post-generation regex filters scan for unverified transactional claims ("I have refunded", "I have cancelled") and block the message if an unverified claim is detected.

### 15. How does policy grounding work?
Every intent is mapped to explicit operational rules (e.g. `order_cancellation` requires checking if the item has already shipped; `missing_delivered_package` requires waiting 36 hours and checking mailrooms). Candidate responses must contain the corresponding verified URL and required action steps before emission.

### 16. When does the system escalate?
Escalation is triggered if:
- Sentiment is detected as highly frustrated/angry.
- High-severity keywords match (e.g., "supervisor", "manager", "lawyer", "better business bureau").
- Classification confidence falls below operational threshold (0.45).
- An irremediable policy constraint is encountered.

### 17. How did you measure quality?
We built an evaluation harness measuring:
- **Classification**: Overall accuracy, 95% Bootstrap Confidence Intervals, Macro F1, Weighted F1, Confusion Matrix, Difficulty Breakdown.
- **Generation & Safety**: Guidance adherence, policy safety rate, unsupported action rate, hallucinated guarantee rate, ROUGE-1/2/L, BLEU-1/2.
- **Operations**: Escalation accuracy, unnecessary escalation rate, latency profile (Mean, Median, P95).

### 18. Why aren't ROUGE/BLEU sufficient?
ROUGE and BLEU only measure n-gram overlap against historical references. A response saying "I have refunded your $500 order" might score high ROUGE against a historical Twitter agent response, but in an automated bot it is a catastrophic operational hallucination. Domain-specific guidance adherence and policy safety rates are required.

### 19. What were the biggest failure modes?
In Stage 11, we identified:
1. Lexical confusion between `delivery_delay` and `missing_delivered_package` due to shared words like "not received".
2. Entity dominance in `order_cancellation` where mentions of "Prime" caused the model to predict `prime_membership` instead of cancellation.
3. Over-confidence on short ambiguous inputs (< 25 characters).

### 20. What did you improve after failure analysis?
We evaluated targeted improvements with regression guards:
- Constrained `missing_delivered_package` to require explicit delivery status markers ("says delivered", "marked delivered"), eliminating false lost-package advice on delayed parcels.
- Added compound cancellation verbs ("cancel my prime", "cancel subscription") to assert verb dominance over noun entities.
- Result: Accuracy increased from **79.00% to 82.00% (+3.00%)**, hard accuracy jumped from **75.00% to 79.55% (+4.55%)**, and Macro F1 rose from **0.7846 to 0.8204** with zero regressions.

### 21. How did you prevent data leakage?
1. Split assignment was done at the conversation level (not tweet level), ensuring no tweets from the same customer interaction appeared across train and test splits.
2. The Stage 7 Golden Set was isolated in a protected directory and verified immutable via SHA-256 manifest checks.
3. Training routines explicitly verify that zero Golden Set conversation IDs exist in the training corpus.

### 22. How reproducible is the evaluation?
100% reproducible. The code uses fixed random seeds for bootstrap resampling, runs fully offline with zero external API calls, and produces identical metric outputs across runs.

### 23. What would you improve next?
1. **Multi-Turn State Machine**: Add session memory to collect missing order numbers across turns.
2. **Hybrid Dense-Sparse Retrieval**: Add bi-encoder embeddings (e.g. MiniLM) to complement BM25 for synonym matching.
3. **Real-Time ERP Integration**: Connect verified account auth tokens to check live order status via REST APIs.

### 24. What was your personal contribution?
Architected the end-to-end system: raw TWCS conversation reconstruction, quantitative brand selection, intent discovery and taxonomy definition, Golden Evaluation Set curation, baseline benchmarking, agent pipeline design, evaluation harness implementation, failure taxonomy, and decision log framework.

### 25. What was the hardest technical problem?
Disambiguating overlapping multi-intent requests without sacrificing latency. When a user asks "My Prime delivery is 3 days late, should I cancel Prime?", traditional classifiers struggle because both intents are active. Designing compound precedence rules where action verbs dominate entity nouns while executing in under 3.5ms was the key engineering breakthrough.
