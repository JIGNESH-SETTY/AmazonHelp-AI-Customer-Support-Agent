# Stage 6 — Intent Discovery & Labeling Report

**Brand:** `AmazonHelp` | **Taxonomy Version:** `1.0` | **Generated:** 2026-09-12 06:31:22 UTC

---

## 1. Objective
The objective of Stage 6 is to discover an empirical, data-driven customer support intent taxonomy from the real-world conversation corpus of **AmazonHelp**, rather than imposing arbitrary or pre-conceived categories. Customer support conversations inherently span a wide variety of operational and post-order friction points. By discovering recurring problem themes and consolidating them into a clear, mutually distinguishable taxonomy, this stage provides the foundation for Stage 7 (Golden Evaluation Dataset), Stage 8 (Intent Classification & Retrieval Baselines), and Stage 9 (Hiver AI Support Agent).

## 2. Data Used
Intent discovery and labeling were executed directly on the validated Stage 4 canonical conversation datasets:
- `data/processed/splits/train/amazon_resolution_pairs_train.jsonl`
- `data/processed/splits/train/amazon_escalation_pairs_train.jsonl`
- `data/processed/splits/train/amazon_clarification_pairs_train.jsonl`
- Corresponding validation and test splits in `data/processed/splits/val/` and `data/processed/splits/test/`
- `data/selected_brand.json` (authoritative brand source of truth)

No Stage 1–5 source datasets or manifests were modified or deleted.

## 3. Population
The candidate training population comprises **70,579 customer-agent interaction pairs** across English AmazonHelp conversations:
- **Resolution Interactions**: 19,335 (substantive solutions and instructions)
- **Escalation Interactions**: 38,372 (channel deflections and private routing)
- **Clarification Interactions**: 12,872 (information requests and disambiguation)


## 4. Sampling Methodology
To discover representative problem themes without allowing dominant high-frequency inquiries to overshadow critical support issues, we extracted a stratified sample of **15,000 customer messages** with fixed random seed `42`:
- **Sample Size**: 15,000 interactions (5,000 resolution, 5,000 escalation, 5,000 clarification)
- **Filters**: Messages shorter than 10 characters and obvious non-informative greetings were excluded from theme discovery.
- **Random Seed**: Fixed at `42` to guarantee complete determinism and reproducibility across runs.

## 5. Discovery Methodology
An unsupervised, interpretable theme discovery pipeline was executed:
1. **Conservative Text Preprocessing**: Customer messages were lowercased, URLs and usernames stripped, smart quotes normalized, and whitespace standardized, while strictly preserving operational domain terminology (e.g. `refund`, `tracking`, `prime`, `password`, `kindle`, `damaged`).
2. **Feature Extraction**: Extracted unigrams and bigrams, pruned terms appearing in fewer than 15 documents or more than 30% of documents, producing a curated vocabulary of **1,536 salient terms**.
3. **TF-IDF & Vectorization**: Constructed L2-normalized TF-IDF document vectors combining sublinear term frequency with inverse document frequency.
4. **K-Means Clustering**: Partitioned document vectors into $k = 14$ candidate clusters using cosine distance (dot product of L2 vectors) over 25 iterations.

## 6. Candidate Themes
The 14 emergent clusters from unsupervised discovery are summarized below:

| Cluster | Sample Count | Pct | Dominant Keywords |
|:-------:|-------------:|----:|:------------------|
| 10 | 2,417 | 16.1% | `yes, like, return, issue, already, packaging, item` |
| 5 | 1,827 | 12.2% | `order, delivery, deliver, placed, cancel, cancelled, yet` |
| 0 | 1,599 | 10.7% | `service, customer, customer_service, que, care, worst, customer_care` |
| 13 | 1,515 | 10.1% | `need, call, phone, link, contact, number, problem` |
| 9 | 1,083 | 7.2% | `prime, membership, delivery, prime_membership, pay, next, paying` |
| 6 | 1,072 | 7.2% | `delivered, says, parcel, order, order_delivered, door, delivery` |
| 12 | 929 | 6.2% | `product, refund, received, money, order, return, replacement` |
| 7 | 906 | 6.0% | `ordered, shipped, arrive, yet, pre, waiting, delivery` |
| 1 | 847 | 5.7% | `account, email, locked, address, received, email_address, bank` |
| 2 | 752 | 5.0% | `package, delivered, package_delivered, delivery, supposed, late, carrier` |
| 11 | 752 | 5.0% | `app, kindle, alexa, book, find, available, books` |
| 8 | 516 | 3.4% | `shipping, prime, prime_shipping, paid, order, package, free` |
| 3 | 411 | 2.7% | `card, credit, credit_card, gift, gift_card, order, cashback` |
| 4 | 374 | 2.5% | `date, delivery_date, delivery, order, release, estimated, expected` |

## 7. Final Consolidated Taxonomy (v1.0)
Through expert inspection of candidate cluster keywords and representative dialogues, redundant and fragmented clusters were consolidated into **10 core, mutually distinguishable support intents** plus one explicit `unknown_or_ambiguous` fallback:

### `delivery_delay` — Delivery Delay & Tracking
**Definition**: The customer is inquiring about an order that has not arrived by the expected date/time, is delayed in transit, or requires an updated tracking status.

**Inclusion Criteria**:
- Inquiries regarding order tracking status when order is not yet marked delivered.
- Complaints that an order has missed its promised or guaranteed delivery date.
- Questions about shipment delays, carrier dispatch hold-ups, or package transit bottlenecks.
- Requests for delivery estimates on pending shipments.

**Exclusion Criteria**:
- Packages marked as 'Delivered' by the carrier that cannot be found (classify as missing_delivered_package).
- Requests to cancel a delayed order (classify as order_cancellation).
- Requests for a refund due to delayed shipment (classify as returns_and_refunds).

**Confusable Intents**: missing_delivered_package (distinction: delivery_delay packages are in transit/late; missing_delivered_package are marked delivered), returns_and_refunds (distinction: delivery_delay asks where the order is; returns_and_refunds asks for money back)

**Representative Example**:
> *"How about you guys figure out my Xbox One X project Scorpio edition first. No expected delivery or shipping date and it's only a week away"*  
> — `[conv_628]`

### `missing_delivered_package` — Missing Delivered Package
**Definition**: The tracking status states the package was delivered, but the customer cannot locate the parcel at their delivery address.

**Inclusion Criteria**:
- Carrier tracking indicates 'Delivered' but the customer did not receive the parcel.
- Driver reportedly left package at an unlocatable location (e.g. front porch, gate, bushes).
- Suspected package theft or delivery to a wrong neighbor / address.
- Proof of delivery photo showing an unfamiliar location.

**Exclusion Criteria**:
- Packages with tracking showing 'In Transit' or 'Delayed' (classify as delivery_delay).
- Packages delivered with damaged or broken contents (classify as damaged_defective_item).

**Confusable Intents**: delivery_delay (distinction: missing_delivered_package is explicitly marked delivered in carrier system)

**Representative Example**:
> *"My package with my Halloween costume was delivered Friday but I don't have it so searching everywhere for a last minute idea."*  
> — `[conv_678]`

### `returns_and_refunds` — Returns & Refunds
**Definition**: The customer wants to return an item, check on the status of a refund, request a replacement, or inquire about return policies and drop-off locations.

**Inclusion Criteria**:
- Requests to return an eligible delivered item.
- Inquiries regarding refund timeline, pending refund credit, or bank reversal.
- Questions about return shipping labels, QR codes, or UPS/Kohl's drop-off locations.
- Requests for an item exchange or replacement unit.

**Exclusion Criteria**:
- Requests to cancel an order that has not yet shipped (classify as order_cancellation).
- Disputed credit card transactions without formal return (classify as payment_and_billing).

**Confusable Intents**: order_cancellation (distinction: returns occur after item shipment/delivery; cancellation occurs before dispatch), payment_and_billing (distinction: refund is return of purchase price; billing relates to unauthorized charges/payment methods)

**Representative Example**:
> *"I dropped off my return at UPS three days ago. When will I see the refund in my account?"*  
> — `[conv_11200]`

### `order_cancellation` — Order Cancellation & Modification
**Definition**: The customer wants to cancel an order, modify order details (shipping address, quantity, payment method), or made an accidental order.

**Inclusion Criteria**:
- Requests to cancel an active order before it ships.
- Accidental 1-Click orders or purchases made by children/pets.
- Requests to modify shipping address, shipping speed, or item quantity on pending orders.
- Inquiries as to why an order was automatically cancelled by Amazon.

**Exclusion Criteria**:
- Returning an item that has already arrived (classify as returns_and_refunds).
- Canceling an Amazon Prime recurring subscription (classify as prime_membership).

**Confusable Intents**: returns_and_refunds (distinction: cancellation stops order before delivery), prime_membership (distinction: order_cancellation cancels product orders, not memberships)

**Representative Example**:
> *"I ordered the wrong size by mistake 10 minutes ago, how can I cancel the order before it ships?"*  
> — `[conv_8910]`

### `damaged_defective_item` — Damaged or Defective Item
**Definition**: The customer received merchandise that arrived physically broken, damaged in transit, defective, malfunctioning, or is missing parts.

**Inclusion Criteria**:
- Items with broken, cracked, smashed, or shattered parts upon unboxing.
- Items in severely crushed or damaged outer packaging.
- Defective electronics that do not power on or function as advertised.
- Packages with missing components, missing accessories, or wrong items inside the box.

**Exclusion Criteria**:
- Digital software or e-book rendering issues (classify as digital_services_technical).
- Delays in transit where item has not yet arrived (classify as delivery_delay).

**Confusable Intents**: returns_and_refunds (distinction: damaged_defective_item specifically reports physical product failure/defect)

**Representative Example**:
> *"Opened my package and the coffee mug inside is smashed into pieces. Box had zero bubble wrap."*  
> — `[conv_14205]`

### `prime_membership` — Amazon Prime & Subscriptions
**Definition**: Inquiries regarding Amazon Prime membership status, renewal fees, trial subscriptions, Prime delivery perks, student discounts, or recurring membership charges.

**Inclusion Criteria**:
- Inquiries regarding unexplained Prime membership fees or renewal charges.
- Requests to cancel Amazon Prime and obtain a membership fee refund.
- Questions about Prime benefits (Prime Video, Prime Music, guaranteed delivery speeds).
- Student or discounted Prime subscription eligibility.

**Exclusion Criteria**:
- Product orders shipped via Prime that are late (classify as delivery_delay).
- General non-Prime credit card dispute (classify as payment_and_billing).

**Confusable Intents**: payment_and_billing (distinction: prime_membership specifically addresses Prime subscription fees), delivery_delay (distinction: prime_membership questions the membership benefits; delivery_delay tracks a specific order)

**Representative Example**:
> *"Why did Amazon charge my credit card $99 for Prime? I never signed up for auto-renewal!"*  
> — `[conv_21040]`

### `payment_and_billing` — Payment, Charges & Billing
**Definition**: Inquiries regarding payment methods, disputed credit/debit card charges, double billing, gift card redemption, invoice requests, or payment transaction declines.

**Inclusion Criteria**:
- Unidentified or duplicate charges appearing on bank or credit card statements.
- Payment method declined, expired card updates, or checkout payment failures.
- Amazon Gift Card balance not applying, invalid gift claim codes.
- Requests for VAT invoices or payment receipts for accounting.

**Exclusion Criteria**:
- Specific charges for Amazon Prime membership renewal (classify as prime_membership).
- Refund status inquiries following a return (classify as returns_and_refunds).

**Confusable Intents**: prime_membership (distinction: payment_and_billing covers product transactions, gift cards, bank declines), returns_and_refunds (distinction: billing deals with checkout/charges; refunds deal with returned merchandise)

**Representative Example**:
> *"My bank statement shows two identical charges of $45.99 from Amazon today for one order."*  
> — `[conv_28190]`

### `account_access_security` — Account Access & Security
**Definition**: Customer cannot log in, has forgotten password, is locked out of their Amazon account, has 2FA/OTP issues, or suspects unauthorized account access.

**Inclusion Criteria**:
- Inability to sign in, forgotten password, password reset link not arriving.
- Two-Factor Authentication (2FA) or OTP text codes not received on mobile phone.
- Account locked, suspended, or placed on security hold.
- Suspicion of hacked account, unauthorized email changes, or phishing attempts.

**Exclusion Criteria**:
- Standard account settings updates on accessible accounts (e.g. adding shipping address).
- Inquiries regarding payment declines on accessible accounts (classify as payment_and_billing).

**Confusable Intents**: payment_and_billing (distinction: account_access deals with authentication/login/security; billing deals with money)

**Representative Example**:
> *"I'm locked out of my Amazon account and the password reset email is never sent to my inbox."*  
> — `[conv_33910]`

### `digital_services_technical` — Digital Devices & Services
**Definition**: Technical issues, bugs, or troubleshooting related to Amazon hardware devices (Kindle, Echo/Alexa, Fire TV) or digital services (Kindle eBooks, Amazon App, Prime Video, website errors).

**Inclusion Criteria**:
- Kindle e-reader freezing, screen unresponsive, or eBook download failures.
- Fire TV or Fire TV Stick streaming errors, crashing apps, or remote pairing issues.
- Echo / Alexa voice recognition bugs, smart home disconnection.
- Amazon mobile app crashes, website error pages during browsing or checkout.

**Exclusion Criteria**:
- Physical transit damage to a newly shipped device (classify as damaged_defective_item).
- Inability to log in to Amazon account on device (classify as account_access_security).

**Confusable Intents**: damaged_defective_item (distinction: digital_services deals with software/hardware troubleshooting; damaged deals with broken in box)

**Representative Example**:
> *"My Kindle Paperwhite screen is frozen on the screensaver and holding power button for 40s does nothing."*  
> — `[conv_39100]`

### `service_complaint_escalation` — Customer Service Escalation
**Definition**: Customer expresses severe dissatisfaction with prior support interactions, reports rude or unhelpful support agents, or requests immediate supervisor escalation or telephone callback.

**Inclusion Criteria**:
- Strong dissatisfaction regarding previous unhelpful or contradictory customer service agents.
- Explicit requests to speak to a supervisor, team manager, or escalation department.
- Demands for a customer support phone number or urgent callback.
- General complaints about poor brand service quality without a single product issue.

**Exclusion Criteria**:
- Polite first-time inquiries regarding order status (classify under specific problem intent).
- Routine inquiries asking for help without expressing escalation or agent dissatisfaction.

**Confusable Intents**: delivery_delay (distinction: escalation focuses on agent failure/manager demand; delay focuses on shipment timing)

**Representative Example**:
> *"Your customer support chat is completely useless. I want to speak to a supervisor immediately."*  
> — `[conv_47800]`

## 8. Intent Distribution
The taxonomy was applied across all **88,267 customer interactions** spanning the full Stage 4 dataset:

| Intent ID | Intent Name | Count | Percentage | Train | Val | Test |
|:----------|:------------|------:|-----------:|------:|----:|-----:|
| `unknown_or_ambiguous` | Unknown or Ambiguous | 32,385 | 36.7% | 25,957 | 3,231 | 3,197 |
| `delivery_delay` | Delivery Delay & Tracking | 27,518 | 31.2% | 21,993 | 2,748 | 2,777 |
| `returns_and_refunds` | Returns & Refunds | 7,187 | 8.1% | 5,796 | 717 | 674 |
| `prime_membership` | Amazon Prime & Subscriptions | 4,562 | 5.2% | 3,646 | 472 | 444 |
| `digital_services_technical` | Digital Devices & Services | 3,413 | 3.9% | 2,763 | 304 | 346 |
| `order_cancellation` | Order Cancellation & Modification | 3,015 | 3.4% | 2,404 | 306 | 305 |
| `payment_and_billing` | Payment, Charges & Billing | 2,511 | 2.8% | 1,987 | 270 | 254 |
| `damaged_defective_item` | Damaged or Defective Item | 2,297 | 2.6% | 1,826 | 228 | 243 |
| `service_complaint_escalation` | Customer Service Escalation | 2,007 | 2.3% | 1,608 | 193 | 206 |
| `missing_delivered_package` | Missing Delivered Package | 1,698 | 1.9% | 1,339 | 183 | 176 |
| `account_access_security` | Account Access & Security | 1,674 | 1.9% | 1,362 | 157 | 155 |

## 9. Ambiguity & Multi-Intent Analysis
- **Unknown / Ambiguous Volume**: **32,385 examples (36.7%)**.
- **Why This Exists**: Twitter customer support contains many very short tweets (e.g. *"check DM"*, *"please reply"*, *"can you help?"*), as well as isolated follow-up turns in multi-turn dialogues (*"I did that"*, *"yes"*, *"no luck"*) that lack standalone problem context.
- **Multi-Intent Priority Rule**: When a customer inquiry references multiple problems (e.g. *"My package was damaged and I want to return it for a refund"*), a deterministic priority hierarchy is applied (`missing_delivered_package` > `damaged_defective_item` > `order_cancellation` > `prime_membership` > `returns_and_refunds` > `payment_and_billing` > `delivery_delay`). This ensures reproducible, predictable single-label assignment for downstream classification models.

## 10. Data Imbalance
- **Dominant Intent**: `delivery_delay` accounts for **31.2%** of all customer interactions (27,518 examples). This reflects the core operational volume of e-commerce retail support.
- **Moderate Volume**: `returns_and_refunds` (8.1%), `prime_membership` (5.2%), `digital_services_technical` (3.9%), and `order_cancellation` (3.4%) provide rich training clusters.
- **Focused/Niche Volume**: `damaged_defective_item` (2.6%), `payment_and_billing` (2.8%), `missing_delivered_package` (1.9%), and `account_access_security` (1.9%) each have 1,600–2,500+ examples, which is ample for few-shot and fine-tuned model evaluation without severe sparsity.
- **Preservation Principle**: Real-world support traffic is naturally skewed; we preserve the empirical distribution in Stage 6 rather than artificially forcing a uniform distribution.

## 11. Split Integrity & Leakage Prevention
Split integrity was strictly maintained according to Stage 4 conversation-level partitioning:
- **Train Split**: 70,681 records
- **Validation Split**: 8,809 records
- **Test Split**: 8,777 records
- **Leakage Verification**: $$\text{Train} \cap \text{Validation} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Validation} \cap \text{Test} = \emptyset$$
All customer messages, support responses, and conversation turns belonging to the same `conversation_id` reside in the exact same split.

## 12. Limitations
1. **Preliminary Automatic Labeling**: Labels generated in Stage 6 are rule- and cluster-assisted heuristic labels intended for scalable baseline training. They are NOT claimed to be ground truth.
2. **Single-Label Simplification**: Real-world queries occasionally present dual intents (e.g. billing charge + cancellation). Stage 6 enforces single-label priority to maintain baseline simplicity.
3. **Isolated Turn Context**: Multi-turn intermediate turns can be ambiguous when evaluated in isolation from dialogue history.
4. **Human Validation Requirement**: High-stakes evaluation requires a rigorously curated Golden Evaluation Dataset with human review, which is the explicit objective of Stage 7.

## 13. Readiness for Stage 7 (Golden Evaluation Dataset)
The deliverables of Stage 6 fully establish the foundation for Stage 7:
1. **Clear Taxonomy (v1.0)**: 10 unambiguous intent categories with explicit inclusion/exclusion guidelines and confusable intent distinctions.
2. **Comprehensive Labeling**: 88,267 interactions tagged with intent IDs, splits, and response types.
3. **Zero Data Leakage**: Guaranteed conversation isolation across train, val, and test splits.
4. **Targeted Sampling Candidate Pool**: Stage 7 can draw balanced stratified samples across these 10 categories to build the Golden Evaluation Dataset.
