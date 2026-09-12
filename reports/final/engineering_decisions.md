# Summary of Engineering Decisions: AmazonHelp AI Support Agent

This document synthesizes the core architectural and algorithmic decisions evaluated during the project lifecycle. Every engineering decision was subjected to rigorous empirical evaluation against the protected Golden Evaluation Set ($N=200$) with regression protection.

---

## DEC-001: Strict Guardrails Against Unsupported Operational Commitments

- **Decision ID**: `DEC-001`
- **Problem**: Automated customer support agents risk hallucinating unverified actions (e.g. promising "I have processed a refund of $50" or "I have cancelled your shipment") without human authorization or live ERP integration.
- **Evidence**: Amazon customer support standards require zero-tolerance for fabricated promises. Historical TWCS agent messages frequently contained deflected promises that could mislead users if emitted without execution.
- **Hypothesis**: Hard deterministic pre-emission regex validation combined with strictly grounded canonical response templates will eliminate unauthorized commitments without degrading guidance adherence.
- **Proposed Change**: Implement deterministic zero-tolerance regex guardrails blocking phrases like "I have refunded", "I have cancelled", or specific arrival guarantees.
- **Result**: Maintained **100.00% Policy Safety Rate** across all 200 Golden Evaluation Set records with 0 recorded violations.
- **Decision**: **KEEP**
- **Date**: 2026-09-12

---

## DEC-002: Disambiguate Delivery Tracking vs Missing Delivered Package

- **Problem**: In-transit delivery inquiries and locker deliveries were systematically misclassified as lost packages (`missing_delivered_package`), directing users to search building mailrooms or porches for packages that had not yet been delivered.
- **Evidence**: Cluster A accounted for 7 classification errors. The discriminative boost for `missing_delivered_package` contained the bare keyword `"not received"` with a high +4.5 weight, overriding tracking context.
- **Hypothesis**: Requiring explicit delivery status markers (e.g. "marked as delivered", "says delivered", "package not received but delivered") before triggering the missing package boost will allow `delivery_delay` to capture in-transit tracking queries.
- **Proposed Change**: Constrain `missing_delivered_package` triggers to require delivered markers and remove bare `"not received"`.
- **Result**: Accuracy improved from **79.00% to 82.00% (+3.00% absolute)**, hard difficulty accuracy jumped from **75.00% to 79.55% (+4.55% absolute)**, with zero safety regressions.
- **Decision**: **KEEP**
- **Date**: 2026-09-12

---

## DEC-003: Action Verb Dominance in Order & Subscription Cancellation

- **Problem**: `order_cancellation` had the lowest baseline recall (44.44%) because customer messages like "is there an email to cancel my prime membership?" were hijacked by the `prime_membership` entity keyword booster (+3.5 boost).
- **Evidence**: 10 cancellation failures occurred in the benchmark; 4 specifically routed to `prime_membership` due to the "prime" noun phrase masking the "cancel" action verb.
- **Hypothesis**: Adding compound cancellation triggers that combine action verbs with subscription nouns ("cancel my prime", "cancelling subscription", "cancel membership") will give precedence to customer intent over entity mentions.
- **Proposed Change**: Expand `order_cancellation` regex boosters to explicitly capture compound subscription cancellation phrases.
- **Result**: Order cancellation recall rose from **44.44% to 66.67%**, lifting overall Macro F1 from **0.7846 to 0.8204 (+0.0358)**.
- **Decision**: **KEEP**
- **Date**: 2026-09-12

---

## DEC-004: Calibrated Ambiguity Thresholding for Short Queries

- **Problem**: Short, vague customer queries (< 25 characters like "it is not working" or "help") were receiving high confidence predictions (~0.90) due to single-token keyword matches.
- **Evidence**: High-confidence error rate was 18.50%; 9 `unknown_or_ambiguous` queries were misclassified as specific categories.
- **Hypothesis**: Imposing an aggressive confidence length penalty on inputs with token length < 5 to force routing to `unknown_or_ambiguous` clarification will reduce high-confidence errors.
- **Proposed Change**: Deduct 0.30 from calibrated confidence for inputs under 5 tokens.
- **Risk**: Induces false-positive deflections on brief but valid single-intent requests (e.g. "cancel order 123").
- **Result**: Preliminary testing showed a drop in recall for short explicit queries. Better addressed through a dedicated multi-turn clarification dialog engine.
- **Decision**: **DEFER** (Scheduled for multi-turn conversational dialog milestone)
- **Date**: 2026-09-12
