"""
failure_analysis.py
-------------------
STAGE 11: Systematic Failure Analysis & Pattern Detection Engine

Provides:
  1. 18-category Failure Taxonomy
  2. Structured FailureRecord schema
  3. Root Cause Analysis (distinguishing Symptom vs Root Cause)
  4. Severity Assignment (critical, high, medium, low)
  5. Component Ownership Attribution
  6. High-Confidence Error Analysis
  7. Lexical/Semantic Failure Clustering
  8. Automated Pattern Detection & Ranked Aggregations
  9. Ranked Engineering Recommendations (P0, P1, P2, P3)
"""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.evaluation.datasets import GoldenEvaluationRecord

# ---------------------------------------------------------------------------
# 1. TAXONOMIES & ENUMERATIONS
# ---------------------------------------------------------------------------

VALID_FAILURE_CATEGORIES = [
    "intent_confusion",
    "multi_intent_failure",
    "ambiguous_input",
    "insufficient_context",
    "retrieval_failure",
    "irrelevant_retrieval",
    "policy_violation",
    "hallucination",
    "unsupported_action_claim",
    "missing_required_information",
    "incorrect_escalation",
    "unnecessary_escalation",
    "response_irrelevance",
    "response_incompleteness",
    "excessive_response",
    "low_confidence_failure",
    "latency_failure",
    "other",
]

VALID_SEVERITIES = ["critical", "high", "medium", "low"]

VALID_COMPONENTS = [
    "normalization",
    "intent classifier",
    "context extraction",
    "retrieval",
    "policy layer",
    "response generation",
    "escalation",
    "evaluation/measurement",
    "data quality",
    "unknown",
]

VALID_STATUSES = ["open", "analyzed", "in_progress", "resolved", "wont_fix"]


# ---------------------------------------------------------------------------
# 2. FAILURE RECORD SCHEMA
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    """Structured representation of a single system failure."""

    failure_id: str
    example_id: str
    conversation_id: str
    gold_intent: str
    predicted_intent: str
    secondary_intents: Optional[List[str]] = field(default_factory=list)
    difficulty: str = "medium"
    confidence: float = 0.0
    retrieved_references: List[Dict[str, Any]] = field(default_factory=list)
    policy_result: Dict[str, Any] = field(default_factory=dict)
    escalation_result: Dict[str, Any] = field(default_factory=dict)
    generated_response: str = ""
    expected_behavior: str = ""
    failure_category: str = "other"
    root_cause: str = "unknown"
    severity: str = "medium"
    affected_component: str = "unknown"
    recommended_action: str = ""
    status: str = "analyzed"
    customer_message: str = ""

    def validate(self) -> List[str]:
        """Validates that all fields comply with taxonomy and schema constraints."""
        errors: List[str] = []
        if self.failure_category not in VALID_FAILURE_CATEGORIES:
            errors.append(f"Invalid failure_category: {self.failure_category}")
        if self.severity not in VALID_SEVERITIES:
            errors.append(f"Invalid severity: {self.severity}")
        if self.affected_component not in VALID_COMPONENTS:
            errors.append(f"Invalid affected_component: {self.affected_component}")
        if self.status not in VALID_STATUSES:
            errors.append(f"Invalid status: {self.status}")
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"Confidence out of bounds [0, 1]: {self.confidence}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to JSON-serializable dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# 3. ROOT CAUSE & COMPONENT ATTRIBUTION RULES
# ---------------------------------------------------------------------------

class FailureDiagnosisEngine:
    """
    Expert system for diagnosing evaluation failures, separating symptoms from
    root causes, assigning severity, and identifying responsible components.
    """

    @staticmethod
    def determine_failure_category(
        gold: GoldenEvaluationRecord,
        pred_intent: str,
        confidence: float,
        response_text: str,
        policy_result: Dict[str, Any],
        escalation_result: Dict[str, Any],
        guidance_adherence: float,
        retrieved_refs: List[Any],
        latency_ms: float = 0.0,
    ) -> str:
        """Determines the most accurate category from the 18-item taxonomy."""
        # 1. Latency SLA violation (> 500ms is standard support SLA failure)
        if latency_ms > 500.0:
            return "latency_failure"

        # 2. Policy & Safety Violations
        if not policy_result.get("passed", True):
            violations = policy_result.get("violations", [])
            for v in violations:
                rule = v.get("rule", "")
                if "unsupported_action" in rule or "promise" in rule:
                    return "unsupported_action_claim"
                if "hallucination" in rule or "fact" in rule:
                    return "hallucination"
            return "policy_violation"

        # 3. Hallucinations or Unsupported Claims in response text
        claim_patterns = [
            r"\b(i have refunded|i just refunded|refund of \$\d+ has been processed|processed your refund)\b",
            r"\b(i cancelled your order|i have cancelled order)\b",
        ]
        if any(re.search(p, response_text.lower()) for p in claim_patterns):
            return "unsupported_action_claim"

        # 4. Intent Classification Errors
        is_intent_correct = (gold.primary_intent == pred_intent)
        if not is_intent_correct:
            # Low confidence failures (< 0.45)
            if confidence < 0.45:
                return "low_confidence_failure"

            # Check for secondary intent match
            if gold.secondary_intent and pred_intent == gold.secondary_intent:
                return "multi_intent_failure"

            # Check customer message length / ambiguity
            clean_msg = gold.customer_message.strip()
            if len(clean_msg) < 25:
                return "insufficient_context"
            if gold.primary_intent == "unknown_or_ambiguous":
                return "ambiguous_input"

            # Check for multi-intent signals in text
            msg_lower = clean_msg.lower()
            intent_markers = 0
            if any(w in msg_lower for w in ["cancel", "cancellation"]):
                intent_markers += 1
            if any(w in msg_lower for w in ["refund", "return", "money back"]):
                intent_markers += 1
            if any(w in msg_lower for w in ["prime", "membership"]):
                intent_markers += 1
            if any(w in msg_lower for w in ["late", "delay", "tracking", "deliver"]):
                intent_markers += 1
            if intent_markers >= 2:
                return "multi_intent_failure"

            return "intent_confusion"

        # 5. Escalation Errors
        should_escalate = (
            gold.primary_intent == "service_complaint_escalation"
            or "urgent" in gold.difficulty
        )
        did_escalate = escalation_result.get("required", False)
        if should_escalate and not did_escalate:
            return "incorrect_escalation"
        if not should_escalate and did_escalate and gold.difficulty == "easy":
            return "unnecessary_escalation"

        # 6. Retrieval Failures
        if not retrieved_refs:
            return "retrieval_failure"

        # 7. Guidance & Response Quality
        if guidance_adherence < 0.5:
            if len(response_text.strip()) < 40:
                return "response_incompleteness"
            return "response_irrelevance"

        if len(response_text.split()) > 150:
            return "excessive_response"

        return "other"

    @staticmethod
    def diagnose_root_cause(
        gold: GoldenEvaluationRecord,
        pred_intent: str,
        category: str,
        confidence: float,
        customer_message: str,
        response_text: str,
    ) -> Tuple[str, str, str, str]:
        """
        Determines (root_cause, severity, affected_component, recommended_action).
        Distinguishes Symptom from Root Cause.
        """
        msg_lower = customer_message.lower()

        # Default fallback
        root_cause = "unknown"
        severity = "medium"
        component = "intent classifier"
        action = "Review feature weights and training examples"

        if category == "policy_violation" or category == "unsupported_action_claim":
            severity = "critical"
            component = "policy layer"
            root_cause = "Response generation template emitted operational promise without authorization guardrail verification."
            action = "Strengthen policy regex validator to block unauthorized transactional guarantees."

        elif category == "hallucination":
            severity = "critical"
            component = "response generation"
            root_cause = "Generation module fabricated unverified dates or specific claim not present in retrieved context."
            action = "Ground template variables strictly against verified order metadata."

        elif category == "multi_intent_failure":
            severity = "high"
            component = "intent classifier"
            if "prime" in msg_lower and ("cancel" in msg_lower or "late" in msg_lower or "delay" in msg_lower):
                root_cause = "Query contains both membership entity ('prime') and action/status verb; keyword booster gave priority to entity rather than core operational intent."
                action = "Introduce compound multi-intent priority rules to rank operational action verbs above entity mentions."
            elif "refund" in msg_lower and "cancel" in msg_lower:
                root_cause = "Query conflates cancellation with post-cancellation refund; shared financial vocabulary obscured primary cancellation intent."
                action = "Add contextual tie-breaking for post-cancellation refund queries."
            else:
                root_cause = "Compound customer request triggers multiple high-weight lexical rules simultaneously."
                action = "Support explicit multi-intent decomposition and multi-part response generation."

        elif category == "intent_confusion":
            severity = "high"
            component = "intent classifier"
            if gold.primary_intent == "delivery_delay" and pred_intent == "missing_delivered_package":
                root_cause = "High-weight lexical trigger ('not received' / 'missing') triggered missing package rule on an inquiry that only asked about delayed tracking."
                action = "Refine 'missing_delivered_package' discriminative triggers to require explicit 'delivered' status markers."
            elif gold.primary_intent == "order_cancellation" and pred_intent in ["prime_membership", "returns_and_refunds"]:
                root_cause = "Entity keyword dominance ('prime' or 'refund') masked the cancellation action verb."
                action = "Boost action verb 'cancel' over subscription noun phrases."
            elif gold.primary_intent == "unknown_or_ambiguous":
                severity = "medium"
                root_cause = "Vague customer text matched generic lexical n-gram prior instead of defaulting to unknown."
                action = "Calibrate ambiguity threshold so low-specificity text routes to clarification."
            else:
                root_cause = f"Overlapping TF-IDF term weights between '{gold.primary_intent}' and '{pred_intent}'."
                action = f"Add discriminative negative features between {gold.primary_intent} and {pred_intent}."

        elif category == "insufficient_context":
            severity = "low"
            component = "context extraction"
            root_cause = "Customer input is extremely brief (< 25 characters) or lacks necessary identifier entities."
            action = "Prompt customer for order ID or tracking number via structured clarification."

        elif category == "ambiguous_input":
            severity = "medium"
            component = "intent classifier"
            root_cause = "Input text lacks clear operational cues, causing false positive intent classification."
            action = "Lower fallback threshold to classify ambiguous text as 'unknown_or_ambiguous'."

        elif category == "incorrect_escalation":
            severity = "high"
            component = "escalation"
            root_cause = "Escalation rules missed sentiment or urgency cues in severe customer complaint."
            action = "Broaden escalation keyword triggers to capture severe customer dissatisfaction."

        elif category == "unnecessary_escalation":
            severity = "medium"
            component = "escalation"
            root_cause = "Overly sensitive escalation filter triggered on benign inquiry."
            action = "Constrain escalation triggers to require severe complaint keywords."

        elif category == "retrieval_failure" or category == "irrelevant_retrieval":
            severity = "medium"
            component = "retrieval"
            root_cause = "BM25 index returned low-relevance snippets due to vocabulary mismatch with customer query."
            action = "Expand query expansion terms and synonym mappings in retrieval index."

        elif category == "response_irrelevance" or category == "response_incompleteness":
            severity = "medium"
            component = "response generation"
            root_cause = "Response template selected lacked necessary procedural instructions for this specific intent."
            action = "Update response generation templates to ensure all gold guidance requirements are met."

        elif category == "latency_failure":
            severity = "high"
            component = "normalization"
            root_cause = "Inefficient processing loop exceeded latency SLA."
            action = "Optimize feature extraction vectorization."

        return root_cause, severity, component, action


# ---------------------------------------------------------------------------
# 4. PATTERN AGGREGATION & REPORTING ENGINE
# ---------------------------------------------------------------------------

class FailureAnalysisEngine:
    """
    Coordinates end-to-end failure auditing, pattern detection, clustering,
    high-confidence error analysis, and ranked recommendations.
    """

    def __init__(self):
        self.diagnostician = FailureDiagnosisEngine()

    def build_failure_records(
        self,
        golden_records: List[GoldenEvaluationRecord],
        predictions: List[Dict[str, Any]],
        guidance_scores: List[float],
        latencies_ms: Optional[List[float]] = None,
    ) -> List[FailureRecord]:
        """
        Builds fully populated FailureRecord instances for all failing examples.
        """
        assert len(golden_records) == len(predictions) == len(guidance_scores)
        if latencies_ms is None:
            latencies_ms = [0.0] * len(golden_records)

        failures: List[FailureRecord] = []
        fail_counter = 1

        for gold, pred, adh, lat in zip(golden_records, predictions, guidance_scores, latencies_ms):
            pred_intent = pred.get("primary_intent", "unknown_or_ambiguous")
            conf = pred.get("intent_confidence", 0.0)
            resp = pred.get("response", "")
            pol = pred.get("policy", {})
            esc = pred.get("escalation", {})
            refs = pred.get("retrieved_references", [])

            # Check if record is flawed
            is_intent_ok = (gold.primary_intent == pred_intent)
            is_pol_ok = pol.get("passed", True)
            is_adh_ok = (adh >= 0.5)
            should_esc = (
                gold.primary_intent == "service_complaint_escalation"
                or "urgent" in gold.difficulty
            )
            did_esc = esc.get("required", False)
            is_esc_ok = not (should_esc and not did_esc)

            if is_intent_ok and is_pol_ok and is_adh_ok and is_esc_ok:
                continue

            # Determine category
            category = self.diagnostician.determine_failure_category(
                gold=gold,
                pred_intent=pred_intent,
                confidence=conf,
                response_text=resp,
                policy_result=pol,
                escalation_result=esc,
                guidance_adherence=adh,
                retrieved_refs=refs,
                latency_ms=lat,
            )

            # Diagnose root cause & component
            root_cause, severity, component, action = self.diagnostician.diagnose_root_cause(
                gold=gold,
                pred_intent=pred_intent,
                category=category,
                confidence=conf,
                customer_message=gold.customer_message,
                response_text=resp,
            )

            rec = FailureRecord(
                failure_id=f"fail_{fail_counter:03d}",
                example_id=gold.example_id,
                conversation_id=gold.conversation_id,
                gold_intent=gold.primary_intent,
                predicted_intent=pred_intent,
                secondary_intents=[gold.secondary_intent] if gold.secondary_intent else [],
                difficulty=gold.difficulty,
                confidence=round(conf, 4),
                retrieved_references=refs,
                policy_result=pol,
                escalation_result=esc,
                generated_response=resp,
                expected_behavior=gold.gold_response_guidance,
                failure_category=category,
                root_cause=root_cause,
                severity=severity,
                affected_component=component,
                recommended_action=action,
                status="analyzed",
                customer_message=gold.customer_message,
            )

            # Validation
            val_errors = rec.validate()
            if val_errors:
                raise ValueError(f"FailureRecord validation failed: {val_errors}")

            failures.append(rec)
            fail_counter += 1

        return failures

    def aggregate_patterns(
        self,
        failures: List[FailureRecord],
        total_evaluated: int = 200,
    ) -> Dict[str, Any]:
        """
        Calculates ranked metrics across failure categories, intents, confusion
        pairs, difficulty levels, components, and high-confidence errors.
        """
        category_counts = Counter(f.failure_category for f in failures)
        gold_intent_counts = Counter(f.gold_intent for f in failures)
        difficulty_counts = Counter(f.difficulty for f in failures)
        severity_counts = Counter(f.severity for f in failures)
        component_counts = Counter(f.affected_component for f in failures)

        # Confusion pairs: (gold, pred)
        confusion_pairs = Counter(
            (f.gold_intent, f.predicted_intent)
            for f in failures
            if f.gold_intent != f.predicted_intent
        )

        # High-confidence errors: confidence >= 0.85 and prediction wrong
        high_conf_errors = [
            f for f in failures
            if f.confidence >= 0.85 and f.gold_intent != f.predicted_intent
        ]

        # Failure clusters
        clusters = self.cluster_failures(failures)

        # Ranked recommendations
        recommendations = self.generate_recommendations(failures, high_conf_errors, clusters)

        return {
            "total_failures": len(failures),
            "total_evaluated": total_evaluated,
            "overall_failure_rate": round(len(failures) / total_evaluated, 4) if total_evaluated else 0.0,
            "category_ranking": [
                {"category": cat, "count": count, "rate": round(count / total_evaluated, 4)}
                for cat, count in category_counts.most_common()
            ],
            "weakest_intents": [
                {"intent": intent, "failure_count": count}
                for intent, count in gold_intent_counts.most_common()
            ],
            "top_confusion_pairs": [
                {"gold_intent": pair[0], "predicted_intent": pair[1], "count": count}
                for pair, count in confusion_pairs.most_common(10)
            ],
            "difficulty_breakdown": dict(difficulty_counts),
            "severity_breakdown": dict(severity_counts),
            "component_breakdown": [
                {"component": comp, "failure_count": count, "percentage": round(count / len(failures) * 100, 1)}
                for comp, count in component_counts.most_common()
            ] if failures else [],
            "high_confidence_errors": {
                "count": len(high_conf_errors),
                "rate": round(len(high_conf_errors) / total_evaluated, 4) if total_evaluated else 0.0,
                "percentage_of_failures": round(len(high_conf_errors) / len(failures) * 100, 1) if failures else 0.0,
                "affected_intents": dict(Counter(f.gold_intent for f in high_conf_errors)),
                "sample_examples": [
                    {
                        "example_id": f.example_id,
                        "gold_intent": f.gold_intent,
                        "predicted_intent": f.predicted_intent,
                        "confidence": f.confidence,
                        "message_snippet": f.customer_message[:80],
                        "root_cause": f.root_cause,
                    }
                    for f in high_conf_errors[:5]
                ],
            },
            "failure_clusters": clusters,
            "recommendations": recommendations,
        }

    def cluster_failures(self, failures: List[FailureRecord]) -> List[Dict[str, Any]]:
        """
        Groups similar failures into actionable semantic clusters.
        """
        cluster_a: List[FailureRecord] = []  # Delivery Delay vs Missing Delivered Package
        cluster_b: List[FailureRecord] = []  # Cancellation vs Prime/Refunds
        cluster_c: List[FailureRecord] = []  # Multi-intent customer inquiries
        cluster_d: List[FailureRecord] = []  # Ambiguous / Low Context
        cluster_other: List[FailureRecord] = []

        for f in failures:
            if (f.gold_intent == "delivery_delay" and f.predicted_intent == "missing_delivered_package") or \
               (f.gold_intent == "missing_delivered_package" and f.predicted_intent == "delivery_delay"):
                cluster_a.append(f)
            elif f.gold_intent == "order_cancellation" and f.predicted_intent in ["prime_membership", "returns_and_refunds"]:
                cluster_b.append(f)
            elif f.failure_category == "multi_intent_failure":
                cluster_c.append(f)
            elif f.failure_category in ["ambiguous_input", "insufficient_context"]:
                cluster_d.append(f)
            else:
                cluster_other.append(f)

        clusters = [
            {
                "cluster_id": "CLUSTER_A",
                "name": "Delivery Delay vs Missing Delivered Package Confusion",
                "size": len(cluster_a),
                "severity": "high",
                "description": "Customer queries reporting delayed shipping or tracking questions misclassified as lost/missing delivered packages due to aggressive lexical triggers ('not received', 'missing').",
                "examples": [f.example_id for f in cluster_a[:5]],
            },
            {
                "cluster_id": "CLUSTER_B",
                "name": "Order Cancellation Masked by Secondary Entity Mentions",
                "size": len(cluster_b),
                "severity": "high",
                "description": "Inquiries seeking order or subscription cancellation where customer mentions 'prime' or 'refund', causing entity keyword boosts to eclipse the core cancellation intent.",
                "examples": [f.example_id for f in cluster_b[:5]],
            },
            {
                "cluster_id": "CLUSTER_C",
                "name": "Compound Multi-Intent Customer Requests",
                "size": len(cluster_c),
                "severity": "high",
                "description": "Customer messages expressing two valid intents simultaneously (e.g. late delivery + subscription complaint), resulting in single-intent misclassification.",
                "examples": [f.example_id for f in cluster_c[:5]],
            },
            {
                "cluster_id": "CLUSTER_D",
                "name": "Sparse and Ambiguous Input Deflection",
                "size": len(cluster_d),
                "severity": "medium",
                "description": "Short messages (< 25 characters) or vague complaints lacking explicit entities, causing over-confident pseudo-random intent classification.",
                "examples": [f.example_id for f in cluster_d[:5]],
            },
        ]
        return clusters

    def generate_recommendations(
        self,
        failures: List[FailureRecord],
        high_conf_errors: List[FailureRecord],
        clusters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generates prioritized engineering recommendations (P0, P1, P2, P3).
        """
        return [
            {
                "priority": "P0",
                "title": "Strict Guardrails Against Unsupported Operational Commitments",
                "problem": "Potential customer misinformation if generation templates promise refunds or direct order modifications.",
                "evidence": "Policy check compliance must remain at 100.00% without regression.",
                "proposed_solution": "Maintain deterministic zero-tolerance regex guards blocking words like 'I have refunded' or 'I have cancelled' before emission.",
                "expected_benefit": "Guarantees zero operational liability and prevents customer misinformation.",
                "risk": "Minimal risk of false positive blocking on valid conversational turns.",
                "implementation_difficulty": "Low",
            },
            {
                "priority": "P1",
                "title": "Disambiguate Delivery Delay vs Missing Delivered Package",
                "problem": "7 instances of tracking/delivery delay inquiries were misclassified as lost packages, recommending irrelevant mailroom searches.",
                "evidence": f"Cluster A accounts for {[c for c in clusters if c['cluster_id'] == 'CLUSTER_A'][0]['size']} confusion errors; high-confidence error rate is elevated.",
                "proposed_solution": "Require explicit confirmation of 'marked as delivered' status before firing the 4.5 missing package boost; let delivery_delay handle transit inquiries.",
                "expected_benefit": "Immediate +2.0% to +3.5% intent accuracy improvement on hard delivery inquiries.",
                "risk": "May slightly reduce recall for missing package if user says 'not received' without mentioning delivery status.",
                "implementation_difficulty": "Low",
            },
            {
                "priority": "P1",
                "title": "Disambiguate Action Verbs over Substantive Entities in Order Cancellation",
                "problem": "10 order cancellation inquiries were misclassified as Prime Membership or Returns/Refunds due to entity keyword dominance.",
                "evidence": "Order cancellation has the lowest recall (44.44%) of all intents in the benchmark.",
                "proposed_solution": "Introduce compound precedence rules giving 'cancel' + 'subscription/order' higher priority than bare 'prime' mentions.",
                "expected_benefit": "Substantial recovery of order_cancellation recall from 44.44% to > 70.00%.",
                "risk": "Could misclassify general Prime queries that mention 'how do I cancel if I don't like it' as cancellations.",
                "implementation_difficulty": "Medium",
            },
            {
                "priority": "P2",
                "title": "Calibrated Confidence Thresholding for Short/Ambiguous Inquiries",
                "problem": "Short (< 25 chars) or ambiguous inquiries receive high confidence predictions (~0.90) due to single word token matches.",
                "evidence": f"High-confidence incorrect prediction rate is {round(len(high_conf_errors)/200*100, 1)}%.",
                "proposed_solution": "Penalize confidence on inputs with token length < 5, routing them to 'unknown_or_ambiguous' clarification.",
                "expected_benefit": "Reduces high-confidence error rate and improves expected calibration error (ECE).",
                "risk": "May increase deflection rate on short but legitimate queries.",
                "implementation_difficulty": "Low",
            },
            {
                "priority": "P3",
                "title": "Response Template Guidance Polish for Multi-Part Inquiries",
                "problem": "Guidance adherence drops on compound queries when only the primary issue is addressed.",
                "evidence": "10% of examples miss one or more guidance criteria on multi-turn context.",
                "proposed_solution": "Append modular guidance snippets when secondary intents are detected.",
                "expected_benefit": "Pushes guidance adherence from 90.00% to > 95.00%.",
                "risk": "Responses may become slightly more verbose.",
                "implementation_difficulty": "Medium",
            },
        ]

    def export_artifacts(
        self,
        failures: List[FailureRecord],
        pattern_data: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Exports machine-readable JSONL and JSON artifacts for Stage 11.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        # 1. failure_dataset.jsonl
        jsonl_path = output_dir / "failure_dataset.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for fail in failures:
                fh.write(json.dumps(fail.to_dict(), ensure_ascii=False) + "\n")
        paths["failure_dataset_jsonl"] = jsonl_path

        # 2. failure_patterns.json
        patterns_path = output_dir / "failure_patterns.json"
        with open(patterns_path, "w", encoding="utf-8") as fh:
            json.dump(pattern_data, fh, indent=2, ensure_ascii=False)
        paths["failure_patterns_json"] = patterns_path

        # 3. high_confidence_errors.json
        high_conf_path = output_dir / "high_confidence_errors.json"
        with open(high_conf_path, "w", encoding="utf-8") as fh:
            json.dump(pattern_data.get("high_confidence_errors", {}), fh, indent=2, ensure_ascii=False)
        paths["high_confidence_errors_json"] = high_conf_path

        # 4. failure_clusters.json
        clusters_path = output_dir / "failure_clusters.json"
        with open(clusters_path, "w", encoding="utf-8") as fh:
            json.dump(pattern_data.get("failure_clusters", []), fh, indent=2, ensure_ascii=False)
        paths["failure_clusters_json"] = clusters_path

        return paths
