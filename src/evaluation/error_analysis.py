"""
error_analysis.py
-----------------
STAGE 10: Granular Error Analysis Engine

Diagnoses failures on the Golden Evaluation Set and categorizes them into:
  - intent_confusion: Single-intent classification error
  - multi_intent_failure: Predicted secondary intent instead of primary (or missed secondary issue)
  - insufficient_context: Ambiguous or minimal inquiry lacking necessary operational details
  - policy_violation: Response contained unverified claim or hallucinated date/action
  - response_irrelevance: Response failed to adhere to gold guidance requirements
  - unnecessary_escalation: Escalated standard inquiry unnecessarily
  - missed_escalation: Failed to escalate angry customer or severe complaint
  - other: Unclassified edge-case failure

Outputs:
  - JSONL failure audit logs
  - Aggregate failure category distributions
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.datasets import GoldenEvaluationRecord


class ErrorAnalyzer:
    """Performs deep failure categorization on benchmark evaluation runs."""

    def diagnose_record(
        self,
        gold: GoldenEvaluationRecord,
        pred_intent: str,
        confidence: float,
        response_text: str,
        policy_result: Dict[str, Any],
        escalation_result: Dict[str, Any],
        guidance_adherence: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Diagnoses a single record. Returns failure audit dict if flawed, else None.
        """
        is_intent_correct = gold.primary_intent == pred_intent
        is_policy_clean = policy_result.get("passed", True)
        is_guidance_ok = guidance_adherence >= 0.5
        is_escalation_ok = True

        should_escalate = (
            gold.primary_intent == "service_complaint_escalation"
            or "urgent" in gold.difficulty
        )
        did_escalate = escalation_result.get("required", False)

        if should_escalate and not did_escalate:
            is_escalation_ok = False
        elif not should_escalate and did_escalate and gold.difficulty == "easy":
            is_escalation_ok = False

        # If everything succeeded, no error
        if is_intent_correct and is_policy_clean and is_guidance_ok and is_escalation_ok:
            return None

        # Determine failure category
        failure_category = "other"

        if not is_policy_clean:
            failure_category = "policy_violation"
        elif not is_intent_correct:
            if gold.secondary_intent and pred_intent == gold.secondary_intent:
                failure_category = "multi_intent_failure"
            elif len(gold.customer_message.strip()) < 30 or gold.primary_intent == "unknown_or_ambiguous":
                failure_category = "insufficient_context"
            else:
                failure_category = "intent_confusion"
        elif not is_guidance_ok:
            failure_category = "response_irrelevance"
        elif not is_escalation_ok:
            if should_escalate and not did_escalate:
                failure_category = "missed_escalation"
            else:
                failure_category = "unnecessary_escalation"

        return {
            "example_id": gold.example_id,
            "conversation_id": gold.conversation_id,
            "difficulty": gold.difficulty,
            "gold_intent": gold.primary_intent,
            "gold_intent_name": gold.primary_intent_name,
            "predicted_intent": pred_intent,
            "confidence": round(confidence, 4),
            "secondary_intent": gold.secondary_intent,
            "customer_message": gold.customer_message,
            "generated_response": response_text,
            "policy_result": policy_result,
            "escalation_decision": escalation_result,
            "guidance_adherence": guidance_adherence,
            "failure_category": failure_category,
        }

    def analyze_dataset(
        self,
        golden_records: List[GoldenEvaluationRecord],
        predictions: List[Dict[str, Any]],
        guidance_scores: List[float],
        output_jsonl_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Runs error diagnosis across all golden records and exports error_analysis.jsonl.
        """
        assert len(golden_records) == len(predictions) == len(guidance_scores)

        failures: List[Dict[str, Any]] = []

        for gold, pred, adh in zip(golden_records, predictions, guidance_scores):
            diagnosis = self.diagnose_record(
                gold=gold,
                pred_intent=pred["primary_intent"],
                confidence=pred.get("intent_confidence", 0.0),
                response_text=pred.get("response", ""),
                policy_result=pred.get("policy", {}),
                escalation_result=pred.get("escalation", {}),
                guidance_adherence=adh,
            )
            if diagnosis is not None:
                failures.append(diagnosis)

        # Export to JSONL
        if output_jsonl_path:
            output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_jsonl_path, "w", encoding="utf-8") as fh:
                for fail in failures:
                    fh.write(json.dumps(fail, ensure_ascii=False) + "\n")

        categories = Counter(f["failure_category"] for f in failures)
        diff_breakdown = Counter(f["difficulty"] for f in failures)

        return {
            "total_failures": len(failures),
            "total_evaluated": len(golden_records),
            "failure_rate": round(len(failures) / len(golden_records), 4) if golden_records else 0.0,
            "failure_categories": dict(categories),
            "failures_by_difficulty": dict(diff_breakdown),
            "sample_failures": failures[:5],
        }
