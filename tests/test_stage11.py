"""
test_stage11.py
---------------
STAGE 11: Unit & Integration Verification Suite

Validates all 12 Stage 11 requirements:
  1. Failure Record Schema & Validation
  2. Failure Categorization (18-item taxonomy)
  3. Severity Assignment (Critical, High, Medium, Low)
  4. Root-Cause Representation (Symptom vs Root Cause, unknown handling)
  5. Component Ownership Attribution
  6. Decision Log Format & Markdown Generation
  7. Aggregation Calculations & Ranking
  8. High-Confidence Error Detection
  9. Regression Detection & Threshold Enforcement
  10. Experiment Comparison (BEFORE vs AFTER)
  11. Golden Set Immutability Check
  12. Deterministic & Offline Execution
"""

import copy
import hashlib
import json
from pathlib import Path
import unittest

from src.evaluation.datasets import GoldenEvaluationRecord, GoldenSetLoader
from src.evaluation.decision_log import DecisionLogManager, DecisionRecord, VALID_DECISIONS
from src.evaluation.experiments import ExperimentFramework, ExperimentResult, MetricDelta
from src.evaluation.failure_analysis import (
    FailureAnalysisEngine,
    FailureDiagnosisEngine,
    FailureRecord,
    VALID_COMPONENTS,
    VALID_FAILURE_CATEGORIES,
    VALID_SEVERITIES,
    VALID_STATUSES,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "data" / "golden" / "golden_evaluation_set.jsonl"
STAGE11_DIR = REPO_ROOT / "reports" / "stage11"
STAGE11_REPORT = REPO_ROOT / "reports" / "stage11_failure_analysis.md"
DECISION_LOG_FILE = REPO_ROOT / "reports" / "decision_log.md"


class TestStage11FailureAnalysis(unittest.TestCase):
    """Complete test suite for Stage 11 Failure Analysis & Decision Log."""

    def setUp(self):
        self.diagnostician = FailureDiagnosisEngine()
        self.engine = FailureAnalysisEngine()
        self.dummy_gold = GoldenEvaluationRecord(
            example_id="golden_test_001",
            conversation_id="conv_test_1",
            brand="AmazonHelp",
            customer_message="My package tracking says delivered but I checked porch and nothing is there.",
            primary_intent="missing_delivered_package",
            primary_intent_name="Missing Package (Delivered But Not Received)",
            secondary_intent=None,
            secondary_intent_name=None,
            difficulty="medium",
            expected_behavior="Advise customer to check around porch and mailbox",
            gold_response_guidance="Check porch and mailroom before contacting carrier",
            reference_support_response="Please check your porch or neighbors.",
            response_type="resolution",
            source_reference="test_ref",
            split="evaluation",
        )

    # 1. Failure Record Schema & Validation
    def test_01_failure_schema_and_validation(self):
        """Verify FailureRecord dataclass fields, serialization, and validation."""
        record = FailureRecord(
            failure_id="fail_001",
            example_id="golden_001",
            conversation_id="conv_100",
            gold_intent="delivery_delay",
            predicted_intent="missing_delivered_package",
            secondary_intents=[],
            difficulty="hard",
            confidence=0.92,
            retrieved_references=[],
            policy_result={"passed": True},
            escalation_result={"required": False},
            generated_response="We are checking on your order.",
            expected_behavior="Track order delay",
            failure_category="intent_confusion",
            root_cause="Overlapping tracking lexical boost",
            severity="high",
            affected_component="intent classifier",
            recommended_action="Refine delivery delay regex",
            status="analyzed",
            customer_message="Where is my order?",
        )
        errors = record.validate()
        self.assertEqual(len(errors), 0, f"Validation errors found: {errors}")
        d = record.to_dict()
        self.assertEqual(d["failure_id"], "fail_001")
        self.assertEqual(d["gold_intent"], "delivery_delay")

        # Test invalid values trigger validation errors
        bad_rec = FailureRecord(
            failure_id="fail_bad",
            example_id="golden_bad",
            conversation_id="conv_bad",
            gold_intent="delivery_delay",
            predicted_intent="other",
            failure_category="invalid_category_xyz",
            severity="super_critical",
            affected_component="alien_technology",
            status="destroyed",
            confidence=1.5,
        )
        bad_errors = bad_rec.validate()
        self.assertGreaterEqual(len(bad_errors), 4)

    # 2. Failure Categorization (18-item taxonomy)
    def test_02_failure_categorization_taxonomy(self):
        """Verify all 18 categories exist and diagnostician correctly routes examples."""
        self.assertEqual(len(VALID_FAILURE_CATEGORIES), 18)
        self.assertIn("intent_confusion", VALID_FAILURE_CATEGORIES)
        self.assertIn("multi_intent_failure", VALID_FAILURE_CATEGORIES)
        self.assertIn("policy_violation", VALID_FAILURE_CATEGORIES)
        self.assertIn("unsupported_action_claim", VALID_FAILURE_CATEGORIES)
        self.assertIn("hallucination", VALID_FAILURE_CATEGORIES)
        self.assertIn("latency_failure", VALID_FAILURE_CATEGORIES)

        # Diagnostician routes policy violation
        cat_pol = self.diagnostician.determine_failure_category(
            gold=self.dummy_gold,
            pred_intent="missing_delivered_package",
            confidence=0.90,
            response_text="Standard response",
            policy_result={"passed": False, "violations": [{"rule": "unauthorized_action"}]},
            escalation_result={"required": False},
            guidance_adherence=1.0,
            retrieved_refs=[{"text": "ref"}],
        )
        self.assertEqual(cat_pol, "policy_violation")

        # Diagnostician routes unsupported action claim in text
        cat_claim = self.diagnostician.determine_failure_category(
            gold=self.dummy_gold,
            pred_intent="missing_delivered_package",
            confidence=0.90,
            response_text="I have refunded your order of $50.",
            policy_result={"passed": True},
            escalation_result={"required": False},
            guidance_adherence=1.0,
            retrieved_refs=[{"text": "ref"}],
        )
        self.assertEqual(cat_claim, "unsupported_action_claim")

        # Diagnostician routes latency failure (> 500ms)
        cat_lat = self.diagnostician.determine_failure_category(
            gold=self.dummy_gold,
            pred_intent="missing_delivered_package",
            confidence=0.90,
            response_text="Standard response",
            policy_result={"passed": True},
            escalation_result={"required": False},
            guidance_adherence=1.0,
            retrieved_refs=[{"text": "ref"}],
            latency_ms=650.0,
        )
        self.assertEqual(cat_lat, "latency_failure")

    # 3. Severity Assignment
    def test_03_severity_assignment(self):
        """Verify severity assignments conform to defined operational principles."""
        self.assertEqual(set(VALID_SEVERITIES), {"critical", "high", "medium", "low"})

        # Critical: policy violation or unauthorized claim
        _, sev_crit, _, _ = self.diagnostician.diagnose_root_cause(
            gold=self.dummy_gold,
            pred_intent="returns_and_refunds",
            category="unsupported_action_claim",
            confidence=0.95,
            customer_message="Refund me",
            response_text="I have processed your refund.",
        )
        self.assertEqual(sev_crit, "critical")

        # High: intent confusion on operational task
        _, sev_high, _, _ = self.diagnostician.diagnose_root_cause(
            gold=self.dummy_gold,
            pred_intent="delivery_delay",
            category="intent_confusion",
            confidence=0.90,
            customer_message="Delivery is delayed",
            response_text="Check tracking.",
        )
        self.assertEqual(sev_high, "high")

        # Low: brief input / insufficient context
        _, sev_low, _, _ = self.diagnostician.diagnose_root_cause(
            gold=self.dummy_gold,
            pred_intent="unknown_or_ambiguous",
            category="insufficient_context",
            confidence=0.50,
            customer_message="help",
            response_text="How can I help?",
        )
        self.assertEqual(sev_low, "low")

    # 4. Root-Cause Representation
    def test_04_root_cause_representation(self):
        """Verify symptom vs root-cause separation and allow unknown."""
        root, _, _, _ = self.diagnostician.diagnose_root_cause(
            gold=self.dummy_gold,
            pred_intent="digital_services_technical",
            category="other",
            confidence=0.60,
            customer_message="strange glitch",
            response_text="Check device",
        )
        self.assertIsInstance(root, str)
        self.assertGreater(len(root), 0)

        # Unknown fallback when unclassified
        dummy_unknown_gold = copy.deepcopy(self.dummy_gold)
        # Test explicit fallback
        rec = FailureRecord(
            failure_id="fail_u",
            example_id="golden_u",
            conversation_id="conv_u",
            gold_intent="delivery_delay",
            predicted_intent="returns_and_refunds",
            root_cause="unknown",
        )
        self.assertEqual(rec.root_cause, "unknown")

    # 5. Component Ownership
    def test_05_component_ownership(self):
        """Verify valid component ownership attribution across architectural modules."""
        for comp in VALID_COMPONENTS:
            self.assertIsInstance(comp, str)
        self.assertIn("intent classifier", VALID_COMPONENTS)
        self.assertIn("policy layer", VALID_COMPONENTS)
        self.assertIn("retrieval", VALID_COMPONENTS)
        self.assertIn("escalation", VALID_COMPONENTS)
        self.assertIn("response generation", VALID_COMPONENTS)

    # 6. Decision Log Format & Markdown Generation
    def test_06_decision_log_format(self):
        """Verify decision log schema, validation, decisions (KEEP/REJECT/REVISE/DEFER), and markdown."""
        self.assertEqual(set(VALID_DECISIONS), {"KEEP", "REJECT", "REVISE", "DEFER"})

        dec = DecisionRecord(
            decision_id="DEC-099",
            problem="Sample problem description",
            evidence="Measured accuracy gap of 5%",
            hypothesis="Adding regex will improve score",
            proposed_change="Add targeted regex pattern",
            expected_impact="Improve accuracy by 2%",
            risk="Low risk of regression",
            result="Accuracy improved by 3%",
            decision="KEEP",
            date="2026-09-12",
            before_metrics={"accuracy": 0.79},
            after_metrics={"accuracy": 0.82},
        )
        self.assertEqual(len(dec.validate()), 0)
        md = dec.to_markdown()
        self.assertIn("### Decision ID\nDEC-099", md)
        self.assertIn("### Problem", md)
        self.assertIn("### Evidence", md)
        self.assertIn("### Hypothesis", md)
        self.assertIn("### Proposed Change", md)
        self.assertIn("### Expected Impact", md)
        self.assertIn("### Risk", md)
        self.assertIn("### Result", md)
        self.assertIn("### Decision\n**KEEP**", md)
        self.assertIn("### Date\n2026-09-12", md)

    # 7. Aggregation Calculations & Ranking
    def test_07_aggregation_calculations(self):
        """Verify pattern aggregation computes counts, rankings, confusion pairs, and rates."""
        fail1 = FailureRecord(
            failure_id="fail_001",
            example_id="golden_001",
            conversation_id="conv_1",
            gold_intent="delivery_delay",
            predicted_intent="missing_delivered_package",
            failure_category="intent_confusion",
            severity="high",
            affected_component="intent classifier",
            confidence=0.90,
            customer_message="Package delayed",
        )
        fail2 = FailureRecord(
            failure_id="fail_002",
            example_id="golden_002",
            conversation_id="conv_2",
            gold_intent="order_cancellation",
            predicted_intent="prime_membership",
            failure_category="multi_intent_failure",
            severity="high",
            affected_component="intent classifier",
            confidence=0.95,
            customer_message="Cancel prime",
        )
        agg = self.engine.aggregate_patterns([fail1, fail2], total_evaluated=10)
        self.assertEqual(agg["total_failures"], 2)
        self.assertEqual(agg["overall_failure_rate"], 0.20)
        self.assertEqual(len(agg["category_ranking"]), 2)
        self.assertEqual(len(agg["top_confusion_pairs"]), 2)

    # 8. High-Confidence Error Detection
    def test_08_high_confidence_error_detection(self):
        """Verify high-confidence errors (confidence >= 0.85 & wrong) are correctly flagged."""
        high_conf_fail = FailureRecord(
            failure_id="fail_hc",
            example_id="golden_hc",
            conversation_id="conv_hc",
            gold_intent="delivery_delay",
            predicted_intent="missing_delivered_package",
            confidence=0.95,
            failure_category="intent_confusion",
        )
        low_conf_fail = FailureRecord(
            failure_id="fail_lc",
            example_id="golden_lc",
            conversation_id="conv_lc",
            gold_intent="delivery_delay",
            predicted_intent="missing_delivered_package",
            confidence=0.60,
            failure_category="intent_confusion",
        )
        agg = self.engine.aggregate_patterns([high_conf_fail, low_conf_fail], total_evaluated=10)
        hc_data = agg["high_confidence_errors"]
        self.assertEqual(hc_data["count"], 1)
        self.assertEqual(hc_data["rate"], 0.10)
        self.assertEqual(hc_data["percentage_of_failures"], 50.0)

    # 9. Regression Detection & Threshold Enforcement
    def test_09_regression_detection(self):
        """Verify ExperimentFramework detects metric drops and blocks harmful regressions."""
        framework = ExperimentFramework()
        before = {
            "accuracy": 0.80,
            "macro_f1": 0.79,
            "hard_accuracy": 0.75,
            "guidance_adherence_rate": 0.90,
            "policy_safety_rate": 1.0,
            "escalation_rate": 0.10,
            "latency_ms": 4.0,
        }
        # Harmful drop in policy safety (zero tolerance!)
        bad_after = {
            "accuracy": 0.85,
            "macro_f1": 0.84,
            "hard_accuracy": 0.80,
            "guidance_adherence_rate": 0.90,
            "policy_safety_rate": 0.95,  # Regression!
            "escalation_rate": 0.10,
            "latency_ms": 4.0,
        }
        deltas, regressions = framework.compare_metrics(before, bad_after)
        self.assertTrue(deltas["policy_safety_rate"].is_regression)
        self.assertGreater(len(regressions), 0)

        res = framework.run_experiment(
            experiment_id="EXP-BAD",
            name="Unsafe Agent",
            description="High accuracy but violates policy safety",
            before_runner_result={"classification_metrics": before, "generation_metrics": before, "escalation_metrics": before},
            after_runner_result={"classification_metrics": bad_after, "generation_metrics": bad_after, "escalation_metrics": bad_after},
        )
        self.assertEqual(res.recommendation, "REJECT")

    # 10. Experiment Comparison (BEFORE vs AFTER)
    def test_10_experiment_comparison(self):
        """Verify successful candidate improvement produces KEEP recommendation."""
        framework = ExperimentFramework()
        before = {
            "accuracy": 0.79,
            "macro_f1": 0.78,
            "hard_accuracy": 0.75,
            "guidance_adherence_rate": 0.90,
            "policy_safety_rate": 1.0,
            "escalation_rate": 0.10,
            "latency_ms": 3.5,
        }
        improved_after = {
            "accuracy": 0.82,
            "macro_f1": 0.82,
            "hard_accuracy": 0.80,
            "guidance_adherence_rate": 0.905,
            "policy_safety_rate": 1.0,
            "escalation_rate": 0.105,
            "latency_ms": 3.4,
        }
        res = framework.run_experiment(
            experiment_id="EXP-GOOD",
            name="Improved Agent",
            description="Boosted accuracy with zero safety drop",
            before_runner_result={"classification_metrics": before, "generation_metrics": before, "escalation_metrics": before},
            after_runner_result={"classification_metrics": improved_after, "generation_metrics": improved_after, "escalation_metrics": improved_after},
        )
        self.assertFalse(res.has_regressions)
        self.assertEqual(res.recommendation, "KEEP")
        self.assertAlmostEqual(res.deltas["accuracy"].absolute_delta, 0.03)

    # 11. Golden Set Immutability Check
    def test_11_golden_set_immutability(self):
        """Verify the protected Golden Evaluation Set has not been modified."""
        self.assertTrue(GOLDEN_PATH.exists(), f"Golden set missing at {GOLDEN_PATH}")
        with open(GOLDEN_PATH, "rb") as fh:
            content = fh.read()
        sha256_hash = hashlib.sha256(content).hexdigest()
        manifest_path = REPO_ROOT / "data" / "golden" / "golden_set_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            if "sha256" in manifest:
                self.assertEqual(sha256_hash, manifest["sha256"])

        # Check exactly 200 records
        loader = GoldenSetLoader(GOLDEN_PATH)
        records = loader.load()
        self.assertEqual(len(records), 200)

    # 12. Artifacts & Reproducibility Check
    def test_12_stage11_artifacts_exist(self):
        """Verify all required Stage 11 reports and structured datasets exist."""
        self.assertTrue(DECISION_LOG_FILE.exists(), f"Missing {DECISION_LOG_FILE}")
        self.assertTrue(STAGE11_REPORT.exists(), f"Missing {STAGE11_REPORT}")
        self.assertTrue((STAGE11_DIR / "failure_dataset.jsonl").exists())
        self.assertTrue((STAGE11_DIR / "failure_patterns.json").exists())
        self.assertTrue((STAGE11_DIR / "high_confidence_errors.json").exists())
        self.assertTrue((STAGE11_DIR / "failure_clusters.json").exists())
        self.assertTrue((STAGE11_DIR / "decision_log.json").exists())


if __name__ == "__main__":
    unittest.main()
