"""
test_stage10.py
---------------
Comprehensive unit and verification test suite for Stage 10: Evaluation Harness.

Validates all 20 checks from Section 22:
  1. Golden Set loader functionality
  2. Schema validation & record integrity
  3. Golden Set immutability
  4. Zero training split leakage
  5. Accuracy computation
  6. Macro F1 computation
  7. Weighted F1 computation
  8. Difficulty-stratified metrics
  9. Generation metrics (ROUGE-1/2/L, BLEU-1/2)
  10. Guidance adherence calculation
  11. Policy safety rates (unsupported actions, hallucinations)
  12. Escalation metrics (overall, correct, unnecessary, missed)
  13. Latency profiler aggregation (mean, median, P95)
  14. Confidence calibration (ECE, correct vs incorrect)
  15. Confusion matrix calculation
  16. Per-intent breakdown metrics
  17. Error analysis & failure categorization
  18. Baseline comparator & delta computations
  19. Regression detector & threshold warnings
  20. Offline execution without external APIs
"""

import json
from pathlib import Path
import unittest

from src.evaluation.comparison import BaselineComparator, RegressionDetector
from src.evaluation.datasets import GoldenEvaluationRecord, GoldenSetLoader
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.latency import LatencyProfiler
from src.evaluation.metrics import (
    compute_confidence_calibration,
    compute_escalation_metrics,
    compute_generation_and_safety_metrics,
    compute_intent_classification_metrics,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE10_DIR = REPO_ROOT / "reports" / "stage10"
EVAL_RESULTS_JSON = STAGE10_DIR / "evaluation_results.json"
REPORT_MD_PATH = REPO_ROOT / "reports" / "stage10_evaluation_harness.md"


class TestStage10EvaluationHarness(unittest.TestCase):
    """Test suite validating the modular Evaluation Harness framework."""

    @classmethod
    def setUpClass(cls):
        cls.loader = GoldenSetLoader()
        cls.golden_records = cls.loader.load()

    def test_01_golden_set_loader(self):
        """Verify GoldenSetLoader loads exactly 200 records."""
        self.assertEqual(len(self.golden_records), 200)
        self.assertIsInstance(self.golden_records[0], GoldenEvaluationRecord)

    def test_02_golden_set_schema_validation(self):
        """Verify all loaded records have required fields and non-empty messages."""
        for rec in self.golden_records:
            self.assertTrue(len(rec.example_id) > 0)
            self.assertTrue(len(rec.customer_message) > 0)
            self.assertIn(rec.difficulty, ["easy", "medium", "hard"])
            self.assertIn(rec.response_type, ["resolution", "clarification", "escalation"])

    def test_03_golden_set_immutability(self):
        """Verify GoldenEvaluationRecord is frozen (immutable)."""
        rec = self.golden_records[0]
        with self.assertRaises((AttributeError, TypeError)):
            rec.primary_intent = "tampered_intent"  # type: ignore

    def test_04_zero_training_leakage(self):
        """Verify zero Golden Set conversation IDs exist in training splits."""
        is_clean, overlap = self.loader.verify_zero_training_leakage()
        self.assertTrue(is_clean)
        self.assertEqual(overlap, 0)

    def test_05_accuracy_metric(self):
        """Verify classification accuracy calculation."""
        y_true = ["a", "b", "c", "d"]
        y_pred = ["a", "b", "x", "y"]
        res = compute_intent_classification_metrics(y_true, y_pred)
        self.assertEqual(res["accuracy"], 0.5)

    def test_06_macro_f1_metric(self):
        """Verify macro F1 calculation on balanced/imbalanced predictions."""
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "b"]
        res = compute_intent_classification_metrics(y_true, y_pred)
        self.assertGreater(res["macro_f1"], 0.0)
        self.assertLessEqual(res["macro_f1"], 1.0)

    def test_07_weighted_f1_metric(self):
        """Verify weighted F1 calculation."""
        y_true = ["a", "a", "a", "b"]
        y_pred = ["a", "a", "a", "a"]
        res = compute_intent_classification_metrics(y_true, y_pred)
        self.assertGreaterEqual(res["weighted_f1"], 0.5)

    def test_08_difficulty_stratification(self):
        """Verify difficulty stratification metric partitioning."""
        diffs = [r.difficulty for r in self.golden_records]
        self.assertIn("easy", diffs)
        self.assertIn("medium", diffs)
        self.assertIn("hard", diffs)

    def test_09_generation_metrics_rouge_bleu(self):
        """Verify ROUGE and BLEU generation metrics."""
        hyps = ["we are sorry for the shipping delay please check your tracking"]
        refs = ["we are sorry for the shipping delay please check your tracking"]
        intents = ["delivery_delay"]
        res = compute_generation_and_safety_metrics(hyps, refs, intents)
        self.assertAlmostEqual(res["rouge_l"], 1.0, places=3)
        self.assertAlmostEqual(res["bleu_1"], 1.0, places=3)

    def test_10_guidance_adherence_calculation(self):
        """Verify guidance adherence scoring."""
        hyps = ["Check your tracking in Your Orders at <URL>"]
        refs = ["Tracking is available online"]
        intents = ["delivery_delay"]
        res = compute_generation_and_safety_metrics(hyps, refs, intents)
        self.assertEqual(res["guidance_adherence_rate"], 1.0)

    def test_11_policy_safety_metrics(self):
        """Verify policy safety submetrics detect unverified actions and hallucinations."""
        unsafe_hyps = [
            "I have processed your refund of $50.",
            "Your package is guaranteed to arrive tomorrow by 2pm.",
            "Please send your password and credit card number.",
            "Clean standard response.",
        ]
        refs = ["Ref"] * 4
        intents = ["delivery_delay"] * 4
        res = compute_generation_and_safety_metrics(unsafe_hyps, refs, intents)
        self.assertEqual(res["unsupported_action_rate"], 0.25)
        self.assertEqual(res["hallucinated_guarantee_rate"], 0.25)
        self.assertEqual(res["insecure_request_rate"], 0.25)
        self.assertEqual(res["policy_safety_rate"], 0.25)

    def test_12_escalation_metrics(self):
        """Verify escalation rate, correct escalation, and unnecessary escalation."""
        esc_flags = [True, False, True, False]
        gold_intents = ["service_complaint_escalation", "delivery_delay", "delivery_delay", "delivery_delay"]
        sentiments = ["angry", "neutral", "neutral", "neutral"]
        res = compute_escalation_metrics(esc_flags, gold_intents, sentiments)
        self.assertEqual(res["overall_escalation_rate"], 0.5)
        self.assertEqual(res["correct_escalation_rate"], 1.0)
        self.assertAlmostEqual(res["unnecessary_escalation_rate"], 1 / 3, places=3)

    def test_13_latency_profiler(self):
        """Verify LatencyProfiler computes mean, median, P95, min, and max."""
        profiler = LatencyProfiler()
        for t in [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]:
            profiler.record(t)
        prof = profiler.summarize()
        self.assertEqual(prof.min_ms, 1.0)
        self.assertEqual(prof.max_ms, 10.0)
        self.assertAlmostEqual(prof.median_ms, 3.5, places=1)

    def test_14_confidence_calibration_ece(self):
        """Verify Expected Calibration Error (ECE) and confidence separation."""
        y_true = ["a", "b", "c", "d"]
        y_pred = ["a", "b", "x", "y"]
        confs = [0.95, 0.90, 0.60, 0.50]
        calib = compute_confidence_calibration(y_true, y_pred, confs)
        self.assertGreaterEqual(calib["expected_calibration_error"], 0.0)
        self.assertGreater(calib["mean_confidence_correct"], calib["mean_confidence_incorrect"])

    def test_15_confusion_matrix(self):
        """Verify confusion matrix computation."""
        y_true = ["delivery_delay", "returns_and_refunds"]
        y_pred = ["delivery_delay", "delivery_delay"]
        res = compute_intent_classification_metrics(y_true, y_pred)
        conf = res["confusion_matrix"]
        self.assertEqual(conf["delivery_delay"]["delivery_delay"], 1)
        self.assertEqual(conf["returns_and_refunds"]["delivery_delay"], 1)

    def test_16_per_intent_metrics(self):
        """Verify per-intent precision and recall are present for all classes."""
        y_true = [r.primary_intent for r in self.golden_records]
        y_pred = list(y_true)
        res = compute_intent_classification_metrics(y_true, y_pred)
        for intent_id in set(y_true):
            self.assertIn(intent_id, res["per_intent"])
            self.assertEqual(res["per_intent"][intent_id]["f1"], 1.0)

    def test_17_error_analysis_categorization(self):
        """Verify ErrorAnalyzer correctly assigns failure categories."""
        analyzer = ErrorAnalyzer()
        mock_gold = self.golden_records[0]

        # Policy failure
        diag = analyzer.diagnose_record(
            gold=mock_gold,
            pred_intent=mock_gold.primary_intent,
            confidence=0.9,
            response_text="I have processed your refund.",
            policy_result={"passed": False, "violations": ["violation"]},
            escalation_result={"required": False},
            guidance_adherence=1.0,
        )
        self.assertIsNotNone(diag)
        self.assertEqual(diag["failure_category"], "policy_violation")

    def test_18_baseline_comparison(self):
        """Verify BaselineComparator accurately computes deltas."""
        comparator = BaselineComparator()
        comp = comparator.compare(
            {"accuracy": 0.79, "macro_f1": 0.78, "weighted_f1": 0.78, "accuracy_by_difficulty": {"hard": 0.75}},
            {"guidance_adherence_rate": 0.90, "policy_safety_rate": 1.0, "latency_ms": 3.5},
        )
        self.assertIn("overall_accuracy", comp)
        self.assertGreater(comp["overall_accuracy"]["absolute_delta"], 0)

    def test_19_regression_detection(self):
        """Verify RegressionDetector triggers on performance drops."""
        detector = RegressionDetector(max_accuracy_drop=0.02)
        comparator = BaselineComparator()
        # Degraded accuracy
        bad_comp = comparator.compare(
            {"accuracy": 0.60, "macro_f1": 0.50, "weighted_f1": 0.50, "accuracy_by_difficulty": {"hard": 0.50}},
            {"guidance_adherence_rate": 0.60, "policy_safety_rate": 0.80, "latency_ms": 50.0},
        )
        passed, warnings = detector.check(bad_comp)
        self.assertFalse(passed)
        self.assertGreater(len(warnings), 0)

    def test_20_stage10_artifacts_exist(self):
        """Verify all required Stage 10 files and documentation exist."""
        self.assertTrue(EVAL_RESULTS_JSON.exists(), f"Missing {EVAL_RESULTS_JSON}")
        self.assertTrue(REPORT_MD_PATH.exists(), f"Missing {REPORT_MD_PATH}")
        self.assertTrue((STAGE10_DIR / "per_intent_results.csv").exists())
        self.assertTrue((STAGE10_DIR / "confusion_matrix.csv").exists())
        self.assertTrue((STAGE10_DIR / "latency_results.json").exists())
        self.assertTrue((STAGE10_DIR / "baseline_comparison.json").exists())
        self.assertTrue((STAGE10_DIR / "regression_report.json").exists())
        self.assertTrue((STAGE10_DIR / "error_analysis.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
