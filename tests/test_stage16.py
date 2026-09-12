"""
tests/test_stage16.py
---------------------
STAGE 16: Golden Evaluation Set Decision Log Tests
"""

import json
from pathlib import Path
import unittest

from src.evaluation.golden_decision_log import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TAXONOMY_PATH,
    build_golden_decision_log,
    load_taxonomy_data,
    validate_decision_log,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE16_DIR = REPO_ROOT / "reports" / "stage16"


class TestStage16GoldenDecisionLog(unittest.TestCase):
    """Test suite for Stage 16 Decision Log integrity and summary statistics."""

    @classmethod
    def setUpClass(cls):
        cls.log_data = build_golden_decision_log()
        cls.id_to_name, cls.id_to_meta = load_taxonomy_data()
        cls.valid_intent_ids = set(cls.id_to_name.keys())

    def test_01_all_200_golden_ids_accounted_for(self):
        """Verify that all 200 golden examples exist sequentially from golden_001 to golden_200."""
        records = self.log_data["records"]
        self.assertEqual(len(records), 200)

        expected_ids = [f"golden_{i:03d}" for i in range(1, 201)]
        actual_ids = [r["example_id"] for r in records]

        self.assertEqual(actual_ids, expected_ids)
        self.assertEqual(len(set(actual_ids)), 200, "Duplicate golden IDs detected")

    def test_02_zero_pending_examples(self):
        """Verify that zero pending examples remain and all 200 are verified."""
        summary = self.log_data["summary"]
        self.assertEqual(summary["total_examples"], 200)
        self.assertEqual(summary["verified_examples"], 200)
        self.assertEqual(summary["pending_examples"], 0)

    def test_03_intent_names_and_taxonomy_validity(self):
        """Verify that all original and verified intent IDs and names match canonical taxonomy."""
        for r in self.log_data["records"]:
            orig = r["original_intent"]
            fin = r["final_verified_intent"]

            self.assertIn(orig, self.valid_intent_ids, f"Invalid original intent: {orig}")
            self.assertIn(fin, self.valid_intent_ids, f"Invalid final verified intent: {fin}")

            self.assertEqual(r["original_intent_name"], self.id_to_name[orig])
            self.assertEqual(r["final_verified_intent_name"], self.id_to_name[fin])

            # Check decision consistency
            if r["is_changed"]:
                self.assertEqual(r["reviewer_decision"], "CHANGED")
                self.assertNotEqual(orig, fin)
            else:
                self.assertEqual(r["reviewer_decision"], "CONFIRMED")
                self.assertEqual(orig, fin)

    def test_04_summary_reconciliation(self):
        """Verify mathematical reconciliation of confirmed + changed = total."""
        summary = self.log_data["summary"]
        total = summary["total_examples"]
        conf = summary["confirmed_count"]
        ch = summary["changed_count"]

        self.assertEqual(conf + ch, total)
        self.assertEqual(total, 200)
        self.assertEqual(conf, 156)
        self.assertEqual(ch, 44)
        self.assertAlmostEqual(summary["confirmed_rate"], 0.78, places=2)
        self.assertAlmostEqual(summary["changed_rate"], 0.22, places=2)

        # Reconcile final distribution total
        final_dist = summary["final_intent_distribution"]
        self.assertEqual(sum(final_dist.values()), 200)

        # Reconcile original distribution total
        orig_dist = summary["original_intent_distribution"]
        self.assertEqual(sum(orig_dist.values()), 200)

    def test_05_decision_log_artifacts_exist_and_parse(self):
        """Verify that exported JSON and markdown decision logs exist and are valid."""
        json_file = STAGE16_DIR / "golden_decision_log.json"
        md_file = STAGE16_DIR / "golden_decision_log.md"
        report_file = STAGE16_DIR / "decision_log_report.md"

        self.assertTrue(json_file.exists(), f"Missing {json_file}")
        self.assertTrue(md_file.exists(), f"Missing {md_file}")
        self.assertTrue(report_file.exists(), f"Missing {report_file}")

        with open(json_file, "r", encoding="utf-8") as fh:
            loaded_json = json.load(fh)
        self.assertEqual(len(loaded_json["records"]), 200)

        with open(md_file, "r", encoding="utf-8") as fh:
            md_text = fh.read()
        self.assertIn("Golden Evaluation Set: Comprehensive Reviewer Decision Log", md_text)
        self.assertIn("golden_001", md_text)
        self.assertIn("golden_200", md_text)

    def test_06_transition_consistency(self):
        """Verify that every transition record corresponds to an actual intent change."""
        transitions = self.log_data["transitions"]
        self.assertEqual(len(transitions), self.log_data["summary"]["changed_count"])
        self.assertEqual(len(transitions), 44)

        for t in transitions:
            self.assertNotEqual(t["original_intent"], t["final_verified_intent"])
            self.assertTrue(t["example_id"].startswith("golden_"))
            self.assertTrue(bool(t["customer_message"].strip()))

    def test_07_validation_function_passes(self):
        """Verify that validate_decision_log passes cleanly on the actual decision log."""
        errors = validate_decision_log(self.log_data)
        self.assertEqual(errors, [], f"Validation reported unexpected errors: {errors}")


if __name__ == "__main__":
    unittest.main()
