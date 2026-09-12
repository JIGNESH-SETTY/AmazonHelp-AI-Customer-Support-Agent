"""
tests/test_stage15.py
---------------------
STAGE 15: Golden Evaluation Set Integrity & Human-LLM Judge Agreement Tests
"""

import csv
import json
from pathlib import Path
import unittest

from src.evaluation.datasets import GoldenSetLoader
from src.evaluation.judge_agreement import (
    compute_adjacent_agreement,
    compute_cohens_kappa,
    compute_exact_agreement,
    compute_mean_absolute_difference,
    compute_pass_fail_agreement,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
STAGE15_DIR = REPO_ROOT / "reports" / "stage15"


class TestStage15GoldenSetAndAgreement(unittest.TestCase):
    """Test suite for Stage 15 verification and agreement calculations."""

    def test_01_golden_set_size_and_range(self):
        """Verify that the Golden Evaluation Set contains between 150 and 250 examples."""
        loader = GoldenSetLoader()
        records = loader.load()
        self.assertGreaterEqual(len(records), 150)
        self.assertLessEqual(len(records), 250)
        self.assertEqual(len(records), 200)

    def test_02_human_review_manifest_schema(self):
        """Verify the 200-example manual review manifest exists and conforms to schema."""
        manifest_path = GOLDEN_DIR / "golden_human_review_manifest.csv"
        self.assertTrue(manifest_path.exists(), f"Missing {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        required_cols = [
            "example_id",
            "customer_message",
            "current_intent",
            "human_verified_intent",
            "expected_behavior",
            "gold_guidance",
            "human_review_status",
            "reviewer_notes",
        ]
        for col in required_cols:
            self.assertIn(col, fieldnames)

        self.assertEqual(len(rows), 200)
        for r in rows:
            self.assertTrue(r["example_id"].startswith("golden_"))
            self.assertTrue(bool(r["customer_message"].strip()))
            self.assertTrue(bool(r["current_intent"].strip()))

    def test_03_adjacent_agreement_calculation(self):
        """Verify that adjacent agreement correctly accounts for off-by-one differences."""
        h1 = [5.0, 4.0, 3.0, 2.0, 1.0]
        j1 = [5.0, 3.0, 3.0, 1.0, 2.0]
        # Differences: |5-5|=0, |4-3|=1, |3-3|=0, |2-1|=1, |1-2|=1
        # All differences <= 1 -> 100% adjacent agreement
        self.assertAlmostEqual(compute_adjacent_agreement(h1, j1), 1.0)

        h2 = [5.0, 1.0]
        j2 = [3.0, 1.0]
        # Differences: |5-3|=2 (fail), |1-1|=0 (pass) -> 1/2 = 0.50
        self.assertAlmostEqual(compute_adjacent_agreement(h2, j2), 0.5)

    def test_04_agreement_study_artifacts(self):
        """Verify that Stage 15 agreement study files exist and contain valid results."""
        agreement_json = STAGE15_DIR / "judge_agreement.json"
        self.assertTrue(agreement_json.exists(), f"Missing {agreement_json}")

        with open(agreement_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.assertEqual(data["status"], "completed")
        self.assertGreaterEqual(data["sample_size"], 30)
        self.assertLessEqual(data["sample_size"], 50)

        metrics = data["metrics"]
        self.assertGreater(metrics["pass_fail_agreement_rate"], 0.70)
        self.assertGreater(metrics["cohens_kappa_pass_fail"], 0.40)
        self.assertIn("per_criterion", metrics)
        for crit in ["helpfulness", "grounding", "actionability", "clarity", "overall"]:
            self.assertIn(crit, metrics["per_criterion"])
            self.assertIn("adjacent_agreement", metrics["per_criterion"][crit])


if __name__ == "__main__":
    unittest.main()
