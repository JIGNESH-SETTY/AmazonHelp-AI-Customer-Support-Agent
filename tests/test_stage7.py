"""
test_stage7.py
--------------
Verification and integrity tests for Stage 7: Golden Evaluation Set.

Validates:
  1. Artifacts existence (.jsonl, .csv, manifest.json)
  2. Dataset schema validity and required non-empty fields
  3. Unique example_id across all benchmark records
  4. Unique conversation_id across all benchmark records
  5. Zero duplicate customer messages
  6. Valid intent IDs matching Stage 6 taxonomy
  7. Valid difficulty labels (easy, medium, hard)
  8. Canonical intent coverage (all 10 canonical intents covered)
  9. Target dataset size constraint (200 records)
  10. Strict test split membership (zero train/val leakage)
  11. CSV and JSONL record parity
  12. Manifest distribution integrity against dataset content
"""

from collections import Counter
import csv
import json
from pathlib import Path
import unittest

from src.evaluation.validate_golden_set import (
    REQUIRED_FIELDS,
    VALID_DIFFICULTIES,
    VALID_RESPONSE_TYPES,
    run_validation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"

GOLDEN_JSONL_PATH = GOLDEN_DIR / "golden_evaluation_set.jsonl"
GOLDEN_CSV_PATH = GOLDEN_DIR / "golden_evaluation_set.csv"
GOLDEN_MANIFEST_PATH = GOLDEN_DIR / "golden_set_manifest.json"
TAXONOMY_PATH = DATA_DIR / "intent_taxonomy.json"
LABELS_PATH = PROCESSED_DIR / "amazonhelp_intent_labels.jsonl"


class TestStage7GoldenEvaluationSet(unittest.TestCase):
    """Test suite validating Stage 7 Golden Evaluation Set integrity and quality."""

    @classmethod
    def setUpClass(cls):
        """Loads Golden Set artifacts and reference taxonomy."""
        assert GOLDEN_JSONL_PATH.exists(), f"Missing {GOLDEN_JSONL_PATH}"
        assert GOLDEN_CSV_PATH.exists(), f"Missing {GOLDEN_CSV_PATH}"
        assert GOLDEN_MANIFEST_PATH.exists(), f"Missing {GOLDEN_MANIFEST_PATH}"

        with open(TAXONOMY_PATH, encoding="utf-8") as fh:
            cls.taxonomy = json.load(fh)
        cls.canonical_intent_ids = {i["intent_id"] for i in cls.taxonomy["intents"]}
        cls.valid_intent_ids = set(cls.canonical_intent_ids)
        if "fallback_category" in cls.taxonomy:
            cls.valid_intent_ids.add(cls.taxonomy["fallback_category"]["intent_id"])

        with open(GOLDEN_MANIFEST_PATH, encoding="utf-8") as fh:
            cls.manifest = json.load(fh)

        cls.records = []
        with open(GOLDEN_JSONL_PATH, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    cls.records.append(json.loads(line))

        # Train and validation conversation IDs
        cls.train_val_convs = set()
        if LABELS_PATH.exists():
            with open(LABELS_PATH, encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    if row.get("split") in ("train", "val"):
                        cls.train_val_convs.add(row["conversation_id"])

    def test_01_artifacts_exist(self):
        """Verify all expected Golden Set files exist."""
        self.assertTrue(GOLDEN_JSONL_PATH.exists())
        self.assertTrue(GOLDEN_CSV_PATH.exists())
        self.assertTrue(GOLDEN_MANIFEST_PATH.exists())

    def test_02_target_size(self):
        """Verify dataset matches target size of 200 records."""
        self.assertEqual(len(self.records), 200)
        self.assertEqual(self.manifest["actual_size"], 200)

    def test_03_schema_and_required_fields(self):
        """Verify all required fields exist and are non-empty."""
        for idx, rec in enumerate(self.records, start=1):
            for field in REQUIRED_FIELDS:
                val = rec.get(field)
                self.assertIsNotNone(val, f"Record {idx}: Missing field '{field}'")
                if isinstance(val, str):
                    self.assertTrue(bool(val.strip()), f"Record {idx}: Empty string in field '{field}'")

    def test_04_unique_identifiers(self):
        """Verify unique example_id and unique conversation_id."""
        ex_ids = [r["example_id"] for r in self.records]
        conv_ids = [r["conversation_id"] for r in self.records]

        self.assertEqual(len(ex_ids), len(set(ex_ids)), "Duplicate example_id detected")
        self.assertEqual(len(conv_ids), len(set(conv_ids)), "Duplicate conversation_id detected")

    def test_05_no_duplicate_customer_messages(self):
        """Verify no duplicate customer messages exist in the golden set."""
        messages = [r["customer_message"].strip().lower() for r in self.records]
        self.assertEqual(len(messages), len(set(messages)), "Duplicate customer message detected")

    def test_06_valid_intent_references(self):
        """Verify all primary and secondary intent IDs reference valid taxonomy entries."""
        for idx, rec in enumerate(self.records, start=1):
            p_intent = rec["primary_intent"]
            self.assertIn(p_intent, self.valid_intent_ids, f"Record {idx}: Invalid primary intent '{p_intent}'")

            s_intent = rec.get("secondary_intent")
            if s_intent is not None:
                self.assertIn(s_intent, self.valid_intent_ids, f"Record {idx}: Invalid secondary intent '{s_intent}'")

    def test_07_canonical_intent_coverage(self):
        """Verify all 10 canonical intents from Stage 6 are adequately represented."""
        covered = {r["primary_intent"] for r in self.records}
        for intent_id in self.canonical_intent_ids:
            self.assertIn(intent_id, covered, f"Canonical intent '{intent_id}' is missing from golden set")

        # Each canonical intent should have at least 15 examples
        counts = Counter(r["primary_intent"] for r in self.records)
        for intent_id in self.canonical_intent_ids:
            self.assertGreaterEqual(counts[intent_id], 15, f"Insufficient examples for '{intent_id}': {counts[intent_id]}")

    def test_08_valid_difficulties_and_distribution(self):
        """Verify all difficulty labels are valid and cover easy, medium, and hard."""
        diff_counts = Counter(r["difficulty"] for r in self.records)
        for diff in diff_counts.keys():
            self.assertIn(diff, VALID_DIFFICULTIES)

        # All 3 difficulty tiers must be present
        self.assertGreater(diff_counts["easy"], 0)
        self.assertGreater(diff_counts["medium"], 0)
        self.assertGreater(diff_counts["hard"], 0)

    def test_09_split_integrity_and_zero_leakage(self):
        """Verify all records are from test split and zero leakage into train/val."""
        for idx, rec in enumerate(self.records, start=1):
            self.assertEqual(rec.get("split"), "test", f"Record {idx}: Expected split 'test'")
            conv_id = rec["conversation_id"]
            if self.train_val_convs:
                self.assertNotIn(conv_id, self.train_val_convs, f"Record {idx}: Leakage of conv '{conv_id}' into train/val!")

    def test_10_csv_parity(self):
        """Verify CSV export matches JSONL count and attributes."""
        with open(GOLDEN_CSV_PATH, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        self.assertEqual(len(rows), len(self.records), "CSV row count does not match JSONL row count")

    def test_11_manifest_consistency(self):
        """Verify manifest numbers match actual dataset counts."""
        self.assertEqual(self.manifest["actual_size"], len(self.records))
        self.assertEqual(self.manifest["intents_covered_count"], len({r["primary_intent"] for r in self.records}))
        self.assertEqual(self.manifest["canonical_intents_covered"], len(self.canonical_intent_ids))

        actual_intent_counts = dict(Counter(r["primary_intent"] for r in self.records))
        self.assertEqual(self.manifest["intent_distribution"], actual_intent_counts)

        actual_diff_counts = dict(Counter(r["difficulty"] for r in self.records))
        self.assertEqual(self.manifest["difficulty_distribution"], actual_diff_counts)

    def test_12_validation_script_pass(self):
        """Verify the dedicated validation runner returns PASS."""
        is_valid, metrics, errors = run_validation()
        self.assertTrue(is_valid, f"Validation script reported errors: {errors}")
        self.assertEqual(metrics["total_records"], 200)
        self.assertEqual(metrics["train_val_leakage"], 0)


if __name__ == "__main__":
    unittest.main()
