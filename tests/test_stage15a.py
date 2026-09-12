"""
tests/test_stage15a.py
----------------------
STAGE 15A: Manual Golden Evaluation Set Review Helper Tests
"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.review_golden_set import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TAXONOMY_PATH,
    apply_batch_decisions,
    get_pending_and_reviewed_counts,
    load_manifest,
    load_taxonomy_intents,
    parse_compact_batch_input,
    save_manifest,
    validate_manifest_state,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestStage15AManualReviewHelper(unittest.TestCase):
    """Test suite for manual review helper, compact parser, and manifest integrity."""

    @classmethod
    def setUpClass(cls):
        cls.taxonomy = load_taxonomy_intents(DEFAULT_TAXONOMY_PATH)
        cls.valid_intent_ids = {item["intent_id"] for item in cls.taxonomy}

    def test_01_manifest_preservation_and_progress(self):
        """Verify the 200-example review manifest preserves reviewed rows (200 reviewed, 0 pending)."""
        fieldnames, rows = load_manifest(DEFAULT_MANIFEST_PATH)
        self.assertEqual(len(rows), 200, f"Expected 200 rows, got {len(rows)}")

        reviewed, pending = get_pending_and_reviewed_counts(rows)
        self.assertEqual(reviewed, 200, f"Expected 200 reviewed examples, found {reviewed}")
        self.assertEqual(pending, 0, f"Expected 0 pending examples, found {pending}")
        self.assertEqual(reviewed + pending, 200)

        # Check that all 200 reviewed examples have valid status and verified intents
        for r in rows:
            self.assertEqual(r.get("human_review_status", "").strip(), "VERIFIED")
            v_intent = r.get("human_verified_intent", "").strip()
            self.assertIn(v_intent, self.valid_intent_ids)
            self.assertTrue(bool(r.get("customer_message", "").strip()))

    def test_02_taxonomy_loading(self):
        """Verify that taxonomy loader retrieves all 11 valid intents with complete descriptions."""
        intents = load_taxonomy_intents(DEFAULT_TAXONOMY_PATH)
        self.assertEqual(len(intents), 11, f"Expected 11 intents, got {len(intents)}")

        intent_ids = [item["intent_id"] for item in intents]
        self.assertIn("delivery_delay", intent_ids)
        self.assertIn("unknown_or_ambiguous", intent_ids)
        self.assertIn("service_complaint_escalation", intent_ids)
        self.assertIn("damaged_defective_item", intent_ids)

        for item in intents:
            self.assertTrue(bool(item["intent_id"].strip()))
            self.assertTrue(bool(item["name"].strip()))
            self.assertTrue(bool(item["definition"].strip()))

    def test_03_compact_input_range_parsing(self):
        """Verify range parsing such as '1-10', '1-5', and 'all'."""
        # 1. Full range '1-10'
        dec, err = parse_compact_batch_input("1-10", 10, self.taxonomy)
        self.assertEqual(err, [])
        self.assertEqual(len(dec), 10)
        for i in range(1, 11):
            self.assertEqual(dec[i], ("CURRENT", ""))

        # 2. Sub-range '1-5'
        dec, err = parse_compact_batch_input("1-5", 10, self.taxonomy)
        self.assertEqual(err, [])
        self.assertEqual(len(dec), 5)
        for i in range(1, 6):
            self.assertEqual(dec[i], ("CURRENT", ""))
        self.assertNotIn(6, dec)

        # 3. Keyword 'all'
        dec, err = parse_compact_batch_input("all", 10, self.taxonomy)
        self.assertEqual(err, [])
        self.assertEqual(len(dec), 10)

    def test_04_compact_input_selected_examples(self):
        """Verify comma-separated selected examples such as '1,3,5,7'."""
        dec, err = parse_compact_batch_input("1,3,5,7", 10, self.taxonomy)
        self.assertEqual(err, [])
        self.assertEqual(set(dec.keys()), {1, 3, 5, 7})
        for k in [1, 3, 5, 7]:
            self.assertEqual(dec[k], ("CURRENT", ""))

    def test_05_compact_input_corrections(self):
        """Verify single and multiple corrections such as '3=11' and '2=10,5=3'."""
        # 1. Single correction '3=11'
        dec, err = parse_compact_batch_input("3=11", 10, self.taxonomy)
        self.assertEqual(err, [])
        self.assertEqual(len(dec), 1)
        self.assertEqual(dec[3], ("unknown_or_ambiguous", ""))

        # 2. Multiple corrections '2=10,5=3'
        dec, err = parse_compact_batch_input("2=10,5=3", 10, self.taxonomy)
        self.assertEqual(err, [])
        self.assertEqual(len(dec), 2)
        self.assertEqual(dec[2], ("service_complaint_escalation", ""))
        self.assertEqual(dec[5], ("returns_and_refunds", ""))

        # 3. Direct intent_id string
        dec, err = parse_compact_batch_input("4=order_cancellation", 10, self.taxonomy)
        self.assertEqual(err, [])
        self.assertEqual(dec[4], ("order_cancellation", ""))

    def test_06_mixed_confirmation_and_corrections(self):
        """Verify mixed syntax like '1-6,8-10;7=5' and optional notes."""
        # 1. '1-6,8-10;7=5'
        dec, err = parse_compact_batch_input("1-6,8-10;7=5", 10, self.taxonomy)
        self.assertEqual(err, [])
        self.assertEqual(len(dec), 10)
        for i in [1, 2, 3, 4, 5, 6, 8, 9, 10]:
            self.assertEqual(dec[i], ("CURRENT", ""))
        self.assertEqual(dec[7], ("damaged_defective_item", ""))

        # 2. '1-10;7=5' (override syntax)
        dec, err = parse_compact_batch_input("1-10;7=5", 10, self.taxonomy)
        self.assertEqual(err, [])
        self.assertEqual(len(dec), 10)
        self.assertEqual(dec[7], ("damaged_defective_item", ""))
        self.assertEqual(dec[1], ("CURRENT", ""))

        # 3. Optional note via '#'
        dec, err = parse_compact_batch_input("1-6,8-10;7=5#customer noted broken lid", 10, self.taxonomy)
        self.assertEqual(err, [])
        self.assertEqual(dec[7], ("damaged_defective_item", "customer noted broken lid"))
        self.assertEqual(dec[1], ("CURRENT", ""))

    def test_07_invalid_input_rejection(self):
        """Verify invalid intent numbers, out-of-range indices, and malformed syntax are rejected."""
        # Invalid intent number 99
        dec, err = parse_compact_batch_input("3=99", 10, self.taxonomy)
        self.assertEqual(dec, {})
        self.assertTrue(any("Invalid intent" in e for e in err))

        # Index out of bounds (15 > 10)
        dec, err = parse_compact_batch_input("15=5", 10, self.taxonomy)
        self.assertEqual(dec, {})
        self.assertTrue(any("out of range" in e for e in err))

        # Invalid range (8-3)
        dec, err = parse_compact_batch_input("8-3", 10, self.taxonomy)
        self.assertEqual(dec, {})
        self.assertTrue(any("Invalid range" in e for e in err))

        # Malformed token
        dec, err = parse_compact_batch_input("random_gibberish", 10, self.taxonomy)
        self.assertEqual(dec, {})
        self.assertTrue(any("Unrecognized token" in e for e in err))

    def test_08_apply_batch_decisions_and_atomic_save(self):
        """Verify apply_batch_decisions updates rows correctly and save_manifest persists atomically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_manifest.csv"
            fieldnames = [
                "example_id",
                "customer_message",
                "current_intent",
                "human_verified_intent",
                "expected_behavior",
                "gold_guidance",
                "human_review_status",
                "reviewer_notes",
            ]
            sample_rows = [
                {
                    "example_id": f"golden_{i:03d}",
                    "customer_message": f"message_{i}",
                    "current_intent": "delivery_delay",
                    "human_verified_intent": "",
                    "expected_behavior": "exp",
                    "gold_guidance": "guid",
                    "human_review_status": "PENDING_REVIEW",
                    "reviewer_notes": "",
                }
                for i in range(1, 11)
            ]

            batch_indices = list(range(10))
            # Command: 1-6,8-10;7=5#broken lid
            decisions, errors = parse_compact_batch_input("1-6,8-10;7=5#broken lid", 10, self.taxonomy)
            self.assertEqual(errors, [])

            applied = apply_batch_decisions(sample_rows, batch_indices, decisions)
            self.assertEqual(applied, 10)

            # Check in-memory state
            for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]:
                self.assertEqual(sample_rows[i]["human_review_status"], "VERIFIED")
                self.assertEqual(sample_rows[i]["human_verified_intent"], "delivery_delay")
            self.assertEqual(sample_rows[6]["human_review_status"], "VERIFIED")
            self.assertEqual(sample_rows[6]["human_verified_intent"], "damaged_defective_item")
            self.assertEqual(sample_rows[6]["reviewer_notes"], "broken lid")

            # Save and verify reload
            save_manifest(tmp_path, fieldnames, sample_rows)
            loaded_fields, loaded_rows = load_manifest(tmp_path)
            self.assertEqual(len(loaded_rows), 10)
            rev, pend = get_pending_and_reviewed_counts(loaded_rows)
            self.assertEqual(rev, 10)
            self.assertEqual(pend, 0)


if __name__ == "__main__":
    unittest.main()
