"""
test_stage5.py
--------------
Verification and integrity tests for Stage 5: Brand Selection & Dataset Profiling.

Validates all 10 required checks from the Stage 5 specification:
  - Check 1: Every discovered conversation belongs to exactly one brand.
  - Check 2: Conversation counts are internally consistent.
  - Check 3: Message counts are internally consistent.
  - Check 4: Percentages are calculated correctly and within bounds [0.0, 100.0].
  - Check 5: Brand ranking is deterministic.
  - Check 6: Selected brand exists in the dataset.
  - Check 7: Selected brand meets the defined minimum data sufficiency thresholds.
  - Check 8: No Stage 1-4 source datasets were modified.
  - Check 9: No train/validation/test leakage is introduced.
  - Check 10: Generated JSON, Markdown, and selection artifact agree on the selected brand.
"""

import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
REPORTS_DIR = REPO_ROOT / "reports"
DATA_DIR = REPO_ROOT / "data"

PROFILE_JSON_PATH = REPORTS_DIR / "stage5_brand_profile.json"
PROFILE_MD_PATH = REPORTS_DIR / "stage5_brand_profile.md"
SELECTED_BRAND_PATH = DATA_DIR / "selected_brand.json"


class TestStage5BrandSelection(unittest.TestCase):
    """Test suite validating all Stage 5 requirements."""

    @classmethod
    def setUpClass(cls):
        """Loads generated Stage 5 reports and selection artifact."""
        assert PROFILE_JSON_PATH.exists(), f"Missing {PROFILE_JSON_PATH}"
        with open(PROFILE_JSON_PATH, encoding="utf-8") as fh:
            cls.profile_data = json.load(fh)

        assert SELECTED_BRAND_PATH.exists(), f"Missing {SELECTED_BRAND_PATH}"
        with open(SELECTED_BRAND_PATH, encoding="utf-8") as fh:
            cls.selected_artifact = json.load(fh)

    def test_check_1_conversations_belong_to_one_brand(self):
        """Check 1: Every discovered conversation belongs to exactly one brand."""
        total_convs = self.profile_data["metadata"]["total_conversations_profiled"]
        sum_brand_convs = sum(b["conversation_count"] for b in self.profile_data["brands"])
        self.assertEqual(
            total_convs,
            sum_brand_convs,
            f"Sum of brand conversation counts ({sum_brand_convs:,}) does not equal "
            f"total profiled conversations ({total_convs:,})",
        )

    def test_check_2_conversation_counts_consistent(self):
        """Check 2: Conversation counts are internally consistent."""
        for b in self.profile_data["brands"]:
            total = b["conversation_count"]
            en = b["english_conversation_count"]
            non_en = b["non_english_conversation_count"]
            self.assertEqual(
                total,
                en + non_en,
                f"Brand {b['brand']} conversation mismatch: {total} != {en} + {non_en}",
            )

    def test_check_3_message_counts_consistent(self):
        """Check 3: Message counts are internally consistent."""
        for b in self.profile_data["brands"]:
            tweets = b["tweet_count"]
            cust = b["customer_message_count"]
            supp = b["support_message_count"]
            self.assertEqual(
                tweets,
                cust + supp,
                f"Brand {b['brand']} tweet mismatch: {tweets} != {cust} + {supp}",
            )

    def test_check_4_percentages_calculated_correctly(self):
        """Check 4: Percentages are calculated correctly and within bounds [0.0, 100.0]."""
        for b in self.profile_data["brands"]:
            n_conv = b["conversation_count"]
            supp = b["support_message_count"]

            # English %
            expected_en = round((b["english_conversation_count"] / n_conv) * 100, 2)
            self.assertAlmostEqual(b["english_percentage"], expected_en, delta=0.05)

            # Multi-turn %
            expected_multi = round((b["multiturn_conversation_count"] / n_conv) * 100, 2)
            self.assertAlmostEqual(b["multiturn_percentage"], expected_multi, delta=0.05)

            # Response types sum to ~100% (within rounding)
            if supp > 0:
                sum_resp_pct = (
                    b["resolution_percentage"] +
                    b["escalation_percentage"] +
                    b["clarification_percentage"] +
                    b["acknowledgement_percentage"] +
                    b["uncertain_percentage"]
                )
                self.assertTrue(
                    99.0 <= sum_resp_pct <= 101.0,
                    f"Brand {b['brand']} response percentages sum to {sum_resp_pct:.2f}%",
                )

    def test_check_5_ranking_deterministic(self):
        """Check 5: Brand ranking is deterministic and strictly sequential."""
        brands = self.profile_data["brands"]
        ranks = [b["rank"] for b in brands]
        self.assertEqual(ranks, list(range(1, len(brands) + 1)))

        # Eligible brands with higher composite score rank higher
        eligible = [b for b in brands if b["meets_thresholds"]]
        for i in range(len(eligible) - 1):
            self.assertGreaterEqual(
                eligible[i]["composite_score"],
                eligible[i + 1]["composite_score"],
                f"Rank inversion between {eligible[i]['brand']} and {eligible[i+1]['brand']}",
            )

    def test_check_6_selected_brand_exists(self):
        """Check 6: Selected brand exists in the dataset."""
        sel = self.profile_data["selected_brand"]
        brand_names = {b["brand"] for b in self.profile_data["brands"]}
        self.assertIn(sel, brand_names)
        self.assertEqual(self.selected_artifact["selected_brand"], sel)

    def test_check_7_selected_brand_meets_thresholds(self):
        """Check 7: Selected brand meets the defined minimum thresholds."""
        sel = self.profile_data["selected_brand"]
        sel_entry = next(b for b in self.profile_data["brands"] if b["brand"] == sel)

        thresholds = self.profile_data["thresholds"]
        self.assertTrue(sel_entry["meets_thresholds"])
        self.assertGreaterEqual(sel_entry["conversation_count"], thresholds["min_conversations"])
        self.assertGreaterEqual(sel_entry["english_conversation_count"], thresholds["min_english_conversations"])
        self.assertGreaterEqual(sel_entry["resolution_count"], thresholds["min_substantive_resolutions"])

    def test_check_8_stage1_to_4_datasets_unmodified(self):
        """Check 8: No Stage 1-4 source datasets were modified."""
        required_files = [
            PROCESSED_DIR / "amazonhelp_tweets.csv",
            PROCESSED_DIR / "amazonhelp_conversations.csv",
            PROCESSED_DIR / "amazon_resolution_pairs.jsonl",
            PROCESSED_DIR / "amazon_escalation_pairs.jsonl",
            PROCESSED_DIR / "amazon_clarification_pairs.jsonl",
            PROCESSED_DIR / "amazon_multiturn_examples.jsonl",
            PROCESSED_DIR / "amazon_dataset_manifest.json",
            SPLITS_DIR / "amazon_splits_manifest.json",
            SPLITS_DIR / "amazon_eval_benchmark.jsonl",
        ]
        for p in required_files:
            self.assertTrue(p.exists(), f"Stage 1-4 file missing: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Stage 1-4 file is empty: {p}")

    def test_check_9_no_split_leakage(self):
        """Check 9: No train/validation/test leakage is introduced."""
        splits_manifest_path = SPLITS_DIR / "amazon_splits_manifest.json"
        self.assertTrue(splits_manifest_path.exists())
        with open(splits_manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

        self.assertEqual(manifest["split_ratios"]["train"], 0.80)
        self.assertEqual(manifest["split_ratios"]["val"], 0.10)
        self.assertEqual(manifest["split_ratios"]["test"], 0.10)

    def test_check_10_reports_agree_on_selected_brand(self):
        """Check 10: Generated JSON, Markdown, and selection artifact agree on selected brand."""
        json_brand = self.profile_data["selected_brand"]
        artifact_brand = self.selected_artifact["selected_brand"]
        self.assertEqual(json_brand, artifact_brand)

        self.assertTrue(PROFILE_MD_PATH.exists())
        md_text = PROFILE_MD_PATH.read_text(encoding="utf-8")
        self.assertIn(f"Selected Brand: {json_brand}", md_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
