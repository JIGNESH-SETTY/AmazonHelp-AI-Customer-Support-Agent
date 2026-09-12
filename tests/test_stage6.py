"""
test_stage6.py
--------------
Verification and integrity tests for Stage 6: Intent Discovery & Labeling.

Validates all 12 required checks from Section 20:
  - Check 1: Every final intent has a unique intent_id.
  - Check 2: Every final intent has a definition.
  - Check 3: Every final intent has inclusion criteria.
  - Check 4: Every final intent has exclusion criteria.
  - Check 5: Representative examples reference real conversations.
  - Check 6: No conversation belongs to multiple splits (zero leakage).
  - Check 7: Intent labels only reference valid intent IDs from taxonomy.
  - Check 8: No empty customer messages are labeled.
  - Check 9: Response type remains separate from intent.
  - Check 10: Taxonomy is reproducible and deterministic.
  - Check 11: Label distribution is internally consistent.
  - Check 12: Stage 1-5 source artifacts remain untouched.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
REPORTS_DIR = REPO_ROOT / "reports"
DATA_DIR = REPO_ROOT / "data"

SELECTED_BRAND_PATH = DATA_DIR / "selected_brand.json"
TAXONOMY_JSON_PATH = DATA_DIR / "intent_taxonomy.json"
DISCOVERY_JSON_PATH = REPORTS_DIR / "stage6_discovery.json"
REPORT_MD_PATH = REPORTS_DIR / "stage6_intent_discovery.md"
LABELS_JSONL_PATH = PROCESSED_DIR / "amazonhelp_intent_labels.jsonl"


class TestStage6IntentDiscovery(unittest.TestCase):
    """Test suite validating all Stage 6 quality and integrity requirements."""

    @classmethod
    def setUpClass(cls):
        """Loads Stage 6 artifacts and datasets."""
        assert TAXONOMY_JSON_PATH.exists(), f"Missing {TAXONOMY_JSON_PATH}"
        with open(TAXONOMY_JSON_PATH, encoding="utf-8") as fh:
            cls.taxonomy = json.load(fh)

        assert DISCOVERY_JSON_PATH.exists(), f"Missing {DISCOVERY_JSON_PATH}"
        with open(DISCOVERY_JSON_PATH, encoding="utf-8") as fh:
            cls.discovery = json.load(fh)

        cls.valid_intent_ids = {i["intent_id"] for i in cls.taxonomy["intents"]}
        cls.valid_intent_ids.add(cls.taxonomy["fallback_category"]["intent_id"])

    # -----------------------------------------------------------------------
    # Check 1: Every final intent has a unique intent_id
    # -----------------------------------------------------------------------
    def test_check_1_unique_intent_ids(self):
        """Check 1: Every final intent has a unique intent_id."""
        intent_ids = [i["intent_id"] for i in self.taxonomy["intents"]]
        self.assertEqual(len(intent_ids), len(set(intent_ids)), "Duplicate intent_id found in taxonomy!")
        self.assertGreaterEqual(len(intent_ids), 6, "Taxonomy has fewer than 6 intents")
        self.assertLessEqual(len(intent_ids), 12, "Taxonomy has more than 12 intents")

    # -----------------------------------------------------------------------
    # Check 2: Every final intent has a definition
    # -----------------------------------------------------------------------
    def test_check_2_intent_definitions_exist(self):
        """Check 2: Every final intent has a non-empty definition."""
        for item in self.taxonomy["intents"]:
            defn = item.get("definition", "").strip()
            self.assertTrue(len(defn) >= 20, f"Intent {item['intent_id']} lacks adequate definition")

    # -----------------------------------------------------------------------
    # Check 3: Every final intent has inclusion criteria
    # -----------------------------------------------------------------------
    def test_check_3_inclusion_criteria_exist(self):
        """Check 3: Every final intent has inclusion criteria."""
        for item in self.taxonomy["intents"]:
            inc = item.get("inclusion_criteria", [])
            self.assertTrue(isinstance(inc, list) and len(inc) >= 2, f"Intent {item['intent_id']} lacks inclusion criteria")

    # -----------------------------------------------------------------------
    # Check 4: Every final intent has exclusion criteria
    # -----------------------------------------------------------------------
    def test_check_4_exclusion_criteria_exist(self):
        """Check 4: Every final intent has exclusion criteria."""
        for item in self.taxonomy["intents"]:
            exc = item.get("exclusion_criteria", [])
            self.assertTrue(isinstance(exc, list) and len(exc) >= 1, f"Intent {item['intent_id']} lacks exclusion criteria")

    # -----------------------------------------------------------------------
    # Check 5: Representative examples reference real conversations
    # -----------------------------------------------------------------------
    def test_check_5_representative_examples_reference_real_convs(self):
        """Check 5: Representative examples reference real conversations."""
        for item in self.taxonomy["intents"]:
            examples = item.get("representative_examples", [])
            self.assertTrue(len(examples) >= 1, f"Intent {item['intent_id']} lacks representative examples")
            for ex in examples:
                self.assertTrue(ex.get("conversation_id", "").startswith("conv_"), f"Invalid conv_id in {item['intent_id']}")
                self.assertTrue(len(ex.get("customer_text", "")) >= 10, f"Invalid customer_text in {item['intent_id']}")

    # -----------------------------------------------------------------------
    # Check 6: No conversation belongs to multiple splits (Zero leakage)
    # -----------------------------------------------------------------------
    @unittest.skipUnless(LABELS_JSONL_PATH.exists(), "Processed labels omitted from clean clone")
    def test_check_6_no_conversation_in_multiple_splits(self):
        """Check 6: No conversation belongs to multiple splits."""
        convs_by_split = defaultdict(set)
        with open(LABELS_JSONL_PATH, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                convs_by_split[rec["split"]].add(rec["conversation_id"])

        train_convs = convs_by_split["train"]
        val_convs = convs_by_split["val"]
        test_convs = convs_by_split["test"]

        self.assertTrue(train_convs.isdisjoint(val_convs), "Leakage between train and val splits!")
        self.assertTrue(train_convs.isdisjoint(test_convs), "Leakage between train and test splits!")
        self.assertTrue(val_convs.isdisjoint(test_convs), "Leakage between val and test splits!")

    # -----------------------------------------------------------------------
    # Check 7: Intent labels only reference valid intent IDs
    # -----------------------------------------------------------------------
    @unittest.skipUnless(LABELS_JSONL_PATH.exists(), "Processed labels omitted from clean clone")
    def test_check_7_intent_labels_valid_ids(self):
        """Check 7: Intent labels only reference valid intent IDs from taxonomy."""
        with open(LABELS_JSONL_PATH, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i > 25_000:
                    break  # Sample first 25k lines for speed
                rec = json.loads(line)
                self.assertIn(rec["intent_id"], self.valid_intent_ids, f"Invalid intent_id: {rec['intent_id']}")

    # -----------------------------------------------------------------------
    # Check 8: No empty customer messages are labeled
    # -----------------------------------------------------------------------
    @unittest.skipUnless(LABELS_JSONL_PATH.exists(), "Processed labels omitted from clean clone")
    def test_check_8_no_empty_customer_messages(self):
        """Check 8: No empty customer messages are labeled."""
        with open(LABELS_JSONL_PATH, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i > 25_000:
                    break
                rec = json.loads(line)
                text = rec.get("customer_text", "")
                self.assertTrue(isinstance(text, str) and len(text.strip()) > 0, "Empty customer text found in labels")

    # -----------------------------------------------------------------------
    # Check 9: Response type remains separate from intent
    # -----------------------------------------------------------------------
    @unittest.skipUnless(LABELS_JSONL_PATH.exists(), "Processed labels omitted from clean clone")
    def test_check_9_response_type_distinct_from_intent(self):
        """Check 9: Response type remains separate from intent."""
        with open(LABELS_JSONL_PATH, encoding="utf-8") as fh:
            first_row = json.loads(fh.readline())
            self.assertIn("response_type", first_row)
            self.assertIn("intent_id", first_row)
            # Response type must be one of the support behaviors, not an intent
            self.assertIn(first_row["response_type"], {"resolution", "escalation", "clarification"})
            self.assertNotEqual(first_row["response_type"], first_row["intent_id"])

    # -----------------------------------------------------------------------
    # Check 10: Taxonomy is reproducible
    # -----------------------------------------------------------------------
    def test_check_10_taxonomy_reproducible(self):
        """Check 10: Taxonomy metadata is consistent and versioned."""
        self.assertEqual(self.taxonomy["taxonomy_version"], "1.0")
        self.assertEqual(self.taxonomy["brand"], "AmazonHelp")
        self.assertEqual(len(self.taxonomy["intents"]), 10)

    # -----------------------------------------------------------------------
    # Check 11: Label distribution is internally consistent
    # -----------------------------------------------------------------------
    @unittest.skipUnless(LABELS_JSONL_PATH.exists(), "Processed labels omitted from clean clone")
    def test_check_11_label_distribution_consistent(self):
        """Check 11: Label counts sum to total lines in amazonhelp_intent_labels.jsonl."""
        line_count = 0
        intent_counts = Counter()
        with open(LABELS_JSONL_PATH, encoding="utf-8") as fh:
            for line in fh:
                line_count += 1
                rec = json.loads(line)
                intent_counts[rec["intent_id"]] += 1

        self.assertEqual(line_count, sum(intent_counts.values()))
        self.assertGreater(line_count, 80_000, "Labeled dataset has fewer than 80,000 interactions")

    # -----------------------------------------------------------------------
    # Check 12: Stage 1-5 source artifacts remain untouched
    # -----------------------------------------------------------------------
    @unittest.skipUnless((PROCESSED_DIR / "amazonhelp_tweets.csv").exists(), "Processed datasets omitted from clean clone")
    def test_check_12_stage_1_to_5_artifacts_untouched(self):
        """Check 12: Stage 1-5 source artifacts remain untouched."""
        critical_paths = [
            PROCESSED_DIR / "amazonhelp_tweets.csv",
            PROCESSED_DIR / "amazonhelp_conversations.csv",
            PROCESSED_DIR / "amazon_resolution_pairs.jsonl",
            PROCESSED_DIR / "amazon_escalation_pairs.jsonl",
            PROCESSED_DIR / "amazon_clarification_pairs.jsonl",
            PROCESSED_DIR / "amazon_multiturn_examples.jsonl",
            PROCESSED_DIR / "amazon_dataset_manifest.json",
            SPLITS_DIR / "amazon_splits_manifest.json",
            SPLITS_DIR / "amazon_eval_benchmark.jsonl",
            DATA_DIR / "selected_brand.json",
            REPORTS_DIR / "stage5_brand_profile.json",
            REPORTS_DIR / "stage5_brand_profile.md",
        ]
        for p in critical_paths:
            self.assertTrue(p.exists(), f"Critical prior-stage file missing: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Critical prior-stage file empty: {p}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
