"""
test_stage8.py
--------------
Verification and unit tests for Stage 8: Baselines.

Validates:
  1. BM25Index functionality (indexing, top-k scoring, term matching)
  2. MajorityClassBaseline behavior and determinism
  3. KeywordRulesBaseline accuracy on canonical intent phrases
  4. TFIDFNaiveBayesBaseline fitting and valid intent prediction
  5. BM25NearestNeighborBaseline 1-NN prediction
  6. Response generators produce valid non-empty responses for all intents
  7. Classification and NLP metric functions (Accuracy, F1, ROUGE, BLEU)
  8. Baseline results artifacts exist and contain valid scores
  9. Zero leakage of Golden Set / test split conversation IDs into baseline training data
"""

import json
from pathlib import Path
import unittest

from src.evaluation.baselines import (
    BM25HistoricRetrievalResponse,
    BM25NearestNeighborBaseline,
    GenericDefaultResponse,
    IntentCannedTemplateResponse,
    KeywordRulesBaseline,
    MajorityClassBaseline,
    TFIDFNaiveBayesBaseline,
)
from src.evaluation.evaluate_baselines import (
    compute_bleu,
    compute_classification_metrics,
    compute_guidance_adherence,
    compute_rouge_l,
    compute_rouge_n,
)
from src.retrieval.bm25 import BM25Index, tokenize

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
GOLDEN_JSONL_PATH = DATA_DIR / "golden" / "golden_evaluation_set.jsonl"
TAXONOMY_PATH = DATA_DIR / "intent_taxonomy.json"
RESULTS_JSON_PATH = REPORTS_DIR / "stage8_baseline_results.json"
REPORT_MD_PATH = REPORTS_DIR / "stage8_baselines.md"


class TestStage8Baselines(unittest.TestCase):
    """Test suite for Stage 8 baselines and evaluation components."""

    @classmethod
    def setUpClass(cls):
        with open(TAXONOMY_PATH, encoding="utf-8") as fh:
            cls.taxonomy = json.load(fh)
        cls.valid_intent_ids = {i["intent_id"] for i in cls.taxonomy["intents"]}
        cls.valid_intent_ids.add("unknown_or_ambiguous")

    def test_01_bm25_index_and_retrieval(self):
        """Verify BM25 index correctly indexes documents and retrieves relevant match."""
        docs = [
            "My package has not arrived and is delayed in transit",
            "I need a refund for a returned item",
            "How do I cancel my Prime membership renewal",
        ]
        meta = [
            {"intent_id": "delivery_delay"},
            {"intent_id": "returns_and_refunds"},
            {"intent_id": "prime_membership"},
        ]
        bm25 = BM25Index()
        bm25.fit(docs, meta)

        self.assertEqual(bm25.corpus_size, 3)
        hits = bm25.search("where is my delayed package", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["intent_id"], "delivery_delay")
        self.assertGreater(hits[0]["bm25_score"], 0.0)

    def test_02_majority_class_baseline(self):
        """Verify MajorityClassBaseline predicts dominant class."""
        labels = ["delivery_delay", "delivery_delay", "returns_and_refunds"]
        clf = MajorityClassBaseline()
        clf.fit(labels)
        self.assertEqual(clf.predict("Any arbitrary query"), "delivery_delay")

    def test_03_keyword_rules_baseline(self):
        """Verify KeywordRulesBaseline classifies unambiguous domain keywords."""
        clf = KeywordRulesBaseline()
        self.assertEqual(clf.predict("I want to cancel my order immediately"), "order_cancellation")
        self.assertEqual(clf.predict("The item arrived with a shattered screen and broken box"), "damaged_defective_item")
        self.assertEqual(clf.predict("Tracking says delivered but nothing in mailbox"), "missing_delivered_package")
        self.assertEqual(clf.predict("Cannot login and need to reset password"), "account_access_security")

    def test_04_tfidf_naive_bayes_baseline(self):
        """Verify TFIDFNaiveBayesBaseline trains and predicts valid classes."""
        texts = [
            "where is my late shipment",
            "delayed delivery package late",
            "want a refund for my return",
            "send money back for returned order",
        ]
        labels = [
            "delivery_delay",
            "delivery_delay",
            "returns_and_refunds",
            "returns_and_refunds",
        ]
        clf = TFIDFNaiveBayesBaseline(min_df=1)
        clf.fit(texts, labels)

        pred = clf.predict("my shipment is very late")
        self.assertEqual(pred, "delivery_delay")

        pred_refund = clf.predict("please give refund on return")
        self.assertEqual(pred_refund, "returns_and_refunds")

    def test_05_bm25_1nn_baseline(self):
        """Verify BM25NearestNeighborBaseline assigns nearest neighbor intent."""
        docs = ["package is missing", "cancel subscription fee"]
        meta = [{"intent_id": "missing_delivered_package"}, {"intent_id": "prime_membership"}]
        bm25 = BM25Index()
        bm25.fit(docs, meta)

        clf = BM25NearestNeighborBaseline(bm25)
        self.assertEqual(clf.predict("missing package from porch"), "missing_delivered_package")

    def test_06_response_generators(self):
        """Verify all response generators produce non-empty strings."""
        gen_default = GenericDefaultResponse()
        self.assertTrue(len(gen_default.generate("query")) > 20)

        gen_canned = IntentCannedTemplateResponse()
        for intent_id in self.valid_intent_ids:
            resp = gen_canned.generate("query", intent_id)
            self.assertTrue(len(resp) > 20, f"Empty template for intent {intent_id}")

        bm25 = BM25Index()
        bm25.fit(["my order is late"], [{"support_response": "We are looking into this late order"}])
        gen_retrieval = BM25HistoricRetrievalResponse(bm25)
        self.assertIn("late order", gen_retrieval.generate("my order is late"))

    def test_07_classification_metric_computation(self):
        """Verify compute_classification_metrics produces bounded correct numbers."""
        y_true = ["delivery_delay", "returns_and_refunds", "delivery_delay"]
        y_pred = ["delivery_delay", "delivery_delay", "delivery_delay"]
        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["accuracy"], 2 / 3, places=3)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)

    def test_08_rouge_and_bleu_computation(self):
        """Verify ROUGE and BLEU metric computations."""
        ref = tokenize("we are sorry for the shipping delay", remove_stopwords=False)
        hyp = tokenize("we are sorry for the shipping delay", remove_stopwords=False)

        rouge_1 = compute_rouge_n(hyp, ref, n=1)
        rouge_l = compute_rouge_l(hyp, ref)
        bleu = compute_bleu(hyp, ref)

        self.assertAlmostEqual(rouge_1["f1"], 1.0, places=3)
        self.assertAlmostEqual(rouge_l["f1"], 1.0, places=3)
        self.assertAlmostEqual(bleu["bleu_1"], 1.0, places=3)

    def test_09_guidance_adherence(self):
        """Verify guidance adherence scoring logic."""
        good_resp = "Please check your latest tracking status in Your Orders."
        bad_resp = "I have canceled your order and changed your password."

        self.assertEqual(compute_guidance_adherence(good_resp, "delivery_delay"), 1.0)
        self.assertEqual(compute_guidance_adherence(bad_resp, "delivery_delay"), 0.0)

    def test_10_baseline_results_artifact_integrity(self):
        """Verify Stage 8 results artifacts exist and contain required keys."""
        self.assertTrue(RESULTS_JSON_PATH.exists(), f"Missing {RESULTS_JSON_PATH}")
        self.assertTrue(REPORT_MD_PATH.exists(), f"Missing {REPORT_MD_PATH}")

        with open(RESULTS_JSON_PATH, encoding="utf-8") as fh:
            data = json.load(fh)

        self.assertEqual(data["stage"], 8)
        self.assertIn("intent_classification_baselines", data)
        self.assertIn("response_generation_baselines", data)

        for model_name in ["majority_class", "keyword_rules", "tfidf_naive_bayes", "bm25_1nn"]:
            self.assertIn(model_name, data["intent_classification_baselines"])

        for model_name in ["generic_default", "intent_canned_template", "bm25_retrieval"]:
            self.assertIn(model_name, data["response_generation_baselines"])

    def test_11_zero_golden_leakage_in_training_corpus(self):
        """Verify no Golden Evaluation Set conversation IDs exist in training corpus."""
        golden_conv_ids = set()
        with open(GOLDEN_JSONL_PATH, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    golden_conv_ids.add(json.loads(line)["conversation_id"])

        train_files = [
            DATA_DIR / "processed" / "splits" / "train" / "amazon_resolution_pairs_train.jsonl",
            DATA_DIR / "processed" / "splits" / "train" / "amazon_clarification_pairs_train.jsonl",
            DATA_DIR / "processed" / "splits" / "train" / "amazon_escalation_pairs_train.jsonl",
        ]

        overlap_count = 0
        for tf in train_files:
            if not tf.exists():
                continue
            with open(tf, encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    if row.get("conversation_id") in golden_conv_ids:
                        overlap_count += 1

        self.assertEqual(overlap_count, 0, f"Found {overlap_count} overlapping conversations in training split!")


if __name__ == "__main__":
    unittest.main()
