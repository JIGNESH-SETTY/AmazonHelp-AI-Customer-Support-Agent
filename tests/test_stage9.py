"""
test_stage9.py
--------------
Comprehensive verification and unit test suite for Stage 9: AI Support Agent.

Validates all 18 requirements from Section 17 & Objectives:
  1. Empty input handling
  2. Input normalization (whitespace, elongation reduction)
  3. Hybrid intent prediction & top-k ranking
  4. Unknown / low-confidence input handling
  5. Multi-intent detection (primary vs secondary)
  6. Context & entity extraction (order ID, carrier, sentiment)
  7. BM25 historical context retrieval
  8. Policy validation & anti-hallucination checks
  9. False-action claim remediation
  10. Deterministic response generation & fallback
  11. Structured output schema (zero chain-of-thought exposure)
  12. Escalation signaling & priority triggers
  13. Golden Set evaluation execution & integrity
  14. Zero Golden Set training leakage
  15. Complete offline execution without external API keys
  16. Ablation configurations execution
  17. Security & credential safety (no hardcoded API keys)
  18. Latency performance
"""

import json
from pathlib import Path
import time
import unittest

from src.agent.agent import SupportAgent
from src.agent.config import AgentConfig
from src.agent.context import extract_context
from src.agent.generation import (
    DeterministicResponseGenerator,
    LLMResponseGenerator,
)
from src.agent.intent import HybridIntentClassifier
from src.agent.normalize import normalize_input
from src.agent.policy import PolicyGuardrail
from src.agent.retrieval import KnowledgeRetriever, sanitize_historical_response
from src.agent.schemas import AgentInput, ContextEntities, IntentResult
from src.retrieval.bm25 import BM25Index

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
GOLDEN_JSONL_PATH = DATA_DIR / "golden" / "golden_evaluation_set.jsonl"
RESULTS_JSON_PATH = REPO_ROOT / "reports" / "stage9_agent_results.json"
REPORT_MD_PATH = REPO_ROOT / "reports" / "stage9_ai_agent.md"


class TestStage9AISupportAgent(unittest.TestCase):
    """Test suite validating all Stage 9 AI Support Agent requirements."""

    @classmethod
    def setUpClass(cls):
        """Initializes a lightweight agent for testing."""
        cls.config = AgentConfig()
        cls.agent = SupportAgent(config=cls.config)

        # Seed mock training data for agent initialization
        docs = [
            "where is my delayed package that has not arrived",
            "I want a refund for the returned book",
            "cancel my order immediately please",
            "package says delivered but nothing in mailbox",
            "my Kindle screen is cracked and broken",
            "charged twice for prime membership renewal fee",
            "cannot log in and need to reset password",
            "supervisor Jennifer was rude and hung up on me",
        ]
        labels = [
            "delivery_delay",
            "returns_and_refunds",
            "order_cancellation",
            "missing_delivered_package",
            "damaged_defective_item",
            "prime_membership",
            "account_access_security",
            "service_complaint_escalation",
        ]
        meta = [{"customer_message": d, "support_response": f"Support for: {d}", "example_id": f"ex_{i}"} for i, d in enumerate(docs)]
        cls.agent.fit_training_data(docs, labels, meta)

    def test_01_empty_and_null_input(self):
        """Verify empty, whitespace-only, and null inputs are gracefully handled."""
        out_empty = self.agent.process("")
        self.assertEqual(out_empty.primary_intent, "unknown_or_ambiguous")
        self.assertTrue(len(out_empty.response) > 0)

        out_spaces = self.agent.process("   \n\t   ")
        self.assertEqual(out_spaces.primary_intent, "unknown_or_ambiguous")

        out_null = self.agent.process(AgentInput(customer_text=None))  # type: ignore
        self.assertEqual(out_null.primary_intent, "unknown_or_ambiguous")

    def test_02_normalization(self):
        """Verify normalization collapses excessive whitespace and repeated characters."""
        res = normalize_input(AgentInput(customer_text="   whyyyy   is   my   packageee   late???   "))
        self.assertTrue(res.is_valid)
        self.assertEqual(res.normalized_text, "whyy is my packagee late??")
        self.assertIn("whyyyy", res.original_text)

    def test_03_intent_prediction_and_top_k(self):
        """Verify hybrid intent classifier outputs valid intent, confidence, and top-k."""
        res = self.agent.intent_classifier.predict("My order has not arrived and is delayed")
        self.assertEqual(res.primary_intent, "delivery_delay")
        self.assertGreater(res.intent_confidence, 0.50)
        self.assertGreater(len(res.top_candidates), 1)

    def test_04_unknown_low_confidence_input(self):
        """Verify vague or uninformative text routes to unknown_or_ambiguous with low confidence."""
        res = self.agent.intent_classifier.predict("hmm ok thanks")
        self.assertEqual(res.primary_intent, "unknown_or_ambiguous")
        self.assertLessEqual(res.intent_confidence, 0.60)

    def test_05_multi_intent_detection(self):
        """Verify agent detects primary and secondary intents in compound customer queries."""
        query = "My package is delayed and I want to cancel it and get a refund!"
        res = self.agent.intent_classifier.predict(query)
        self.assertIn("order_cancellation", [res.primary_intent] + res.secondary_intents)
        self.assertIn("returns_and_refunds", [res.primary_intent] + res.secondary_intents)

    def test_06_context_and_entity_extraction(self):
        """Verify extraction of order numbers, products, carriers, and sentiment."""
        query = "Order 112-3456789-1234567 for my Kindle via USPS was delivered broken! Worst service ever!"
        ctx = extract_context(query)
        self.assertEqual(ctx.order_id, "112-3456789-1234567")
        self.assertEqual(ctx.product_reference, "Kindle E-reader")
        self.assertEqual(ctx.carrier_reference, "USPS")
        self.assertEqual(ctx.sentiment, "angry")

    def test_07_retrieval_sanitization(self):
        """Verify historical tweets are cleansed of @mentions, agent initials, and dead links."""
        raw_tweet = "@123456 Sorry to hear this! Please visit https://t.co/abc to check tracking. ^JZ 1/2"
        clean = sanitize_historical_response(raw_tweet)
        self.assertNotIn("@123456", clean)
        self.assertNotIn("^JZ", clean)
        self.assertNotIn("1/2", clean)
        self.assertIn("<URL>", clean)

    def test_08_policy_hallucination_prevention(self):
        """Verify policy validator flags unsupported action claims and delivery guarantees."""
        guard = PolicyGuardrail()
        bad_response = "I have processed your refund and guaranteed to arrive tomorrow by 2pm!"
        check = guard.validate(bad_response, "delivery_delay", ContextEntities())
        self.assertFalse(check.passed)
        self.assertGreater(len(check.violations), 0)

    def test_09_policy_remediation(self):
        """Verify policy engine can remediate false execution claims into safe guidance."""
        guard = PolicyGuardrail()
        claim = "I have processed your refund for this item. ^AB"
        remediated, was_changed = guard.remediate(claim, "returns_and_refunds", ContextEntities())
        self.assertTrue(was_changed)
        self.assertNotIn("I have processed your refund", remediated)
        self.assertNotIn("^AB", remediated)

    def test_10_deterministic_response_generation(self):
        """Verify deterministic generator produces context-aware, entity-grounded responses."""
        gen = DeterministicResponseGenerator()
        intent_res = IntentResult("delivery_delay", "Delivery Delay & Tracking", 0.90)
        ctx = ContextEntities(product_reference="Fire TV Device", carrier_reference="UPS", has_missing_info=True, missing_info_details=["order_id"])

        resp = gen.generate("where is my Fire TV", intent_res, ctx, [])
        self.assertIn("Fire TV Device", resp)
        self.assertIn("UPS", resp)
        self.assertIn("order ID", resp)

    def test_11_structured_output_schema_no_cot(self):
        """Verify output schema is complete and does not expose chain-of-thought."""
        output = self.agent.process("Where is my package?")
        d = output.to_dict()

        for field in ["conversation_id", "primary_intent", "intent_confidence", "secondary_intents", "context", "retrieval", "policy", "escalation", "response"]:
            self.assertIn(field, d)

        # Confirm no chain-of-thought or reasoning keys exist
        self.assertNotIn("chain_of_thought", d)
        self.assertNotIn("thought", d)
        self.assertNotIn("reasoning_steps", d)

    def test_12_escalation_triggers(self):
        """Verify human escalation triggers on customer service complaints or critical anger."""
        out_complaint = self.agent.process("I want to speak with a supervisor! Your agent was rude and hung up on me!")
        self.assertTrue(out_complaint.escalation["required"])
        self.assertIn(out_complaint.escalation["priority"], ["high", "urgent"])

    def test_13_golden_set_evaluation_results_exist(self):
        """Verify Stage 9 benchmark results artifact and markdown report exist."""
        self.assertTrue(RESULTS_JSON_PATH.exists(), f"Missing {RESULTS_JSON_PATH}")
        self.assertTrue(REPORT_MD_PATH.exists(), f"Missing {REPORT_MD_PATH}")

        with open(RESULTS_JSON_PATH, encoding="utf-8") as fh:
            data = json.load(fh)

        self.assertEqual(data["stage"], 9)
        self.assertIn("complete_agent_metrics", data)
        self.assertIn("ablation_study", data)
        self.assertIn("baseline_comparison", data)

        # Verify Stage 9 beat Stage 8 baseline
        stage9_acc = data["complete_agent_metrics"]["intent_classification"]["accuracy"]
        stage8_acc = data["baseline_comparison"]["stage8_accuracy"]
        self.assertGreater(stage9_acc, stage8_acc, f"Stage 9 ({stage9_acc}) did not beat Stage 8 ({stage8_acc})")

    def test_14_zero_golden_leakage(self):
        """Verify no Golden Set conversation IDs are in agent training corpus."""
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

        overlap = 0
        for tf in train_files:
            if not tf.exists():
                continue
            with open(tf, encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    if row.get("conversation_id") in golden_conv_ids:
                        overlap += 1

        self.assertEqual(overlap, 0, f"Found {overlap} overlapping conversation IDs in training split!")

    def test_15_offline_execution_without_api_key(self):
        """Verify LLM generator safely executes offline when no API key is provided."""
        cfg = AgentConfig(generator_type="llm", llm_api_key=None)
        llm_gen = LLMResponseGenerator(config=cfg)
        intent_res = IntentResult("delivery_delay", "Delivery Delay & Tracking", 0.90)
        ctx = ContextEntities()
        resp = llm_gen.generate("late package", intent_res, ctx, [])
        self.assertTrue(len(resp) > 20)

    def test_16_ablation_configurations(self):
        """Verify agent executes in ablation mode with retrieval or policy disabled."""
        cfg_no_retrieval = AgentConfig(enable_retrieval=False, enable_policy_guardrails=False)
        agent_no_ret = SupportAgent(config=cfg_no_retrieval)
        out = agent_no_ret.process("my package is late")
        self.assertEqual(len(out.retrieval), 0)
        self.assertTrue(out.policy["passed"])

    def test_17_security_no_hardcoded_secrets(self):
        """Verify configuration does not contain hardcoded credentials."""
        cfg = AgentConfig()
        self.assertIsNone(cfg.llm_api_key)

    def test_18_latency_bound(self):
        """Verify single query latency is within real-time interactive SLA (< 50ms)."""
        t0 = time.time()
        self.agent.process("Tracking number shows delivered but I was home and no parcel")
        latency_ms = (time.time() - t0) * 1000.0
        self.assertLess(latency_ms, 50.0, f"Latency exceeded SLA: {latency_ms:.2f}ms")


if __name__ == "__main__":
    unittest.main()
