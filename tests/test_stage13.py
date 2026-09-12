"""
test_stage13.py
---------------
STAGE 13: Verification and unit tests for LLM-as-Judge & Judge-Human Agreement.

Validates:
  1. Judge JSON schema validation (JudgeScore parsing & bounds checking)
  2. Invalid judge output handling (missing fields, invalid scores, CoT rejection)
  3. Missing API key handling (offline mode does not crash)
  4. Cached-result loading (replays verified cached evaluations)
  5. Golden-set sample selection (selects exactly 50 records)
  6. Deterministic selection with seed 42 (identical selection order)
  7. Human annotation schema validation (8 required columns in CSV)
  8. Incomplete human annotation detection (properly marks incomplete status)
  9. Agreement calculation math (exact agreement, MAD, Pearson correlation)
  10. Cohen's kappa calculation (perfect, partial, chance agreement)
  11. No fabricated human scores (verifies CSV rating fields are unpopulated)
  12. Existing evaluation compatibility (integrates cleanly with GoldenSetLoader & SupportAgent)
"""

import csv
import json
import math
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from src.agent.agent import SupportAgent
from src.agent.config import AgentConfig
from src.evaluation.datasets import GoldenEvaluationRecord, GoldenSetLoader
from src.evaluation.judge_agreement import (
    REQUIRED_HUMAN_COLUMNS,
    analyze_agreement,
    compute_cohens_kappa,
    compute_exact_agreement,
    compute_mean_absolute_difference,
    compute_ordinal_correlation,
    compute_pass_fail_agreement,
    load_human_annotations,
)
from src.evaluation.llm_judge import (
    DEFAULT_RESULTS_PATH,
    DEFAULT_SAMPLE_PATH,
    HUMAN_ANNOTATIONS_CSV,
    JudgeScore,
    LLMJudgeClient,
    build_judge_prompt,
    select_golden_sample,
    validate_judge_response,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "data" / "golden" / "golden_evaluation_set.jsonl"
STAGE13_DIR = REPO_ROOT / "reports" / "stage13"


class TestStage13LLMJudgeAndAgreement(unittest.TestCase):
    """Unit test suite for Stage 13 LLM-as-Judge and Human Agreement engine."""

    @classmethod
    def setUpClass(cls):
        """Loads Golden Set records for sampling tests."""
        cls.loader = GoldenSetLoader(golden_path=GOLDEN_PATH)
        cls.golden_records = cls.loader.load()

    # -----------------------------------------------------------------------
    # 1. Judge JSON schema validation
    # -----------------------------------------------------------------------
    def test_01_judge_json_schema_validation(self):
        """Verify that valid judge dictionary converts cleanly to JudgeScore dataclass."""
        valid_input = {
            "helpfulness": 4,
            "grounding": 5,
            "actionability": 4,
            "clarity": 5,
            "overall": 4.5,
            "pass": True,
            "reason": "Direct, empathetic reply directing customer to official self-service tracking.",
            "issue_category": None,
        }
        score = validate_judge_response(valid_input)
        self.assertIsInstance(score, JudgeScore)
        self.assertEqual(score.helpfulness, 4)
        self.assertEqual(score.grounding, 5)
        self.assertEqual(score.actionability, 4)
        self.assertEqual(score.clarity, 5)
        self.assertEqual(score.overall, 4.5)
        self.assertTrue(score.pass_status)
        self.assertEqual(score.reason, valid_input["reason"])
        self.assertIsNone(score.issue_category)

        out_dict = score.to_dict()
        self.assertEqual(out_dict["pass"], True)
        self.assertEqual(out_dict["overall"], 4.5)

    # -----------------------------------------------------------------------
    # 2. Invalid judge output handling
    # -----------------------------------------------------------------------
    def test_02_invalid_judge_output_handling(self):
        """Verify that missing fields, out-of-range scores, and CoT leaks raise ValueError."""
        # Non-dict input
        with self.assertRaises(ValueError):
            validate_judge_response("not a dict")  # type: ignore

        # Missing required criterion
        with self.assertRaises(ValueError):
            validate_judge_response({
                "helpfulness": 5,
                "grounding": 5,
                "actionability": 5,
                # Missing clarity
                "overall": 5.0,
                "pass": True,
                "reason": "Missing clarity criterion",
            })

        # Score out of bounds (< 1)
        with self.assertRaises(ValueError):
            validate_judge_response({
                "helpfulness": 0,
                "grounding": 5,
                "actionability": 5,
                "clarity": 5,
                "overall": 5.0,
                "pass": True,
                "reason": "Out of bounds helpfulness",
            })

        # Score out of bounds (> 5)
        with self.assertRaises(ValueError):
            validate_judge_response({
                "helpfulness": 5,
                "grounding": 6,
                "actionability": 5,
                "clarity": 5,
                "overall": 5.0,
                "pass": True,
                "reason": "Out of bounds grounding",
            })

        # Exposing chain of thought
        with self.assertRaises(ValueError):
            validate_judge_response({
                "helpfulness": 5,
                "grounding": 5,
                "actionability": 5,
                "clarity": 5,
                "overall": 5.0,
                "pass": True,
                "reason": "Good",
                "chain_of_thought": "Thinking step by step...",
            })

    # -----------------------------------------------------------------------
    # 3. Missing API key handling
    # -----------------------------------------------------------------------
    def test_03_missing_api_key_handling(self):
        """Verify that client behaves gracefully in offline/unauthenticated mode."""
        client = LLMJudgeClient(api_key="", offline_mode=True)
        self.assertTrue(client.offline_mode)

        # Calling live evaluation in offline mode must raise clean RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            client.evaluate_single(
                customer_message="Where is my order?",
                expected_behavior="Give tracking link",
                gold_guidance="Must not fabricate dates",
                generated_response="Check tracking at <URL>",
                intent_name="Delivery Delay",
            )
        self.assertIn("offline mode", str(ctx.exception).lower())

    # -----------------------------------------------------------------------
    # 4. Cached-result loading
    # -----------------------------------------------------------------------
    def test_04_cached_result_loading(self):
        """Verify cached evaluation data can be parsed and validated."""
        dummy_cached = {
            "stage": 13,
            "provider_status": "cached_llm",
            "model_configured": "gpt-4o-mini",
            "sample_size": 2,
            "evaluations": [
                {
                    "example_id": "golden_001",
                    "conversation_id": "conv_001",
                    "primary_intent": "delivery_delay",
                    "difficulty": "hard",
                    "customer_message": "Where is my package?",
                    "generated_response": "Track your package in Your Orders.",
                    "judge_score": {
                        "helpfulness": 4,
                        "grounding": 5,
                        "actionability": 4,
                        "clarity": 5,
                        "overall": 4.5,
                        "pass": True,
                        "reason": "Direct self-service advice.",
                        "issue_category": None,
                    },
                },
                {
                    "example_id": "golden_002",
                    "conversation_id": "conv_002",
                    "primary_intent": "order_cancellation",
                    "difficulty": "medium",
                    "customer_message": "Cancel my order please.",
                    "generated_response": "Visit Your Orders to request cancellation.",
                    "judge_score": {
                        "helpfulness": 5,
                        "grounding": 5,
                        "actionability": 5,
                        "clarity": 5,
                        "overall": 5.0,
                        "pass": True,
                        "reason": "Accurate guidance without claiming cancellation processed.",
                        "issue_category": None,
                    },
                },
            ],
        }
        self.assertEqual(len(dummy_cached["evaluations"]), 2)
        score1 = validate_judge_response(dummy_cached["evaluations"][0]["judge_score"])
        self.assertEqual(score1.helpfulness, 4)
        self.assertTrue(score1.pass_status)

    # -----------------------------------------------------------------------
    # 5. Golden-set sample selection
    # -----------------------------------------------------------------------
    def test_05_golden_set_sample_selection(self):
        """Verify stratified selection returns exactly 50 examples from the 200 records."""
        sample = select_golden_sample(self.golden_records, sample_size=50, random_seed=42)
        self.assertEqual(len(sample), 50)

        # All 50 IDs must be distinct
        ex_ids = [r.example_id for r in sample]
        self.assertEqual(len(ex_ids), len(set(ex_ids)))

        # Must span all 11 intent categories
        intents_in_sample = {r.primary_intent for r in sample}
        self.assertEqual(len(intents_in_sample), 11)

        # Must span all 3 difficulty tiers
        diffs_in_sample = {r.difficulty for r in sample}
        self.assertEqual(diffs_in_sample, {"easy", "medium", "hard"})

    # -----------------------------------------------------------------------
    # 6. Deterministic selection with seed 42
    # -----------------------------------------------------------------------
    def test_06_deterministic_selection_with_seed_42(self):
        """Verify that running selection twice with seed 42 produces identical order."""
        run1 = select_golden_sample(self.golden_records, sample_size=50, random_seed=42)
        run2 = select_golden_sample(self.golden_records, sample_size=50, random_seed=42)
        ids1 = [r.example_id for r in run1]
        ids2 = [r.example_id for r in run2]
        self.assertEqual(ids1, ids2, "Sample selection was not deterministic!")

    # -----------------------------------------------------------------------
    # 7. Human annotation schema validation
    # -----------------------------------------------------------------------
    def test_07_human_annotation_schema_validation(self):
        """Verify that human_judge_annotations.csv contains all required headers and 50 rows."""
        self.assertTrue(HUMAN_ANNOTATIONS_CSV.exists(), f"Missing {HUMAN_ANNOTATIONS_CSV}")
        with open(HUMAN_ANNOTATIONS_CSV, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            rows = list(reader)

        for col in REQUIRED_HUMAN_COLUMNS:
            self.assertIn(col, headers, f"Missing required column '{col}' in human CSV")

        self.assertEqual(len(rows), 50, f"Expected 50 template rows, found {len(rows)}")

    # -----------------------------------------------------------------------
    # 8. Incomplete human annotation detection
    # -----------------------------------------------------------------------
    def test_08_incomplete_human_annotation_detection(self):
        """Verify that unpopulated rating columns trigger incomplete status."""
        is_complete, rows = load_human_annotations(HUMAN_ANNOTATIONS_CSV)
        self.assertFalse(is_complete, "Unpopulated template was falsely marked as complete!")
        self.assertEqual(len(rows), 50)

        # Check analyze_agreement behavior on incomplete CSV
        res = analyze_agreement(
            human_csv_path=HUMAN_ANNOTATIONS_CSV,
            judge_results_path=DEFAULT_RESULTS_PATH,
        )
        self.assertEqual(res["status"], "pending_human_annotation")
        self.assertIn("incomplete", res["message"].lower())
        self.assertEqual(res["metrics"], {})

    # -----------------------------------------------------------------------
    # 9. Agreement calculation math
    # -----------------------------------------------------------------------
    def test_09_agreement_calculation_math(self):
        """Verify exact agreement, MAD, and correlation on controlled numerical fixtures."""
        h = [5.0, 4.0, 3.0, 2.0, 1.0]
        j = [5.0, 4.0, 3.0, 1.0, 2.0]

        # Exact agreement: 3 matches out of 5 = 0.60
        ea = compute_exact_agreement(h, j)
        self.assertAlmostEqual(ea, 0.60, places=2)

        # MAD: (|5-5| + |4-4| + |3-3| + |2-1| + |1-2|) / 5 = 2 / 5 = 0.40
        mad = compute_mean_absolute_difference(h, j)
        self.assertAlmostEqual(mad, 0.40, places=2)

        # Ordinal correlation: strong positive correlation (> 0.80)
        corr = compute_ordinal_correlation(h, j)
        self.assertGreater(corr, 0.80)

    # -----------------------------------------------------------------------
    # 10. Cohen's kappa calculation
    # -----------------------------------------------------------------------
    def test_10_cohens_kappa_calculation(self):
        """Verify Cohen's kappa on known benchmark distributions."""
        # 1. Perfect agreement
        h_perf = [True, True, False, False, True]
        j_perf = [True, True, False, False, True]
        self.assertAlmostEqual(compute_cohens_kappa(h_perf, j_perf), 1.0, places=3)

        # 2. Partial agreement
        h_part = [True, True, True, True, False, False, False, False]
        j_part = [True, True, True, False, False, False, False, True]
        kappa_part = compute_cohens_kappa(h_part, j_part)
        self.assertGreater(kappa_part, 0.40)
        self.assertLess(kappa_part, 1.0)

        # 3. Complete disagreement
        h_dis = [True, True, True, True]
        j_dis = [False, False, False, False]
        self.assertLessEqual(compute_cohens_kappa(h_dis, j_dis), 0.0)

    # -----------------------------------------------------------------------
    # 11. No fabricated human scores
    # -----------------------------------------------------------------------
    def test_11_no_fabricated_human_scores(self):
        """
        Anti-fabrication assertion: Verifies that human ratings in the template
        are strictly empty until a real human annotator fills them in.
        """
        with open(HUMAN_ANNOTATIONS_CSV, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader, start=1):
                for rating_col in [
                    "human_helpfulness",
                    "human_grounding",
                    "human_actionability",
                    "human_clarity",
                    "human_overall",
                    "human_pass",
                ]:
                    val = row.get(rating_col, "").strip()
                    self.assertEqual(
                        val,
                        "",
                        f"Row {idx} ({row.get('example_id')}) has non-empty '{rating_col}': '{val}'. "
                        "Fabrication violation! Human template ratings must be empty.",
                    )

    # -----------------------------------------------------------------------
    # 12. Existing evaluation compatibility
    # -----------------------------------------------------------------------
    def test_12_existing_evaluation_compatibility(self):
        """Verify that Stage 13 components do not break SupportAgent or GoldenSetLoader."""
        self.assertGreaterEqual(len(self.golden_records), 200)

        # Instantiate SupportAgent
        agent = SupportAgent(config=AgentConfig())
        first_rec = self.golden_records[0]
        out = agent.process(first_rec.customer_message)
        self.assertIsNotNone(out.primary_intent)
        self.assertIsNotNone(out.response)
        self.assertGreater(len(out.response), 10)

        # Verify prompt builder output
        sys_p, usr_p = build_judge_prompt(
            customer_message=first_rec.customer_message,
            expected_behavior=first_rec.expected_behavior,
            gold_guidance=first_rec.gold_response_guidance,
            generated_response=out.response,
            intent_name=first_rec.primary_intent_name,
        )
        self.assertIn("helpfulness", sys_p)
        self.assertIn("grounding", sys_p)
        self.assertIn(first_rec.primary_intent_name, usr_p)

    # -----------------------------------------------------------------------
    # 13. Groq API Key Detection & Default Config
    # -----------------------------------------------------------------------
    def test_13_groq_api_key_detection_and_defaults(self):
        """Verify that GROQ_API_KEY is detected with correct default base URL and model."""
        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test_dummy_key_123"}, clear=True):
            client = LLMJudgeClient()
            self.assertEqual(client.provider, "groq")
            self.assertEqual(client.base_url, "https://api.groq.com/openai/v1")
            self.assertEqual(client.model_name, "openai/gpt-oss-20b")
            self.assertFalse(client.offline_mode)

    # -----------------------------------------------------------------------
    # 14. Groq Custom Model and Base URL Precedence
    # -----------------------------------------------------------------------
    def test_14_groq_custom_model_and_base_url(self):
        """Verify that GROQ_MODEL and GROQ_BASE_URL override default Groq settings."""
        env = {
            "GROQ_API_KEY": "gsk_test_dummy_key_123",
            "GROQ_MODEL": "llama-3.3-70b-versatile",
            "GROQ_BASE_URL": "https://custom.groq.endpoint/v1",
        }
        with patch.dict("os.environ", env, clear=True):
            client = LLMJudgeClient()
            self.assertEqual(client.provider, "groq")
            self.assertEqual(client.base_url, "https://custom.groq.endpoint/v1")
            self.assertEqual(client.model_name, "llama-3.3-70b-versatile")

    # -----------------------------------------------------------------------
    # 15. OpenAI Fallback Compatibility
    # -----------------------------------------------------------------------
    def test_15_openai_fallback_compatibility(self):
        """Verify that OPENAI_API_KEY continues to configure OpenAI provider when Groq is unset."""
        env = {
            "OPENAI_API_KEY": "sk-test-dummy-openai-key-456",
            "LLM_MODEL": "gpt-4o-mini",
        }
        with patch.dict("os.environ", env, clear=True):
            client = LLMJudgeClient()
            self.assertEqual(client.provider, "openai")
            self.assertEqual(client.base_url, "https://api.openai.com/v1")
            self.assertEqual(client.model_name, "gpt-4o-mini")

    # -----------------------------------------------------------------------
    # 16. Provider Identification When Unavailable
    # -----------------------------------------------------------------------
    def test_16_provider_identification_when_unavailable(self):
        """Verify that missing keys identify provider as unavailable without crashing."""
        with patch.dict("os.environ", {}, clear=True):
            client = LLMJudgeClient()
            self.assertEqual(client.provider, "unavailable")
            self.assertTrue(client.offline_mode)
            self.assertEqual(client.api_key, "")

    # -----------------------------------------------------------------------
    # 17. API Keys Never Serialized Into Result Artifacts
    # -----------------------------------------------------------------------
    def test_17_api_keys_never_serialized_into_artifacts(self):
        """
        Security verification: Asserts that API keys (or prefix patterns)
        are never written to JSON artifacts or Score dictionaries.
        """
        score = JudgeScore(
            helpfulness=5,
            grounding=5,
            actionability=5,
            clarity=5,
            overall=5.0,
            pass_status=True,
            reason="Grounded answer.",
        )
        score_dict = score.to_dict()
        self.assertNotIn("api_key", score_dict)
        self.assertNotIn("key", score_dict)

        # Inspect generated JSON artifacts if they exist
        for artifact_path in [DEFAULT_RESULTS_PATH, DEFAULT_SAMPLE_PATH]:
            if artifact_path.exists():
                with open(artifact_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
                    self.assertNotIn("gsk_", content)
                    self.assertNotIn("sk-", content)
                    self.assertNotIn("api_key", content)


if __name__ == "__main__":
    unittest.main()

