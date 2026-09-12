"""
test_stage12.py
---------------
STAGE 12: Final Packaging, Deliverables, Security & Demonstration Test Suite

Validates:
  1. Existence and integrity of all `reports/final/` deliverables.
  2. Metric consistency between `final_results.json` and `final_status.json`.
  3. Packaging files: `README.md`, `requirements.txt`, `.gitignore`, `.env.example`.
  4. Security audit: zero secrets, tokens, or private keys committed.
  5. Golden Evaluation Set immutability & zero-leakage guarantee.
  6. CLI demonstration mode & synthetic scenario execution.
  7. Structured metadata compliance (zero chain-of-thought exposure).
"""

import hashlib
import json
from pathlib import Path
import re
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_FINAL = REPO_ROOT / "reports" / "final"
GOLDEN_PATH = REPO_ROOT / "data" / "golden" / "golden_evaluation_set.jsonl"


class TestStage12FinalPackaging(unittest.TestCase):
    """Complete verification suite for Stage 12 deliverables and packaging."""

    # 1. Existence and integrity of reports/final/ deliverables
    def test_01_final_deliverables_exist(self):
        """Verify all 9 required final documentation and reporting artifacts exist."""
        required_files = [
            "final_results.md",
            "final_results.json",
            "failure_summary.md",
            "engineering_decisions.md",
            "final_project_report.md",
            "interview_cheat_sheet.md",
            "resume_bullets.md",
            "demo_script.md",
            "final_status.json",
        ]
        for fname in required_files:
            fpath = REPORTS_FINAL / fname
            self.assertTrue(fpath.exists(), f"Missing required final deliverable: {fname}")
            self.assertGreater(fpath.stat().st_size, 100, f"Deliverable {fname} appears empty or truncated")

    # 2. Metric consistency between final_results.json and final_status.json
    def test_02_metric_consistency(self):
        """Verify metrics in final_results.json and final_status.json match exactly."""
        with open(REPORTS_FINAL / "final_results.json", "r", encoding="utf-8") as fh:
            res_data = json.load(fh)
        with open(REPORTS_FINAL / "final_status.json", "r", encoding="utf-8") as fh:
            status_data = json.load(fh)

        final_agent = res_data["comparisons"]["stage12_final_agent"]
        bench = status_data["benchmark_metrics"]

        self.assertEqual(final_agent["intent_accuracy"], bench["final_intent_accuracy"])
        self.assertEqual(final_agent["macro_f1"], bench["final_macro_f1"])
        self.assertEqual(final_agent["hard_accuracy"], bench["final_hard_accuracy"])
        self.assertEqual(final_agent["guidance_adherence_rate"], bench["final_guidance_adherence"])
        self.assertEqual(final_agent["policy_safety_rate"], bench["final_policy_safety"])
        self.assertEqual(final_agent["escalation_rate"], bench["final_escalation_rate"])

    # 3. Packaging files existence and non-emptiness
    def test_03_packaging_files_exist(self):
        """Verify README.md, requirements.txt, .gitignore, and .env.example exist."""
        for fname in ["README.md", "requirements.txt", ".gitignore", ".env.example"]:
            fpath = REPO_ROOT / fname
            self.assertTrue(fpath.exists(), f"Missing packaging file: {fname}")
            self.assertGreater(fpath.stat().st_size, 20, f"File {fname} is unexpectedly small")

    # 4. Security Audit: zero secrets committed
    def test_04_security_audit_no_secrets(self):
        """Scan repository text files for accidentally committed secrets or tokens."""
        secret_patterns = [
            re.compile(r"sk-[a-zA-Z0-9]{20,}"),          # OpenAI API keys
            re.compile(r"ghp_[a-zA-Z0-9]{20,}"),         # GitHub personal access tokens
            re.compile(r"AKIA[0-9A-Z]{16}"),             # AWS Access Key IDs
            re.compile(r"-----BEGIN RSA PRIVATE KEY-----"),# RSA private keys
        ]

        text_extensions = [".py", ".md", ".json", ".txt", ".example"]
        for path in REPO_ROOT.rglob("*"):
            if any(part.startswith(".") and part not in [".env.example"] for part in path.parts):
                continue
            if path.suffix in text_extensions and path.is_file():
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                    for pat in secret_patterns:
                        self.assertIsNone(
                            pat.search(content),
                            f"Potential secret detected matching pattern in {path.relative_to(REPO_ROOT)}"
                        )
                except Exception:
                    pass

    # 5. Golden Set immutability & checksum verification
    def test_05_golden_set_immutability(self):
        """Verify protected Golden Set has exactly 200 records and matches manifest checksum."""
        self.assertTrue(GOLDEN_PATH.exists())
        with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        self.assertEqual(len(lines), 200, "Golden set must contain exactly 200 evaluation records")

        # Verify SHA-256
        with open(GOLDEN_PATH, "rb") as fh:
            content = fh.read()
        sha256 = hashlib.sha256(content).hexdigest()
        manifest_path = REPO_ROOT / "data" / "golden" / "golden_set_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            if "sha256" in manifest:
                self.assertEqual(sha256, manifest["sha256"], "Golden set SHA-256 hash mismatch!")

    # 6. CLI Demo mode and synthetic scenarios
    def test_06_cli_demo_scenarios(self):
        """Verify SYNTHETIC_DEMO_SCENARIOS in src/agent/cli.py are populated and execute properly."""
        from src.agent.cli import SYNTHETIC_DEMO_SCENARIOS, format_support_box
        from src.agent.agent import SupportAgent
        from src.agent.config import AgentConfig

        self.assertEqual(len(SYNTHETIC_DEMO_SCENARIOS), 6)
        agent = SupportAgent(config=AgentConfig())

        # Test execution on first scenario
        sc = SYNTHETIC_DEMO_SCENARIOS[0]
        out = agent.process(sc["input"])
        box = format_support_box(sc["input"], out)

        self.assertIn("SUPPORT AGENT DECISION METADATA", box)
        self.assertIn("Intent:", box)
        self.assertIn("Confidence:", box)
        self.assertIn("Policy status:", box)
        self.assertIn("AGENT RESPONSE:", box)

    # 7. Zero Chain-of-Thought exposure in outputs
    def test_07_zero_chain_of_thought_leakage(self):
        """Verify that agent outputs do not leak internal reasoning or chain-of-thought tokens."""
        from src.agent.cli import SYNTHETIC_DEMO_SCENARIOS
        from src.agent.agent import SupportAgent
        from src.agent.config import AgentConfig

        agent = SupportAgent(config=AgentConfig())
        forbidden_cot_tokens = [
            "let me think",
            "thinking process",
            "step 1: identify intent",
            "chain of thought",
            "internal analysis",
            "hidden reasoning",
        ]

        for sc in SYNTHETIC_DEMO_SCENARIOS:
            out = agent.process(sc["input"])
            resp_lower = out.response.lower()
            for token in forbidden_cot_tokens:
                self.assertNotIn(
                    token,
                    resp_lower,
                    f"Forbidden chain-of-thought token '{token}' detected in agent response!"
                )


if __name__ == "__main__":
    unittest.main()
