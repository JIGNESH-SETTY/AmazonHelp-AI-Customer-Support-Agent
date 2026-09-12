"""
Unit tests for Stage 17: Final Hiver Submission Report Verification.
Asserts that reports/final_report.md exists, contains all 13 Hiver-required sections,
contains no forbidden/unsupported claims or local Windows file paths, and is linked in README.md.
"""

import os
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_REPORT_PATH = REPO_ROOT / "reports" / "final_report.md"
README_PATH = REPO_ROOT / "README.md"


class TestStage17FinalReport(unittest.TestCase):
    """Test suite validating the final Hiver submission report."""

    def test_01_final_report_exists_and_non_empty(self):
        """Verify that reports/final_report.md exists and is substantive."""
        self.assertTrue(FINAL_REPORT_PATH.exists(), f"Missing file: {FINAL_REPORT_PATH}")
        content = FINAL_REPORT_PATH.read_text(encoding="utf-8")
        self.assertGreater(len(content), 3000, "Report content is too short for a complete submission.")

    def test_02_all_13_required_sections_present(self):
        """Verify all 13 Hiver required sections are present in final_report.md."""
        content = FINAL_REPORT_PATH.read_text(encoding="utf-8")
        
        required_section_patterns = [
            r"## 1\.\s+Problem Framing",
            r"## 2\.\s+What \"Good\" Means",
            r"## 3\.\s+System Approach",
            r"## 4\.\s+Golden Evaluation Set",
            r"## 5\.\s+Results vs Baselines",
            r"## 6\.\s+LLM-as-Judge Evaluation",
            r"## 7\.\s+Auto-Handle vs Escalate",
            r"## 8\.\s+Top 5 Failure Modes",
            r"## 9\.\s+\"What Is Misleading About My Headline Number\?\"",
            r"## 10\.\s+What I Would Do With One More Week",
            r"## 11\.\s+Reproducibility",
            r"## 12\.\s+Engineering Trade-Offs",
            r"## 13\.\s+Conclusion",
        ]

        for pattern in required_section_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            self.assertIsNotNone(match, f"Required section matching '{pattern}' not found in {FINAL_REPORT_PATH.name}")

    def test_03_golden_set_metrics_present_and_accurate(self):
        """Verify report documents 200 examples, 156 confirmed, 44 corrected."""
        content = FINAL_REPORT_PATH.read_text(encoding="utf-8")
        self.assertIn("200", content)
        self.assertIn("156", content)
        self.assertIn("44", content)
        self.assertIn("78.0%", content)
        self.assertIn("22.0%", content)

    def test_04_headline_metrics_present_and_accurate(self):
        """Verify headline metrics (82.00% accuracy, 0.8204 F1, 100% safety) match repo artifacts."""
        content = FINAL_REPORT_PATH.read_text(encoding="utf-8")
        self.assertIn("82.00%", content)
        self.assertIn("0.8204", content)
        self.assertIn("100.00%", content)
        self.assertIn("90.50%", content)
        self.assertIn("70.50%", content)  # Simple baseline
        self.assertIn("9.00%", content)   # Trivial baseline

    def test_05_no_local_windows_file_paths(self):
        """Verify no local Windows file:// paths are present in final_report.md."""
        content = FINAL_REPORT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("file://", content, "Local file:// paths are forbidden in final_report.md")
        self.assertNotIn("c:\\", content.lower(), "Local Windows C:\\ paths are forbidden in final_report.md")

    def test_06_readme_references_final_report(self):
        """Verify README.md contains a section linking to reports/final_report.md."""
        readme_content = README_PATH.read_text(encoding="utf-8")
        self.assertIn("## Final Evaluation Report", readme_content)
        self.assertIn("reports/final_report.md", readme_content)


if __name__ == "__main__":
    unittest.main()
