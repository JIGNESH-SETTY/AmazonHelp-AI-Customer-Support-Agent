"""
comparison.py
-------------
STAGE 10: Comparative Scorecard & Regression Detection Engine

Provides:
  - Quantitative head-to-head comparison between Stage 8 Baselines and Stage 9 Agent
  - Metric-by-metric delta calculation (absolute and percentage gain)
  - Regression detector comparing current runs against historical thresholds
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE8_RESULTS_PATH = REPO_ROOT / "reports" / "stage8_baseline_results.json"


class BaselineComparator:
    """Compares current evaluation metrics with Stage 8 baseline benchmarks."""

    def __init__(self, stage8_path: Optional[Path] = None):
        self.stage8_path = stage8_path or STAGE8_RESULTS_PATH
        self.stage8_data = self._load_stage8_results()

    def _load_stage8_results(self) -> Dict[str, Any]:
        if not self.stage8_path.exists():
            return {}
        with open(self.stage8_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def compare(
        self,
        current_intent_metrics: Dict[str, Any],
        current_nlg_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculates side-by-side scorecard comparing current metrics against
        the top Stage 8 baseline (TF-IDF + Naive Bayes / Canned Template).
        """
        # Extract Stage 8 top metrics
        s8_intent = self.stage8_data.get("intent_classification_baselines", {}).get("tfidf_naive_bayes", {})
        s8_resp = self.stage8_data.get("response_generation_baselines", {}).get("intent_canned_template", {})

        s8_acc = s8_intent.get("accuracy", 0.7050)
        s8_macro_f1 = s8_intent.get("macro_f1", 0.7077)
        s8_weighted_f1 = s8_intent.get("weighted_f1", 0.6865)
        s8_diff = s8_intent.get("accuracy_by_difficulty", {})
        s8_hard_acc = s8_diff.get("hard", 0.6667)
        s8_easy_acc = s8_diff.get("easy", 0.9259)
        s8_med_acc = s8_diff.get("medium", 0.6829)

        s8_guidance = s8_resp.get("guidance_adherence_rate", 0.7950)
        s8_safety = 1.0000
        s8_latency = s8_intent.get("latency_ms", 0.09)

        # Current metrics
        c_acc = current_intent_metrics.get("accuracy", 0.0)
        c_macro_f1 = current_intent_metrics.get("macro_f1", 0.0)
        c_weighted_f1 = current_intent_metrics.get("weighted_f1", 0.0)
        c_diff = current_intent_metrics.get("accuracy_by_difficulty", {})
        c_hard_acc = c_diff.get("hard", 0.0)
        c_easy_acc = c_diff.get("easy", 0.0)
        c_med_acc = c_diff.get("medium", 0.0)

        c_guidance = current_nlg_metrics.get("guidance_adherence_rate", 0.0)
        c_safety = current_nlg_metrics.get("policy_safety_rate", 1.0)
        c_latency = current_nlg_metrics.get("latency_ms", 0.0)

        def make_entry(s8_val: float, c_val: float, higher_is_better: bool = True) -> Dict[str, Any]:
            abs_delta = round(c_val - s8_val, 4)
            pct_delta = round((abs_delta / s8_val * 100.0), 2) if s8_val > 0 else 0.0
            is_better = (c_val >= s8_val) if higher_is_better else (c_val <= s8_val)
            return {
                "stage8_baseline": round(s8_val, 4),
                "current_system": round(c_val, 4),
                "absolute_delta": abs_delta,
                "percentage_delta": pct_delta,
                "improved": is_better,
                "winner": "Current System" if is_better else "Stage 8 Baseline",
            }

        return {
            "overall_accuracy": make_entry(s8_acc, c_acc),
            "macro_f1": make_entry(s8_macro_f1, c_macro_f1),
            "weighted_f1": make_entry(s8_weighted_f1, c_weighted_f1),
            "hard_difficulty_accuracy": make_entry(s8_hard_acc, c_hard_acc),
            "easy_difficulty_accuracy": make_entry(s8_easy_acc, c_easy_acc),
            "medium_difficulty_accuracy": make_entry(s8_med_acc, c_med_acc),
            "guidance_adherence_rate": make_entry(s8_guidance, c_guidance),
            "policy_safety_rate": make_entry(s8_safety, c_safety),
            "latency_ms": make_entry(s8_latency, c_latency, higher_is_better=False),
        }


class RegressionDetector:
    """Detects performance regressions against established benchmark thresholds."""

    def __init__(
        self,
        max_accuracy_drop: float = 0.02,
        max_hard_drop: float = 0.03,
        max_policy_drop: float = 0.01,
        max_latency_increase_factor: float = 3.0,
    ):
        self.max_accuracy_drop = max_accuracy_drop
        self.max_hard_drop = max_hard_drop
        self.max_policy_drop = max_policy_drop
        self.max_latency_increase_factor = max_latency_increase_factor

    def check(self, comparison_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Evaluates comparison scorecard. Returns (passed, regression_warnings).
        """
        warnings: List[str] = []

        acc_delta = comparison_data["overall_accuracy"]["absolute_delta"]
        if acc_delta < -self.max_accuracy_drop:
            warnings.append(f"REGRESSION: Intent accuracy dropped by {abs(acc_delta):.2%}")

        hard_delta = comparison_data["hard_difficulty_accuracy"]["absolute_delta"]
        if hard_delta < -self.max_hard_drop:
            warnings.append(f"REGRESSION: Hard-query accuracy dropped by {abs(hard_delta):.2%}")

        policy_delta = comparison_data["policy_safety_rate"]["absolute_delta"]
        if policy_delta < -self.max_policy_drop:
            warnings.append(f"REGRESSION: Policy safety rate dropped by {abs(policy_delta):.2%}")

        passed = len(warnings) == 0
        return passed, warnings
