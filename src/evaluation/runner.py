"""
runner.py
---------
STAGE 10: Model-Agnostic Evaluation Harness Runner

Executes full multi-dimensional benchmarking for any customer-support agent
or baseline system against the protected Stage 7 Golden Evaluation Set.

Exports all structured artifacts to `reports/stage10/`:
  - evaluation_results.json
  - per_intent_results.csv
  - confusion_matrix.csv
  - latency_results.json
  - baseline_comparison.json
  - regression_report.json
  - error_analysis.jsonl
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


from src.evaluation.comparison import BaselineComparator, RegressionDetector
from src.evaluation.datasets import GoldenEvaluationRecord, GoldenSetLoader
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.evaluate_baselines import load_training_corpus
from src.evaluation.latency import LatencyProfiler
from src.evaluation.metrics import (
    compute_confidence_calibration,
    compute_escalation_metrics,
    compute_generation_and_safety_metrics,
    compute_intent_classification_metrics,
)

# Configure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "stage10"


class SystemUnderTest(Protocol):
    """Protocol for any system evaluated by the harness."""

    def process(self, raw_input: Any) -> Any:
        ...


class EvaluationHarness:
    """Orchestrates comprehensive multi-metric benchmarking."""

    def __init__(
        self,
        golden_loader: Optional[GoldenSetLoader] = None,
        output_dir: Optional[Path] = None,
        random_seed: int = 42,
    ):
        self.loader = golden_loader or GoldenSetLoader()
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.random_seed = random_seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_system(
        self,
        system: SystemUnderTest,
        system_name: str = "SupportAgent",
        run_error_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes complete evaluation of a system against the Golden Evaluation Set.
        """
        golden_records = self.loader.load()
        total_records = len(golden_records)
        print(f"[Stage 10 Harness] Evaluating '{system_name}' on {total_records} Golden Set records...")

        profiler = LatencyProfiler()
        predictions: List[Dict[str, Any]] = []

        # Run inference loop
        for record in golden_records:
            t0 = time.time()
            out = system.process(record.customer_message)
            elapsed_ms = (time.time() - t0) * 1000.0
            profiler.record(elapsed_ms)

            # Standardize output structure
            if hasattr(out, "to_dict"):
                out_dict = out.to_dict()
            elif isinstance(out, dict):
                out_dict = out
            else:
                out_dict = {
                    "primary_intent": str(out),
                    "intent_confidence": 1.0,
                    "response": str(out),
                    "policy": {"passed": True, "violations": []},
                    "escalation": {"required": False, "reason": None, "priority": "none"},
                }
            predictions.append(out_dict)

        # 1. Classification Metrics
        y_true = [r.primary_intent for r in golden_records]
        y_pred = [p["primary_intent"] for p in predictions]
        candidate_lists = [
            [c["intent_id"] for c in p.get("top_candidates", [])] if "top_candidates" in p else []
            for p in predictions
        ]

        class_metrics = compute_intent_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            candidate_lists=candidate_lists,
            random_seed=self.random_seed,
        )

        # Difficulty breakdown
        diff_acc = {}
        for tier in ["easy", "medium", "hard"]:
            idxs = [i for i, r in enumerate(golden_records) if r.difficulty == tier]
            if idxs:
                correct = sum(1 for i in idxs if y_true[i] == y_pred[i])
                diff_acc[tier] = round(correct / len(idxs), 4)
            else:
                diff_acc[tier] = 0.0
        class_metrics["accuracy_by_difficulty"] = diff_acc

        # 2. Generation & Safety Metrics
        hypotheses = [p.get("response", "") for p in predictions]
        references = [r.reference_support_response for r in golden_records]

        nlg_metrics = compute_generation_and_safety_metrics(
            hypotheses=hypotheses,
            references=references,
            gold_intents=y_true,
        )

        # 3. Escalation Metrics
        esc_flags = [p.get("escalation", {}).get("required", False) for p in predictions]
        sentiments = [
            p.get("context", {}).get("sentiment", "neutral") for p in predictions
        ]

        esc_metrics = compute_escalation_metrics(
            escalation_flags=esc_flags,
            gold_intents=y_true,
            sentiments=sentiments,
        )

        # 4. Confidence Calibration
        confidences = [p.get("intent_confidence", 1.0) for p in predictions]
        calib_metrics = compute_confidence_calibration(
            y_true=y_true,
            y_pred=y_pred,
            confidences=confidences,
        )

        # 5. Latency Profile
        latency_profile = profiler.summarize()
        nlg_metrics["latency_ms"] = latency_profile.mean_ms

        # 6. Error Analysis
        error_summary = {}
        if run_error_analysis:
            analyzer = ErrorAnalyzer()
            guidance_scores = [
                compute_generation_and_safety_metrics([h], [ref], [yt])["guidance_adherence_rate"]
                for h, ref, yt in zip(hypotheses, references, y_true)
            ]
            error_summary = analyzer.analyze_dataset(
                golden_records=golden_records,
                predictions=predictions,
                guidance_scores=guidance_scores,
                output_jsonl_path=self.output_dir / "error_analysis.jsonl",
            )

        # 7. Baseline Comparison & Regression Check
        comparator = BaselineComparator()
        comparison_table = comparator.compare(class_metrics, nlg_metrics)

        detector = RegressionDetector()
        passed_regression, regression_warnings = detector.check(comparison_table)

        # Compile final results package
        results_package = {
            "harness_version": "1.0",
            "stage": 10,
            "system_name": system_name,
            "timestamp": "2026-09-12 07:30:00 UTC",
            "random_seed": self.random_seed,
            "total_evaluated": total_records,
            "classification_metrics": class_metrics,
            "generation_metrics": nlg_metrics,
            "escalation_metrics": esc_metrics,
            "confidence_calibration": calib_metrics,
            "latency_profile": latency_profile.to_dict(),
            "error_analysis_summary": error_summary,
            "baseline_comparison": comparison_table,
            "regression_status": {
                "passed": passed_regression,
                "warnings": regression_warnings,
            },
        }

        # Export all artifacts
        self._export_artifacts(results_package, class_metrics, latency_profile, comparison_table, passed_regression, regression_warnings)

        return results_package

    def _export_artifacts(
        self,
        results_package: Dict[str, Any],
        class_metrics: Dict[str, Any],
        latency_profile: Any,
        comparison_table: Dict[str, Any],
        passed_regression: bool,
        regression_warnings: List[str],
    ) -> None:
        """Writes out all required Stage 10 files."""
        # 1. evaluation_results.json
        with open(self.output_dir / "evaluation_results.json", "w", encoding="utf-8") as fh:
            json.dump(results_package, fh, indent=2)

        # 2. per_intent_results.csv
        with open(self.output_dir / "per_intent_results.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["intent_id", "support", "precision", "recall", "f1"])
            for intent_id, stats in class_metrics.get("per_intent", {}).items():
                writer.writerow([
                    intent_id,
                    stats["support"],
                    stats["precision"],
                    stats["recall"],
                    stats["f1"],
                ])

        # 3. confusion_matrix.csv
        conf_mat = class_metrics.get("confusion_matrix", {})
        classes = sorted(list(conf_mat.keys()))
        with open(self.output_dir / "confusion_matrix.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["true_intent"] + classes)
            for c in classes:
                row = [c] + [conf_mat[c].get(pred_c, 0) for pred_c in classes]
                writer.writerow(row)

        # 4. latency_results.json
        with open(self.output_dir / "latency_results.json", "w", encoding="utf-8") as fh:
            json.dump(latency_profile.to_dict(), fh, indent=2)

        # 5. baseline_comparison.json
        with open(self.output_dir / "baseline_comparison.json", "w", encoding="utf-8") as fh:
            json.dump(comparison_table, fh, indent=2)

        # 6. regression_report.json
        with open(self.output_dir / "regression_report.json", "w", encoding="utf-8") as fh:
            json.dump({"passed": passed_regression, "warnings": regression_warnings}, fh, indent=2)

        print(f"[Stage 10 Harness] Successfully exported all artifacts to {self.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 10 Evaluation Harness Runner")
    parser.add_argument("--system", choices=["agent", "baseline"], default="agent", help="System to evaluate")
    parser.add_argument("--ablation", action="store_true", help="Run 4-way ablation evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    print("==================================================")
    print("STAGE 10: Running Evaluation Harness")
    print("==================================================")

    # Ingest training data strictly for fitting models
    train_texts, train_labels, train_meta = load_training_corpus(max_samples=25_000)

    harness = EvaluationHarness(output_dir=Path(args.output_dir), random_seed=args.seed)

    from src.agent.agent import SupportAgent
    from src.agent.config import AgentConfig

    agent = SupportAgent(config=AgentConfig())
    agent.fit_training_data(train_texts, train_labels, train_meta)

    results = harness.evaluate_system(agent, system_name="Stage 9 SupportAgent")

    # Display clean executive console summary
    cm = results["classification_metrics"]
    gm = results["generation_metrics"]
    em = results["escalation_metrics"]
    comp = results["baseline_comparison"]
    reg = results["regression_status"]

    print("\n" + "=" * 55)
    print("STAGE 10 EVALUATION HARNESS SUMMARY")
    print("=" * 55)
    print(f"Overall Accuracy       : {cm['accuracy']:.2%} (95% CI: [{cm['accuracy_ci_95'][0]:.2%}, {cm['accuracy_ci_95'][1]:.2%}])")
    print(f"Macro F1               : {cm['macro_f1']:.4f}")
    print(f"Weighted F1            : {cm['weighted_f1']:.4f}")
    print(f"Hard Difficulty Acc    : {cm['accuracy_by_difficulty']['hard']:.2%}")
    print(f"Guidance Adherence     : {gm['guidance_adherence_rate']:.2%}")
    print(f"Policy Safety Rate     : {gm['policy_safety_rate']:.2%}")
    print(f"Escalation Rate        : {em['overall_escalation_rate']:.2%}")
    print(f"Mean Latency           : {results['latency_profile']['mean_ms']:.2f} ms")
    print(f"Regression Check       : {'PASS' if reg['passed'] else 'FAIL: ' + str(reg['warnings'])}")
    print("=" * 55)


if __name__ == "__main__":
    main()
