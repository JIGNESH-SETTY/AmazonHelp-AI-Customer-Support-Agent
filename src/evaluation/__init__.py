"""
src.evaluation
==============
Comprehensive evaluation harness and benchmarking module for the Hiver AI Support Agent.

STAGE 10: Evaluation Harness
----------------------------
Components:
  - `GoldenSetLoader`: Safe, read-only Golden Set loader with immutability and zero-leakage guarantees.
  - `EvaluationHarness`: Multi-metric benchmarking runner for any customer-support agent or baseline.
  - `compute_intent_classification_metrics`: Accuracy, Macro/Weighted F1, Top-k, Confusion Matrix, and 95% Bootstrap CI.
  - `compute_generation_and_safety_metrics`: ROUGE-1/2/L, BLEU-1/2, Guidance Adherence, Policy Safety Rate.
  - `compute_escalation_metrics`: Overall, Correct, Unnecessary, and Missed Escalation Rates.
  - `compute_confidence_calibration`: Expected Calibration Error (ECE) and confidence separation.
  - `LatencyProfiler`: Execution timing profiler (Mean, Median, P95, Min, Max).
  - `ErrorAnalyzer`: Automated failure categorizer generating line-by-line JSONL diagnostics.
  - `BaselineComparator` & `RegressionDetector`: Comparative scorecard and regression testing engine.
"""

from src.evaluation.comparison import BaselineComparator, RegressionDetector
from src.evaluation.datasets import GoldenEvaluationRecord, GoldenSetLoader
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.latency import LatencyProfile, LatencyProfiler
from src.evaluation.metrics import (
    compute_confidence_calibration,
    compute_escalation_metrics,
    compute_generation_and_safety_metrics,
    compute_intent_classification_metrics,
)
from src.evaluation.runner import EvaluationHarness, SystemUnderTest
from src.evaluation.decision_log import DecisionLogManager, DecisionRecord
from src.evaluation.experiments import ExperimentFramework, ExperimentResult
from src.evaluation.failure_analysis import FailureAnalysisEngine, FailureDiagnosisEngine, FailureRecord

__all__ = [
    "GoldenSetLoader",
    "GoldenEvaluationRecord",
    "EvaluationHarness",
    "SystemUnderTest",
    "compute_intent_classification_metrics",
    "compute_generation_and_safety_metrics",
    "compute_escalation_metrics",
    "compute_confidence_calibration",
    "LatencyProfiler",
    "LatencyProfile",
    "ErrorAnalyzer",
    "BaselineComparator",
    "RegressionDetector",
    "FailureRecord",
    "FailureAnalysisEngine",
    "FailureDiagnosisEngine",
    "DecisionRecord",
    "DecisionLogManager",
    "ExperimentFramework",
    "ExperimentResult",
]
