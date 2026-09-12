"""
experiments.py
--------------
STAGE 11: Controlled Experiment Framework & Regression Guard

Enables testing algorithmic changes against the Golden Evaluation Set:
  1. Runs candidate agent alongside baseline agent on identical golden set.
  2. Compares BEFORE vs AFTER metrics:
     - accuracy
     - macro_f1
     - hard_accuracy
     - guidance_adherence_rate
     - policy_safety_rate
     - escalation_rate
     - latency_ms
  3. Detects metric regressions against configurable safety thresholds.
  4. Outputs structured experiment report and integration payloads for the Decision Log.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.datasets import GoldenEvaluationRecord, GoldenSetLoader
from src.evaluation.runner import EvaluationHarness


@dataclass
class MetricDelta:
    metric_name: str
    before_value: float
    after_value: float
    absolute_delta: float
    percentage_delta: float
    improved: bool
    is_regression: bool
    threshold: float


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    description: str
    timestamp: str
    before_summary: Dict[str, Any]
    after_summary: Dict[str, Any]
    deltas: Dict[str, MetricDelta]
    regressions: List[str]
    has_regressions: bool
    recommendation: str  # KEEP, REJECT, REVISE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "before_summary": self.before_summary,
            "after_summary": self.after_summary,
            "deltas": {k: asdict(v) for k, v in self.deltas.items()},
            "regressions": self.regressions,
            "has_regressions": self.has_regressions,
            "recommendation": self.recommendation,
        }


class ExperimentFramework:
    """
    Executes controlled experiments and regression tests between agent variants.
    """

    DEFAULT_REGRESSION_THRESHOLDS = {
        "accuracy": -0.01,              # Max allowed drop: 1.0%
        "macro_f1": -0.015,             # Max allowed drop: 1.5%
        "hard_accuracy": -0.02,         # Max allowed drop: 2.0%
        "guidance_adherence_rate": -0.02, # Max allowed drop: 2.0%
        "policy_safety_rate": 0.0,       # Zero tolerance for policy safety drop!
        "latency_ms": 10.0,             # Max allowed increase: +10ms
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or self.DEFAULT_REGRESSION_THRESHOLDS

    def compare_metrics(
        self,
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any],
    ) -> Tuple[Dict[str, MetricDelta], List[str]]:
        """
        Computes deltas and checks for regressions across critical support metrics.
        """
        deltas: Dict[str, MetricDelta] = {}
        regressions: List[str] = []

        # Target metrics to evaluate
        targets = [
            ("accuracy", True),
            ("macro_f1", True),
            ("hard_accuracy", True),
            ("guidance_adherence_rate", True),
            ("policy_safety_rate", True),
            ("escalation_rate", None),  # Context-dependent
            ("latency_ms", False),       # Lower is better
        ]

        for m_name, higher_is_better in targets:
            b_val = float(before_metrics.get(m_name, 0.0))
            a_val = float(after_metrics.get(m_name, 0.0))
            abs_delta = round(a_val - b_val, 4)
            pct_delta = round(((a_val - b_val) / b_val * 100), 2) if b_val != 0 else 0.0

            thresh = self.thresholds.get(m_name, 0.0)

            if higher_is_better is True:
                improved = abs_delta > 0.0
                is_reg = abs_delta < thresh
            elif higher_is_better is False:
                improved = abs_delta < 0.0
                is_reg = abs_delta > thresh
            else:
                improved = True
                is_reg = False

            if is_reg:
                regressions.append(
                    f"Regression detected in '{m_name}': {b_val:.4f} -> {a_val:.4f} (delta: {abs_delta:+.4f}, threshold: {thresh})"
                )

            deltas[m_name] = MetricDelta(
                metric_name=m_name,
                before_value=b_val,
                after_value=a_val,
                absolute_delta=abs_delta,
                percentage_delta=pct_delta,
                improved=improved,
                is_regression=is_reg,
                threshold=thresh,
            )

        return deltas, regressions

    def run_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        before_runner_result: Dict[str, Any],
        after_runner_result: Dict[str, Any],
    ) -> ExperimentResult:
        """
        Wraps before and after evaluation runner results into an ExperimentResult.
        """
        # Extract scalar metric summaries
        def extract_scalars(res: Dict[str, Any]) -> Dict[str, float]:
            clf = res.get("classification_metrics", {})
            gen = res.get("generation_metrics", {})
            esc = res.get("escalation_metrics", {})
            diff = clf.get("accuracy_by_difficulty", {})
            return {
                "accuracy": clf.get("accuracy", 0.0),
                "macro_f1": clf.get("macro_f1", 0.0),
                "hard_accuracy": diff.get("hard", 0.0),
                "guidance_adherence_rate": gen.get("guidance_adherence_rate", 0.0),
                "policy_safety_rate": gen.get("policy_safety_rate", 1.0),
                "escalation_rate": esc.get("overall_escalation_rate", 0.0),
                "latency_ms": gen.get("latency_ms", 0.0),
            }

        b_summary = extract_scalars(before_runner_result)
        a_summary = extract_scalars(after_runner_result)

        deltas, regressions = self.compare_metrics(b_summary, a_summary)

        # Decision rule:
        # If policy safety regressed or accuracy dropped significantly -> REJECT
        # If accuracy improved and no regressions -> KEEP
        # Otherwise -> REVISE
        has_reg = len(regressions) > 0
        acc_delta = deltas["accuracy"].absolute_delta
        policy_delta = deltas["policy_safety_rate"].absolute_delta

        if policy_delta < 0.0 or (has_reg and acc_delta < 0.0):
            rec = "REJECT"
        elif not has_reg and acc_delta > 0.0:
            rec = "KEEP"
        elif acc_delta == 0.0 and not has_reg:
            rec = "DEFER"
        else:
            rec = "REVISE"

        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        return ExperimentResult(
            experiment_id=experiment_id,
            name=name,
            description=description,
            timestamp=ts,
            before_summary=b_summary,
            after_summary=a_summary,
            deltas=deltas,
            regressions=regressions,
            has_regressions=has_reg,
            recommendation=rec,
        )
