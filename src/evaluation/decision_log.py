"""
decision_log.py
---------------
STAGE 11: Persistent Engineering Decision Log Engine

Manages engineering decisions for agent improvements:
  - Tracks problem, evidence, hypothesis, proposed change, expected impact,
    risk, measured result, decision status, and date.
  - Generates markdown format (reports/decision_log.md)
  - Exports machine-readable JSON records
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_DECISIONS = ["KEEP", "REJECT", "REVISE", "DEFER"]


@dataclass
class DecisionRecord:
    """Represents a single evaluated engineering decision."""

    decision_id: str
    problem: str
    evidence: str
    hypothesis: str
    proposed_change: str
    expected_impact: str
    risk: str
    result: str
    decision: str
    date: str
    before_metrics: Dict[str, Any] = field(default_factory=dict)
    after_metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validates that decision record complies with schema rules."""
        errors = []
        if self.decision not in VALID_DECISIONS:
            errors.append(f"Invalid decision '{self.decision}'. Must be one of {VALID_DECISIONS}")
        if not self.decision_id.startswith("DEC-"):
            errors.append(f"Decision ID must start with 'DEC-': {self.decision_id}")
        if not self.problem or not self.evidence or not self.hypothesis:
            errors.append("Problem, evidence, and hypothesis must not be empty.")
        if not self.proposed_change or not self.result:
            errors.append("Proposed change and result must not be empty.")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Renders decision entry in the required markdown format."""
        before_str = ""
        after_str = ""
        if self.before_metrics and self.after_metrics:
            rows = []
            for k in sorted(self.before_metrics.keys()):
                if k in self.after_metrics:
                    b_val = self.before_metrics[k]
                    a_val = self.after_metrics[k]
                    b_fmt = f"{b_val:.4f}" if isinstance(b_val, float) else str(b_val)
                    a_fmt = f"{a_val:.4f}" if isinstance(a_val, float) else str(a_val)
                    rows.append(f"| {k} | {b_fmt} | {a_fmt} |")
            if rows:
                table_md = "\n\n| Metric | Before | After |\n| --- | --- | --- |\n" + "\n".join(rows)
            else:
                table_md = ""
        else:
            table_md = ""

        return f"""## {self.decision_id}: {self.proposed_change[:60]}...

### Decision ID
{self.decision_id}

### Problem
{self.problem}

### Evidence
{self.evidence}

### Hypothesis
{self.hypothesis}

### Proposed Change
{self.proposed_change}

### Expected Impact
{self.expected_impact}

### Risk
{self.risk}

### Result
{self.result}{table_md}

### Decision
**{self.decision}**

### Date
{self.date}
"""


class DecisionLogManager:
    """Manages the persistent engineering decision log."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.decisions: List[DecisionRecord] = []

    def add_decision(self, record: DecisionRecord) -> None:
        """Validates and appends a decision record."""
        errs = record.validate()
        if errs:
            raise ValueError(f"DecisionRecord validation failed: {errs}")
        # Replace if existing with same decision_id
        self.decisions = [d for d in self.decisions if d.decision_id != record.decision_id]
        self.decisions.append(record)

    def render_markdown(self) -> str:
        """Renders the full markdown decision log document."""
        header = """# Engineering Decision Log: AmazonHelp AI Support Agent

This log tracks all architectural and algorithmic changes tested against the Golden Evaluation Set.
No change is accepted without rigorous before/after metric comparison and regression analysis.

---
"""
        sections = [d.to_markdown() for d in self.decisions]
        return header + "\n---\n\n".join(sections)

    def save(self, json_path: Optional[Path] = None) -> None:
        """Saves decision log to markdown and optional JSON."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as fh:
            fh.write(self.render_markdown())

        if json_path:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump([d.to_dict() for d in self.decisions], fh, indent=2, ensure_ascii=False)
