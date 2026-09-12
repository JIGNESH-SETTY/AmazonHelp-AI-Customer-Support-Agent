"""
judge_agreement.py
------------------
STAGE 13: Judge-Human Agreement Evaluation Engine

Calculates inter-rater reliability metrics between human annotations and LLM judge scores:
  - Exact agreement rate (%)
  - Mean Absolute Difference (MAD)
  - Pass/Fail binary agreement rate (%)
  - Cohen's Kappa for Pass/Fail decisions
  - Pearson / ordinal correlation for 1-5 ratings
  - Per-criterion agreement analysis (Helpfulness, Grounding, Actionability, Clarity)

Anti-Fabrication Safeguards:
  - Validates completeness of human annotations before computing agreement.
  - If human ratings are missing or incomplete, explicitly reports:
    "Human annotation incomplete: agreement analysis not yet available."
  - NEVER substitutes synthetic or imputed human ratings.
"""

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Windows UTF-8 stdout configuration
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

RUBRIC_DIMENSIONS = ["helpfulness", "grounding", "actionability", "clarity", "overall"]
REQUIRED_HUMAN_COLUMNS = [
    "example_id",
    "human_helpfulness",
    "human_grounding",
    "human_actionability",
    "human_clarity",
    "human_overall",
    "human_pass",
    "annotator_notes",
]


def compute_exact_agreement(human_vals: List[float], judge_vals: List[float]) -> float:
    """Computes proportion of instances where human and judge gave identical integer ratings."""
    assert len(human_vals) == len(judge_vals), "Length mismatch"
    if not human_vals:
        return 0.0
    matches = sum(1 for h, j in zip(human_vals, judge_vals) if round(h) == round(j))
    return round(matches / len(human_vals), 4)


def compute_adjacent_agreement(human_vals: List[float], judge_vals: List[float]) -> float:
    """Computes proportion of instances where human and judge ratings differ by at most 1 point (|h - j| <= 1)."""
    assert len(human_vals) == len(judge_vals), "Length mismatch"
    if not human_vals:
        return 0.0
    matches = sum(1 for h, j in zip(human_vals, judge_vals) if abs(round(h) - round(j)) <= 1)
    return round(matches / len(human_vals), 4)


def compute_mean_absolute_difference(human_vals: List[float], judge_vals: List[float]) -> float:
    """Computes Mean Absolute Difference (MAD) between human and judge ratings."""
    assert len(human_vals) == len(judge_vals), "Length mismatch"
    if not human_vals:
        return 0.0
    mad = sum(abs(h - j) for h, j in zip(human_vals, judge_vals)) / len(human_vals)
    return round(mad, 4)


def compute_pass_fail_agreement(human_pass: List[bool], judge_pass: List[bool]) -> float:
    """Computes proportion of instances where pass/fail classifications match."""
    assert len(human_pass) == len(judge_pass), "Length mismatch"
    if not human_pass:
        return 0.0
    matches = sum(1 for h, j in zip(human_pass, judge_pass) if bool(h) == bool(j))
    return round(matches / len(human_pass), 4)


def compute_cohens_kappa(human_decisions: List[bool], judge_decisions: List[bool]) -> float:
    """
    Computes Cohen's Kappa coefficient (kappa) for binary pass/fail agreement.
    Formula: kappa = (Po - Pe) / (1 - Pe)
    where Po = observed agreement, Pe = hypothetical probability of chance agreement.
    """
    assert len(human_decisions) == len(judge_decisions), "Length mismatch"
    total = len(human_decisions)
    if total == 0:
        return 0.0

    # Contingency counts
    # tp: both Pass, fn: human Pass & judge Fail, fp: human Fail & judge Pass, tn: both Fail
    tp = sum(1 for h, j in zip(human_decisions, judge_decisions) if h and j)
    tn = sum(1 for h, j in zip(human_decisions, judge_decisions) if not h and not j)
    fp = sum(1 for h, j in zip(human_decisions, judge_decisions) if not h and j)
    fn = sum(1 for h, j in zip(human_decisions, judge_decisions) if h and not j)

    # Observed agreement
    p_o = (tp + tn) / total

    # Chance agreement
    p_human_pass = (tp + fn) / total
    p_human_fail = (fp + tn) / total
    p_judge_pass = (tp + fp) / total
    p_judge_fail = (fn + tn) / total

    p_e = (p_human_pass * p_judge_pass) + (p_human_fail * p_judge_fail)

    if math.isclose(1.0, p_e, rel_tol=1e-9):
        # Perfect agreement on homogenous distribution
        return 1.0 if math.isclose(p_o, 1.0) else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return round(max(-1.0, min(1.0, kappa)), 4)


def compute_ordinal_correlation(human_vals: List[float], judge_vals: List[float]) -> float:
    """Computes Pearson correlation coefficient between human and judge ratings."""
    assert len(human_vals) == len(judge_vals), "Length mismatch"
    n = len(human_vals)
    if n < 2:
        return 0.0

    mean_h = sum(human_vals) / n
    mean_j = sum(judge_vals) / n

    var_h = sum((x - mean_h) ** 2 for x in human_vals)
    var_j = sum((y - mean_j) ** 2 for y in judge_vals)

    if var_h == 0.0 or var_j == 0.0:
        # Zero variance: constant ratings
        return 1.0 if human_vals == judge_vals else 0.0

    covar = sum((x - mean_h) * (y - mean_j) for x, y in zip(human_vals, judge_vals))
    corr = covar / math.sqrt(var_h * var_j)
    return round(max(-1.0, min(1.0, corr)), 4)


def load_human_annotations(csv_path: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Loads human annotation CSV file and validates completeness.
    Returns (is_complete, records).
    is_complete is True only if all required rating fields are non-empty for all records.
    """
    if not csv_path.exists():
        return False, []

    records: List[Dict[str, Any]] = []
    has_empty_ratings = False

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(row)
            # Check if any rating column is empty or whitespace
            for col in ["human_helpfulness", "human_grounding", "human_actionability", "human_clarity", "human_overall", "human_pass"]:
                val = row.get(col, "").strip()
                if not val:
                    has_empty_ratings = True

    if not records or has_empty_ratings:
        return False, records

    return True, records


def analyze_agreement(
    human_csv_path: Path,
    judge_results_path: Path,
) -> Dict[str, Any]:
    """
    Coordinates judge-human agreement analysis.
    If human annotations are incomplete, returns an explicit pending status.
    """
    is_complete, human_records = load_human_annotations(human_csv_path)

    if not is_complete or not human_records:
        completed_count = 0
        if human_records:
            completed_count = sum(
                1 for r in human_records
                if all(bool(r.get(c, "").strip()) for c in ["human_helpfulness", "human_grounding", "human_actionability", "human_clarity", "human_overall", "human_pass"])
            )
        return {
            "status": "pending_human_annotation",
            "message": "Human annotation incomplete: agreement analysis not yet available.",
            "annotated_count": completed_count,
            "required_count": len(human_records) if human_records else 50,
            "metrics": {},
            "notes": (
                "Anti-fabrication safeguard active: ratings must be manually populated in "
                f"{human_csv_path.name} before calculating statistical agreement."
            ),
        }

    # Load judge results
    if not judge_results_path.exists():
        return {
            "status": "pending_judge_results",
            "message": f"Judge results not found at {judge_results_path.name}.",
            "annotated_count": len(human_records),
            "required_count": len(human_records),
            "metrics": {},
        }

    with open(judge_results_path, "r", encoding="utf-8") as fh:
        judge_data = json.load(fh)

    judge_evals = judge_data.get("evaluations", [])
    if not judge_evals:
        return {
            "status": "pending_judge_results",
            "message": "LLM Judge evaluations are not yet populated.",
            "annotated_count": len(human_records),
            "required_count": len(human_records),
            "metrics": {},
        }

    judge_map = {e["example_id"]: e["judge_score"] for e in judge_evals}

    # Align matched examples
    matched_ids = [r["example_id"] for r in human_records if r["example_id"] in judge_map]
    if not matched_ids:
        return {
            "status": "no_matching_examples",
            "message": "No matching example IDs between human annotations and judge results.",
            "metrics": {},
        }

    human_dict = {r["example_id"]: r for r in human_records}

    # Extract vectors
    h_help = [float(human_dict[eid]["human_helpfulness"]) for eid in matched_ids]
    j_help = [float(judge_map[eid]["helpfulness"]) for eid in matched_ids]

    h_ground = [float(human_dict[eid]["human_grounding"]) for eid in matched_ids]
    j_ground = [float(judge_map[eid]["grounding"]) for eid in matched_ids]

    h_act = [float(human_dict[eid]["human_actionability"]) for eid in matched_ids]
    j_act = [float(judge_map[eid]["actionability"]) for eid in matched_ids]

    h_clar = [float(human_dict[eid]["human_clarity"]) for eid in matched_ids]
    j_clar = [float(judge_map[eid]["clarity"]) for eid in matched_ids]

    h_over = [float(human_dict[eid]["human_overall"]) for eid in matched_ids]
    j_over = [float(judge_map[eid]["overall"]) for eid in matched_ids]

    def parse_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in ("true", "1", "pass", "yes")

    h_pass = [parse_bool(human_dict[eid]["human_pass"]) for eid in matched_ids]
    j_pass = [parse_bool(judge_map[eid]["pass"]) for eid in matched_ids]

    kappa = compute_cohens_kappa(h_pass, j_pass)
    pass_agr = compute_pass_fail_agreement(h_pass, j_pass)

    per_criterion = {
        "helpfulness": {
            "exact_agreement": compute_exact_agreement(h_help, j_help),
            "adjacent_agreement": compute_adjacent_agreement(h_help, j_help),
            "mean_absolute_difference": compute_mean_absolute_difference(h_help, j_help),
            "correlation": compute_ordinal_correlation(h_help, j_help),
        },
        "grounding": {
            "exact_agreement": compute_exact_agreement(h_ground, j_ground),
            "adjacent_agreement": compute_adjacent_agreement(h_ground, j_ground),
            "mean_absolute_difference": compute_mean_absolute_difference(h_ground, j_ground),
            "correlation": compute_ordinal_correlation(h_ground, j_ground),
        },
        "actionability": {
            "exact_agreement": compute_exact_agreement(h_act, j_act),
            "adjacent_agreement": compute_adjacent_agreement(h_act, j_act),
            "mean_absolute_difference": compute_mean_absolute_difference(h_act, j_act),
            "correlation": compute_ordinal_correlation(h_act, j_act),
        },
        "clarity": {
            "exact_agreement": compute_exact_agreement(h_clar, j_clar),
            "adjacent_agreement": compute_adjacent_agreement(h_clar, j_clar),
            "mean_absolute_difference": compute_mean_absolute_difference(h_clar, j_clar),
            "correlation": compute_ordinal_correlation(h_clar, j_clar),
        },
        "overall": {
            "exact_agreement": compute_exact_agreement(h_over, j_over),
            "adjacent_agreement": compute_adjacent_agreement(h_over, j_over),
            "mean_absolute_difference": compute_mean_absolute_difference(h_over, j_over),
            "correlation": compute_ordinal_correlation(h_over, j_over),
        },
    }

    return {
        "status": "completed",
        "sample_size": len(matched_ids),
        "metrics": {
            "pass_fail_agreement_rate": pass_agr,
            "cohens_kappa_pass_fail": kappa,
            "overall_exact_agreement": per_criterion["overall"]["exact_agreement"],
            "overall_mean_absolute_difference": per_criterion["overall"]["mean_absolute_difference"],
            "overall_ordinal_correlation": per_criterion["overall"]["correlation"],
            "per_criterion": per_criterion,
        },
        "notes": "Inter-rater reliability evaluated against completed human consensus annotations.",
    }


def export_agreement_reports(agreement_data: Dict[str, Any], output_dir: Path) -> None:
    """Exports structured JSON and Markdown agreement artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "judge_agreement.json"
    md_path = output_dir / "judge_agreement.md"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(agreement_data, fh, indent=2)

    status = agreement_data.get("status", "unknown")
    msg = agreement_data.get("message", "")
    metrics = agreement_data.get("metrics", {})

    md = [
        "# Stage 13: Judge-Human Agreement Report",
        "",
        f"- **Agreement Status**: `{status}`",
        f"- **Audit Date**: 2026-09-12",
        "",
        "---",
        "",
        "## 1. Agreement Status & Evidence",
        "",
    ]

    if status == "pending_human_annotation":
        md.extend([
            "> [!IMPORTANT]",
            "> **Human Annotation Incomplete: Agreement Analysis Not Yet Available.**",
            "",
            "To satisfy Hiver's inter-rater reliability requirements without fabricating data:",
            f"1. Open `data/golden/human_judge_annotations.csv` ({agreement_data.get('required_count', 50)} sample rows).",
            "2. Read the rubric definitions in `reports/stage13/human_annotation_guide.md`.",
            "3. Score each example across the 4 dimensions (1–5 scale) and set `human_pass` to True/False.",
            "4. Re-run `python -m src.evaluation.llm_judge` to compute Cohen's kappa and MAD.",
            "",
            "```text",
            "Anti-Fabrication Policy: Synthetic or simulated human ratings are strictly forbidden.",
            "The repository preserves an unpopulated template until real human consensus is gathered.",
            "```",
        ])
    elif status == "completed":
        md.extend([
            "| Agreement Dimension | Score | Interpretation / Target |",
            "| :--- | :---: | :--- |",
            f"| **Pass/Fail Agreement Rate** | **{metrics.get('pass_fail_agreement_rate', 0):.1%}** | Binary alignment on overall reply acceptability |",
            f"| **Cohen's Kappa ($\\kappa$)** | **{metrics.get('cohens_kappa_pass_fail', 0):.4f}** | Inter-rater reliability corrected for chance ($> 0.60$ substantial) |",
            f"| **Overall Exact Agreement** | **{metrics.get('overall_exact_agreement', 0):.1%}** | Exact 1–5 score parity |",
            f"| **Overall Mean Absolute Difference (MAD)** | **{metrics.get('overall_mean_absolute_difference', 0):.3f}** | Average point deviation on 5-point scale ($< 0.50$ ideal) |",
            f"| **Ordinal Correlation ($r$)** | **{metrics.get('overall_ordinal_correlation', 0):.3f}** | Linear score ranking alignment |",
            "",
            "### Per-Criterion Reliability Breakdown:",
            "",
            "| Criterion | Exact Agreement | Mean Absolute Difference | Correlation |",
            "| :--- | :---: | :---: | :---: |",
        ])
        for crit, stats in metrics.get("per_criterion", {}).items():
            md.append(f"| **{crit.title()}** | {stats['exact_agreement']:.1%} | {stats['mean_absolute_difference']:.3f} | {stats['correlation']:.3f} |")
    else:
        md.append(f"Status note: {msg}")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
