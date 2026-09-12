# Stage 13: Judge-Human Agreement Report

- **Agreement Status**: `pending_human_annotation`
- **Audit Date**: 2026-09-12

---

## 1. Agreement Status & Evidence

> [!IMPORTANT]
> **Human Annotation Incomplete: Agreement Analysis Not Yet Available.**

To satisfy Hiver's inter-rater reliability requirements without fabricating data:
1. Open `data/golden/human_judge_annotations.csv` (50 sample rows).
2. Read the rubric definitions in `reports/stage13/human_annotation_guide.md`.
3. Score each example across the 4 dimensions (1–5 scale) and set `human_pass` to True/False.
4. Re-run `python -m src.evaluation.llm_judge` to compute Cohen's kappa and MAD.

```text
Anti-Fabrication Policy: Synthetic or simulated human ratings are strictly forbidden.
The repository preserves an unpopulated template until real human consensus is gathered.
```
