# Stage 13: LLM-as-Judge Reply Quality Evaluation

- **Evaluation Provider**: `unavailable`
- **Evaluation Status**: `unavailable`
- **Model Configured**: `openai/gpt-oss-20b`
- **Evaluation Sample**: 50 stratified examples (Seed 42)
- **Evaluation Date**: 2026-09-12

---

## 1. Rubric Architecture

The LLM-as-judge assesses generated customer responses across 4 operational dimensions (1–5 scale):

| Dimension | Evaluation Criteria | Operational Meaning |
| :--- | :--- | :--- |
| **Helpfulness** | Does response meaningfully resolve/progress the issue? | Prevents empty deflections; ensures customer problem is directly acknowledged. |
| **Grounding / Safety** | Does response adhere to verified policy without inventing actions? | Hard anti-hallucination metric; blocks unauthorized refunds, fake dates, or PII requests. |
| **Actionability** | Does response give concrete, valid next steps? | Directs customer to authenticated portal, self-service tracking, or secure DM. |
| **Clarity / Conciseness** | Is the tone direct, professional, and clear? | Customer service brevity standard; avoids confusing technical jargon or verbose filler. |

---

## 2. Benchmark Sample Stratification

Selected 50 representative dialogues from the Stage 7 Golden Evaluation Set ($N=200$):

- **Difficulty Distribution**: {'hard': 31, 'medium': 13, 'easy': 6}
- **Response Type Distribution**: {'escalation': 27, 'resolution': 15, 'clarification': 8}
- **Intent Distribution**: 11 unique intent classes represented (4–5 examples each).

---

## 3. Judge Evaluation Results

> [!NOTE]
> **Offline / Keyless Environment**: No live GROQ_API_KEY (or OPENAI_API_KEY) was detected.
> The test suite and evaluation runner operate 100% offline without crashing.
> To execute live model evaluations with Groq, configure `GROQ_API_KEY` in environment or `.env` and run:
> ```bash
> python -m src.evaluation.llm_judge --live
> ```

---

## 4. Human-in-the-Loop & Anti-Fabrication Safeguards

In strict compliance with evaluation integrity principles:
1. **No Fake Human Ratings**: `data/golden/human_judge_annotations.csv` contains empty rating fields awaiting manual human consensus.
2. **No Fake LLM Scores**: Heuristic or rule scores are never labeled as LLM-generated.
3. **Transparent Status**: Provider status explicitly identifies whether evaluations are `live_llm`, `cached_llm`, or `unavailable`.

