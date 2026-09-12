"""
metrics.py
----------
STAGE 10: Multi-Dimensional Evaluation Metrics Engine

Calculates:
  1. Classification Metrics (Accuracy, Macro/Weighted F1, Top-k, Confusion Matrix, 95% Bootstrap CI)
  2. Generation & Support Metrics (ROUGE-1/2/L, BLEU-1/2, Guidance Adherence, Required Content)
  3. Policy & Safety Metrics (Policy Safety Rate, Action Claim Rate, Hallucination Rate, Insecure Request Rate)
  4. Escalation Metrics (Overall, Correct, Unnecessary, Missed)
  5. Confidence Calibration (Expected Calibration Error - ECE, Correct vs Incorrect Confidence)
"""

from collections import Counter, defaultdict
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.evaluation.evaluate_baselines import (
    compute_bleu,
    compute_guidance_adherence,
    compute_rouge_l,
    compute_rouge_n,
)
from src.retrieval.bm25 import tokenize


# ---------------------------------------------------------------------------
# 1. Classification Metrics & Bootstrap CI
# ---------------------------------------------------------------------------
def compute_intent_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    candidate_lists: Optional[List[List[str]]] = None,
    k: int = 3,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes accuracy, macro/weighted F1, top-k accuracy, per-intent stats,
    confusion matrix, and 95% bootstrap confidence intervals for accuracy.
    """
    assert len(y_true) == len(y_pred), "Mismatched y_true and y_pred lengths"
    total = len(y_true)
    if total == 0:
        return {}

    all_classes = sorted(list(set(y_true).union(set(y_pred))))
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total

    # Per-intent metrics
    class_stats = {}
    macro_p, macro_r, macro_f1 = [], [], []
    weighted_p, weighted_r, weighted_f1 = [], [], []

    # Confusion matrix: true_class -> {pred_class -> count}
    confusion_matrix: Dict[str, Dict[str, int]] = {
        tc: {pc: 0 for pc in all_classes} for tc in all_classes
    }
    for yt, yp in zip(y_true, y_pred):
        confusion_matrix[yt][yp] += 1

    for c in all_classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        support = sum(1 for yt in y_true if yt == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        class_stats[c] = {
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

        macro_p.append(prec)
        macro_r.append(rec)
        macro_f1.append(f1)

        weight = support / total
        weighted_p.append(prec * weight)
        weighted_r.append(rec * weight)
        weighted_f1.append(f1 * weight)

    # Top-k accuracy
    top_k_acc = None
    if candidate_lists and len(candidate_lists) == total:
        top_k_correct = sum(1 for yt, cands in zip(y_true, candidate_lists) if yt in cands[:k])
        top_k_acc = round(top_k_correct / total, 4)

    # Bootstrap 95% Confidence Interval for Accuracy
    rng = np.random.RandomState(random_seed)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    bootstrap_accs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, total, size=total)
        b_acc = np.mean(y_true_arr[idx] == y_pred_arr[idx])
        bootstrap_accs.append(b_acc)

    ci_lower = round(float(np.percentile(bootstrap_accs, 2.5)), 4)
    ci_upper = round(float(np.percentile(bootstrap_accs, 97.5)), 4)

    return {
        "accuracy": round(accuracy, 4),
        "accuracy_ci_95": [ci_lower, ci_upper],
        "macro_precision": round(float(np.mean(macro_p)), 4),
        "macro_recall": round(float(np.mean(macro_r)), 4),
        "macro_f1": round(float(np.mean(macro_f1)), 4),
        "weighted_f1": round(float(np.sum(weighted_f1)), 4),
        "top_k_accuracy": top_k_acc,
        "per_intent": class_stats,
        "confusion_matrix": confusion_matrix,
    }


# ---------------------------------------------------------------------------
# 2. Generation & Policy Safety Metrics
# ---------------------------------------------------------------------------
def compute_generation_and_safety_metrics(
    hypotheses: List[str],
    references: List[str],
    gold_intents: List[str],
) -> Dict[str, Any]:
    """
    Computes ROUGE-1/2/L, BLEU-1/2, guidance adherence, and granular safety rates.
    """
    assert len(hypotheses) == len(references) == len(gold_intents)
    total = len(hypotheses)
    if total == 0:
        return {}

    r1_list, r2_list, rl_list = [], [], []
    bleu1_list, bleu2_list = [], []
    guidance_list = []

    unsupported_action_count = 0
    hallucinated_guarantee_count = 0
    insecure_request_count = 0
    policy_safe_count = 0

    action_pat = re.compile(r"\b(i\s+have\s+(processed|issued|sent|cancelled|refunded))\b", re.I)
    guarantee_pat = re.compile(r"\b(guaranteed\s+to\s+arrive\s+(today|tomorrow|by\s+\d))\b", re.I)
    insecure_pat = re.compile(r"\b(send\s+your\s+(password|pin|cvv|card\s*number))\b", re.I)

    for hyp, ref, intent in zip(hypotheses, references, gold_intents):
        hyp_toks = tokenize(hyp, remove_stopwords=False)
        ref_toks = tokenize(ref, remove_stopwords=False)

        r1_list.append(compute_rouge_n(hyp_toks, ref_toks, n=1)["f1"])
        r2_list.append(compute_rouge_n(hyp_toks, ref_toks, n=2)["f1"])
        rl_list.append(compute_rouge_l(hyp_toks, ref_toks)["f1"])

        b = compute_bleu(hyp_toks, ref_toks)
        bleu1_list.append(b["bleu_1"])
        bleu2_list.append(b["bleu_2"])

        # Guidance adherence
        guidance_list.append(compute_guidance_adherence(hyp, intent))

        # Safety submetrics
        is_safe = True
        if action_pat.search(hyp):
            unsupported_action_count += 1
            is_safe = False
        if guarantee_pat.search(hyp):
            hallucinated_guarantee_count += 1
            is_safe = False
        if insecure_pat.search(hyp):
            insecure_request_count += 1
            is_safe = False

        if is_safe:
            policy_safe_count += 1

    return {
        "rouge_1": round(float(np.mean(r1_list)), 4),
        "rouge_2": round(float(np.mean(r2_list)), 4),
        "rouge_l": round(float(np.mean(rl_list)), 4),
        "bleu_1": round(float(np.mean(bleu1_list)), 4),
        "bleu_2": round(float(np.mean(bleu2_list)), 4),
        "guidance_adherence_rate": round(float(np.mean(guidance_list)), 4),
        "policy_safety_rate": round(policy_safe_count / total, 4),
        "unsupported_action_rate": round(unsupported_action_count / total, 4),
        "hallucinated_guarantee_rate": round(hallucinated_guarantee_count / total, 4),
        "insecure_request_rate": round(insecure_request_count / total, 4),
    }


# ---------------------------------------------------------------------------
# 3. Escalation Quality Metrics
# ---------------------------------------------------------------------------
def compute_escalation_metrics(
    escalation_flags: List[bool],
    gold_intents: List[str],
    sentiments: List[str],
) -> Dict[str, Any]:
    """
    Evaluates escalation behavior against gold operational criteria.
    True escalation needed: intent == 'service_complaint_escalation' or sentiment == 'angry'.
    """
    assert len(escalation_flags) == len(gold_intents) == len(sentiments)
    total = len(escalation_flags)
    if total == 0:
        return {}

    escalated_count = sum(1 for f in escalation_flags if f)
    escalation_rate = escalated_count / total

    # Identify ground truth escalation candidates
    gold_should_escalate = [
        (intent == "service_complaint_escalation" or sent == "angry")
        for intent, sent in zip(gold_intents, sentiments)
    ]
    total_should_escalate = sum(1 for e in gold_should_escalate if e)

    # Correct escalations (TP)
    tp_esc = sum(1 for pred, should in zip(escalation_flags, gold_should_escalate) if pred and should)
    correct_esc_rate = tp_esc / total_should_escalate if total_should_escalate > 0 else 1.0

    # Unnecessary escalations (FP): escalated when not required
    fp_esc = sum(1 for pred, should in zip(escalation_flags, gold_should_escalate) if pred and not should)
    total_should_not_escalate = total - total_should_escalate
    unnecessary_esc_rate = fp_esc / total_should_not_escalate if total_should_not_escalate > 0 else 0.0

    # Missed escalations (FN): failed to escalate when required
    fn_esc = sum(1 for pred, should in zip(escalation_flags, gold_should_escalate) if not pred and should)
    missed_esc_rate = fn_esc / total_should_escalate if total_should_escalate > 0 else 0.0

    return {
        "overall_escalation_rate": round(escalation_rate, 4),
        "correct_escalation_rate": round(correct_esc_rate, 4),
        "unnecessary_escalation_rate": round(unnecessary_esc_rate, 4),
        "missed_escalation_rate": round(missed_esc_rate, 4),
        "total_escalated": escalated_count,
    }


# ---------------------------------------------------------------------------
# 4. Confidence Calibration (ECE)
# ---------------------------------------------------------------------------
def compute_confidence_calibration(
    y_true: List[str],
    y_pred: List[str],
    confidences: List[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and mean confidence on correct vs incorrect cases.
    """
    assert len(y_true) == len(y_pred) == len(confidences)
    total = len(y_true)
    if total == 0:
        return {}

    corrects = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true, y_pred)]

    correct_confs = [c for c, is_corr in zip(confidences, corrects) if is_corr == 1.0]
    incorrect_confs = [c for c, is_corr in zip(confidences, corrects) if is_corr == 0.0]

    mean_conf_correct = float(np.mean(correct_confs)) if correct_confs else 0.0
    mean_conf_incorrect = float(np.mean(incorrect_confs)) if incorrect_confs else 0.0

    # Calculate Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        # Elements in bin
        in_bin = [
            (c, is_corr)
            for c, is_corr in zip(confidences, corrects)
            if bin_lower <= c < bin_upper or (i == n_bins - 1 and c == bin_upper)
        ]

        if in_bin:
            bin_size = len(in_bin)
            bin_acc = sum(item[1] for item in in_bin) / bin_size
            bin_conf = sum(item[0] for item in in_bin) / bin_size
            ece += (bin_size / total) * abs(bin_acc - bin_conf)

    low_conf_count = sum(1 for c in confidences if c < 0.55)

    return {
        "expected_calibration_error": round(float(ece), 4),
        "mean_confidence_overall": round(float(np.mean(confidences)), 4),
        "mean_confidence_correct": round(mean_conf_correct, 4),
        "mean_confidence_incorrect": round(mean_conf_incorrect, 4),
        "low_confidence_rate": round(low_conf_count / total, 4),
    }
