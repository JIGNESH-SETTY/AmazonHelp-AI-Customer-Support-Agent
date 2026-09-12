"""
intent.py
---------
STAGE 9: Hybrid Intent Understanding & Multi-Intent Classifier

Combines:
  1. Calibrated TF-IDF Naive Bayes log-posteriors
  2. Taxonomy-aligned discriminative keyword boost rules
  3. Multi-intent detection (primary vs secondary intents)
  4. Normalized confidence calibration and top-k candidate ranking
"""

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

import numpy as np

from src.agent.schemas import CandidateIntent, IntentResult
from src.evaluation.baselines import KeywordRulesBaseline, TFIDFNaiveBayesBaseline
from src.retrieval.bm25 import tokenize

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TAXONOMY_PATH = REPO_ROOT / "data" / "intent_taxonomy.json"


def load_taxonomy_names() -> Dict[str, str]:
    """Loads intent ID to human-readable name mapping."""
    with open(TAXONOMY_PATH, encoding="utf-8") as fh:
        tax = json.load(fh)
    names = {i["intent_id"]: i["name"] for i in tax["intents"]}
    if "fallback_category" in tax:
        names[tax["fallback_category"]["intent_id"]] = tax["fallback_category"]["name"]
    names["unknown_or_ambiguous"] = "Unknown or Ambiguous"
    return names


class IntentClassifier(Protocol):
    """Protocol for intent classification strategies."""

    def predict(self, text: str) -> IntentResult:
        ...


class HybridIntentClassifier:
    """
    Production-grade hybrid intent classifier combining probabilistic TF-IDF
    features with deterministic domain keyword boosts and multi-intent detection.
    """

    def __init__(self, tfidf_nb_model: Optional[TFIDFNaiveBayesBaseline] = None):
        self.tfidf_nb = tfidf_nb_model
        self.intent_names = load_taxonomy_names()
        self.keyword_rules = KeywordRulesBaseline()

        # High-precision discriminative keyword triggers: (intent_id, weight_boost, patterns)
        self.discriminative_boosts: List[Tuple[str, float, List[str]]] = [
            (
                "missing_delivered_package",
                4.5,
                [
                    r"\b(says delivered|marked delivered|marked as delivered|stated delivered|delivered but|never arrived|stolen|mailbox|porch|doorstep|left outside|package not received but delivered)\b"
                ],
            ),
            (
                "order_cancellation",
                4.0,
                [
                    r"\b(cancel order|cancel my order|cancellation|cancel item|want to cancel|stop order|cancel this|cancel.*membership|cancelling.*membership|cancel.*subscription|cancelling.*subscription|cancel.*prime|cancelling.*prime|cancel my prime)\b"
                ],
            ),
            (
                "returns_and_refunds",
                3.5,
                [
                    r"\b(refund|return|send back|drop off|money back|return label|refund status|get refund|refunded)\b"
                ],
            ),
            (
                "damaged_defective_item",
                4.0,
                [
                    r"\b(damaged|broken|shattered|cracked|defective|scratched|smashed|faulty|does not work|arrived broken)\b"
                ],
            ),
            (
                "prime_membership",
                3.5,
                [
                    r"\b(prime|membership|prime fee|annual fee|charged for prime|prime renewal|prime trial|prime video)\b"
                ],
            ),
            (
                "account_access_security",
                4.0,
                [
                    r"\b(password|login|log in|sign in|locked account|2fa|otp|verification code|reset password|hacked)\b"
                ],
            ),
            (
                "payment_and_billing",
                3.5,
                [
                    r"\b(charged twice|double charge|charged me|gift card|billing|credit card|declined|payment failed|bank account|invoice)\b"
                ],
            ),
            (
                "digital_services_technical",
                3.5,
                [
                    r"\b(kindle|fire tv|fire stick|firestick|alexa|echo|app crash|ebook|audible|streaming)\b"
                ],
            ),
            (
                "service_complaint_escalation",
                3.5,
                [
                    r"\b(supervisor|manager|worst service|terrible service|pathetic service|speak to someone|call me|useless support|unacceptable|escalate|lawyer|sue)\b"
                ],
            ),
            (
                "delivery_delay",
                2.5,
                [
                    r"\b(late|delay|delayed|tracking|track|carrier|transit|where is my|expected delivery|not arrived|hasn't arrived|due date|shipping date)\b"
                ],
            ),
        ]

    def set_tfidf_model(self, model: TFIDFNaiveBayesBaseline) -> None:
        """Sets the underlying trained TF-IDF Naive Bayes model."""
        self.tfidf_nb = model

    def predict(self, text: str) -> IntentResult:
        """
        Predicts primary intent, secondary intents, calibrated confidence,
        and top candidate ranking.
        """
        lower = text.lower()
        words = set(re.findall(r"\b[a-z]{2,}\b", lower))

        # Check for very brief or ambiguous text
        if len(text.strip()) < 15 and not any(w in words for w in ["cancel", "refund", "broken", "stolen", "late"]):
            return IntentResult(
                primary_intent="unknown_or_ambiguous",
                primary_intent_name=self.intent_names.get("unknown_or_ambiguous", "Unknown or Ambiguous"),
                intent_confidence=0.40,
                secondary_intents=[],
                top_candidates=[
                    CandidateIntent("unknown_or_ambiguous", "Unknown or Ambiguous", 0.40),
                    CandidateIntent("delivery_delay", "Delivery Delay & Tracking", 0.30),
                ],
            )

        # 1. Base log-posteriors from TF-IDF Naive Bayes
        scores: Dict[str, float] = {}
        if self.tfidf_nb and len(self.tfidf_nb.classes) > 0 and len(self.tfidf_nb.vocab) > 0:
            terms = self.tfidf_nb._extract_terms(text)
            term_counts = Counter(terms)
            q_vec = np.zeros(len(self.tfidf_nb.vocab), dtype=np.float64)
            for term, count in term_counts.items():
                if term in self.tfidf_nb.vocab:
                    v_idx = self.tfidf_nb.vocab[term]
                    tf = 1.0 + math.log(count)
                    q_vec[v_idx] = tf * self.tfidf_nb.idf[v_idx]

            raw_log_probs = self.tfidf_nb.log_priors + self.tfidf_nb.feature_log_prob.dot(q_vec)
            if len(raw_log_probs) > 0:
                # Softmax normalization over TF-IDF scores
                max_log = np.max(raw_log_probs)
                exp_probs = np.exp(raw_log_probs - max_log)
                prob_dist = exp_probs / np.sum(exp_probs)

                for c, p in zip(self.tfidf_nb.classes, prob_dist):
                    scores[c] = float(p)
            else:
                for c in self.tfidf_nb.classes:
                    scores[c] = 1.0 / len(self.tfidf_nb.classes)
        else:
            # Fallback uniform prior across canonical intents
            all_intents = list(self.intent_names.keys())
            uniform_p = 1.0 / len(all_intents)
            for c in all_intents:
                scores[c] = uniform_p

        # 2. Discriminative keyword boosts & secondary intent tracking
        matched_intents: List[str] = []
        for intent_id, boost, regexes in self.discriminative_boosts:
            for rgx in regexes:
                if re.search(rgx, lower):
                    scores[intent_id] = scores.get(intent_id, 0.0) + boost
                    matched_intents.append(intent_id)
                    break

        # Normalize final scores to [0, 1]
        total_score = sum(scores.values())
        norm_scores = {k: v / total_score for k, v in scores.items()} if total_score > 0 else scores

        # Sort candidate intents descending
        ranked = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [
            CandidateIntent(
                intent_id=k,
                intent_name=self.intent_names.get(k, k),
                score=round(v, 4),
            )
            for k, v in ranked[:4]
        ]

        best_intent, best_score = ranked[0]

        # Multi-intent extraction: check if second highest is above margin
        secondary_intents: List[str] = []
        for cand_id in matched_intents:
            if cand_id != best_intent and cand_id not in secondary_intents:
                secondary_intents.append(cand_id)

        # Confidence calibration
        confidence = min(0.95, max(0.40, best_score * 1.25))

        return IntentResult(
            primary_intent=best_intent,
            primary_intent_name=self.intent_names.get(best_intent, best_intent),
            intent_confidence=round(confidence, 4),
            secondary_intents=secondary_intents[:2],
            top_candidates=top_candidates,
        )
