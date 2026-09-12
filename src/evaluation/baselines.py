"""
baselines.py
------------
STAGE 8: Intent Classification & Support Response Baselines

Defines:
  - Intent Classifiers:
      1. MajorityClassBaseline: Naive statistical floor (predicts delivery_delay).
      2. KeywordRulesBaseline: Deterministic lexical & taxonomy keyword matching.
      3. TFIDFNaiveBayesBaseline: Pure Python/NumPy TF-IDF + Multinomial Naive Bayes.
      4. BM25NearestNeighborBaseline: 1-NN retrieval classifier over training exemplars.
  - Response Generators / Retrievers:
      1. GenericDefaultResponse: Static polite customer service acknowledgment.
      2. IntentCannedTemplateResponse: Policy-grounded canned response templates per intent.
      3. BM25HistoricRetrievalResponse: Top-1 historic human agent response from training corpus.
"""

from collections import Counter, defaultdict
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.retrieval.bm25 import BM25Index, tokenize

# Canonical intent canned response templates
INTENT_CANNED_TEMPLATES: Dict[str, str] = {
    "delivery_delay": (
        "We are sorry your delivery is running late! Please check the latest carrier tracking update "
        "in 'Your Orders' at <URL>. If the expected delivery date has passed, please contact us with your order ID."
    ),
    "missing_delivered_package": (
        "We apologize that your package is marked delivered but not received. Please check safe locations around "
        "your property, with neighbors, or your building mailroom. If it cannot be located, please message us directly."
    ),
    "returns_and_refunds": (
        "You can initiate a return or track your refund status directly at our Online Returns Center: <URL>. "
        "Refunds are typically processed to your original payment method within 3-5 business days of receipt."
    ),
    "order_cancellation": (
        "To request cancellation, please visit 'Your Orders' at <URL> and select 'Cancel Items'. "
        "If the order has already entered dispatch, it cannot be cancelled directly and can be returned upon delivery."
    ),
    "damaged_defective_item": (
        "We are deeply sorry your item arrived damaged or defective. Please visit our Returns Center at <URL> "
        "to arrange a free return or replacement shipment."
    ),
    "prime_membership": (
        "You can manage your Amazon Prime subscription, view renewal dates, or cancel membership benefits "
        "at Manage Your Prime Membership: <URL>. Prorated refunds are available if benefits were not utilized."
    ),
    "payment_and_billing": (
        "For billing discrepancies or payment declines, please check your payment method and order invoice at <URL>. "
        "Please never share your full credit card or sensitive details on public social media."
    ),
    "account_access_security": (
        "If you are having trouble logging in or need password assistance, please use our secure Password Recovery page: <URL>. "
        "Our team will never ask for your account password or two-step verification code."
    ),
    "digital_services_technical": (
        "For Kindle, Fire TV, or Alexa device troubleshooting, please try restarting your device and ensuring your app is updated. "
        "You can find step-by-step guides at Device Support: <URL>."
    ),
    "service_complaint_escalation": (
        "We sincerely apologize for your frustrating customer service experience. We want to make this right. "
        "Please connect with a senior support specialist directly via private message or at <URL>."
    ),
    "unknown_or_ambiguous": (
        "Thank you for contacting Amazon Help. Could you please provide a few more details or your order number "
        "so we can guide you to the right solution?"
    ),
}

GENERIC_DEFAULT_RESPONSE = (
    "Thank you for reaching out to Amazon Customer Service. We are here to help! "
    "Please visit our Help Center at <URL> or share more details securely via direct message."
)


# ---------------------------------------------------------------------------
# 1. Majority Class Intent Baseline
# ---------------------------------------------------------------------------
class MajorityClassBaseline:
    """Predicts the most common training intent (delivery_delay)."""

    def __init__(self, majority_intent: str = "delivery_delay"):
        self.majority_intent = majority_intent

    def fit(self, labels: List[str]) -> None:
        if labels:
            counts = Counter(labels)
            self.majority_intent = counts.most_common(1)[0][0]

    def predict(self, text: str) -> str:
        return self.majority_intent

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.majority_intent for _ in texts]


# ---------------------------------------------------------------------------
# 2. Keyword & Rule-Based Intent Baseline
# ---------------------------------------------------------------------------
class KeywordRulesBaseline:
    """Deterministic keyword & regex rules based on Stage 6 taxonomy criteria."""

    def __init__(self, fallback_intent: str = "unknown_or_ambiguous"):
        self.fallback_intent = fallback_intent

        self.rules: List[Tuple[str, List[str]]] = [
            ("missing_delivered_package", [
                r"\b(says delivered|marked delivered|stated delivered|delivered but|haven't received|not received|never received|stolen|mailbox|porch|driver left|wrong house|missing package)\b"
            ]),
            ("damaged_defective_item", [
                r"\b(damaged|broken|shattered|cracked|defective|scratched|torn box|smashed|faulty|does not work|arrived broken)\b"
            ]),
            ("order_cancellation", [
                r"\b(cancel order|cancel my order|cancellation|cancel item|cancel this|want to cancel|stop delivery|cancel it)\b"
            ]),
            ("prime_membership", [
                r"\b(prime|membership|prime fee|annual fee|charged for prime|prime renewal|prime trial|prime video|prime student)\b"
            ]),
            ("account_access_security", [
                r"\b(password|login|log in|sign in|locked account|account locked|2fa|otp|verification code|reset password|hacked)\b"
            ]),
            ("returns_and_refunds", [
                r"\b(refund|return|send back|drop off|money back|return label|get refund|refund status|haven't got refund|return item)\b"
            ]),
            ("payment_and_billing", [
                r"\b(charged twice|double charge|charged me|gift card|billing|credit card|debit card|declined|payment failed|bank account|invoice)\b"
            ]),
            ("digital_services_technical", [
                r"\b(kindle|fire tv|fire stick|firestick|alexa|echo|app crash|ebook|audible|prime music|streaming)\b"
            ]),
            ("delivery_delay", [
                r"\b(late|delay|delayed|tracking|track|carrier|transit|where is my|expected delivery|not arrived|hasn't arrived|due date|shipping date)\b"
            ]),
            ("service_complaint_escalation", [
                r"\b(supervisor|manager|worst service|terrible service|pathetic service|speak to someone|call me|useless support|unacceptable|escalate|lawyer|sue)\b"
            ]),
        ]

    def predict(self, text: str) -> str:
        lower = text.lower()
        for intent_id, patterns in self.rules:
            for pat in patterns:
                if re.search(pat, lower):
                    return intent_id
        return self.fallback_intent

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict(t) for t in texts]


# ---------------------------------------------------------------------------
# 3. TF-IDF + Naive Bayes Intent Baseline (Pure Python / NumPy)
# ---------------------------------------------------------------------------
class TFIDFNaiveBayesBaseline:
    """Multinomial Naive Bayes on unigram+bigram TF-IDF representations."""

    def __init__(self, alpha: float = 1.0, min_df: int = 2):
        self.alpha = alpha
        self.min_df = min_df
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.log_priors: np.ndarray = np.array([])
        self.feature_log_prob: np.ndarray = np.array([])

    def _extract_terms(self, text: str) -> List[str]:
        words = tokenize(text)
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def fit(self, texts: List[str], labels: List[str]) -> None:
        assert len(texts) == len(labels), "Mismatched texts and labels"
        n_samples = len(texts)
        if n_samples == 0:
            return

        self.classes = sorted(list(set(labels)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        n_classes = len(self.classes)

        # 1. Build vocabulary with Document Frequency >= min_df
        df_counts: Counter = Counter()
        tokenized_docs: List[List[str]] = []
        for text in texts:
            terms = self._extract_terms(text)
            tokenized_docs.append(terms)
            for term in set(terms):
                df_counts[term] += 1

        self.vocab = {}
        for term, df in df_counts.items():
            if df >= self.min_df:
                self.vocab[term] = len(self.vocab)

        n_vocab = len(self.vocab)
        if n_vocab == 0:
            return

        # 2. Compute IDF: log((N + 1) / (df + 1)) + 1
        self.idf = np.zeros(n_vocab, dtype=np.float64)
        for term, idx in self.vocab.items():
            df = df_counts[term]
            self.idf[idx] = math.log((n_samples + 1.0) / (df + 1.0)) + 1.0

        # 3. Accumulate TF-IDF term weights per class
        class_term_weights = np.zeros((n_classes, n_vocab), dtype=np.float64)
        class_doc_counts = np.zeros(n_classes, dtype=np.float64)

        for terms, label in zip(tokenized_docs, labels):
            c_idx = self.class_to_idx[label]
            class_doc_counts[c_idx] += 1

            term_counts = Counter(terms)
            for term, count in term_counts.items():
                if term in self.vocab:
                    v_idx = self.vocab[term]
                    # Sublinear TF scaling: (1 + log(tf)) * idf
                    tf = 1.0 + math.log(count)
                    weight = tf * self.idf[v_idx]
                    class_term_weights[c_idx, v_idx] += weight

        # 4. Compute Log Priors with smoothing
        self.log_priors = np.log((class_doc_counts + 1.0) / (n_samples + n_classes))

        # 5. Compute Feature Log Probabilities with Laplace smoothing
        total_class_weights = class_term_weights.sum(axis=1, keepdims=True)
        smoothed_weights = class_term_weights + self.alpha
        smoothed_totals = total_class_weights + self.alpha * n_vocab
        self.feature_log_prob = np.log(smoothed_weights / smoothed_totals)

    def predict(self, text: str) -> str:
        if len(self.classes) == 0:
            return "delivery_delay"

        terms = self._extract_terms(text)
        term_counts = Counter(terms)

        # Compute query vector
        term_indices = []
        term_weights = []
        for term, count in term_counts.items():
            if term in self.vocab:
                v_idx = self.vocab[term]
                tf = 1.0 + math.log(count)
                weight = tf * self.idf[v_idx]
                term_indices.append(v_idx)
                term_weights.append(weight)

        if not term_indices:
            # Fallback to class prior
            return self.classes[int(np.argmax(self.log_priors))]

        q_vec = np.zeros(len(self.vocab), dtype=np.float64)
        for idx, w in zip(term_indices, term_weights):
            q_vec[idx] = w

        # Score per class: log_prior + dot(q_vec, feature_log_prob)
        scores = self.log_priors + self.feature_log_prob.dot(q_vec)
        best_class_idx = int(np.argmax(scores))
        return self.classes[best_class_idx]

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict(t) for t in texts]


# ---------------------------------------------------------------------------
# 4. BM25 Nearest Neighbor Intent Baseline
# ---------------------------------------------------------------------------
class BM25NearestNeighborBaseline:
    """Classifies query into the intent of the top-1 BM25 retrieved training interaction."""

    def __init__(self, bm25_index: BM25Index, fallback_intent: str = "delivery_delay"):
        self.bm25_index = bm25_index
        self.fallback_intent = fallback_intent

    def predict(self, text: str) -> str:
        hits = self.bm25_index.search(text, top_k=1)
        if hits and hits[0].get("bm25_score", 0.0) > 0.0:
            return hits[0].get("intent_id", self.fallback_intent)
        return self.fallback_intent

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict(t) for t in texts]


# ---------------------------------------------------------------------------
# 5. Response Generation / Retrieval Baselines
# ---------------------------------------------------------------------------
class GenericDefaultResponse:
    """Emits static standard acknowledgment template."""

    def generate(self, customer_query: str, predicted_intent: Optional[str] = None) -> str:
        return GENERIC_DEFAULT_RESPONSE


class IntentCannedTemplateResponse:
    """Emits policy-grounded canned response mapped to predicted intent."""

    def generate(self, customer_query: str, predicted_intent: Optional[str] = None) -> str:
        intent_id = predicted_intent or "unknown_or_ambiguous"
        return INTENT_CANNED_TEMPLATES.get(intent_id, INTENT_CANNED_TEMPLATES["unknown_or_ambiguous"])


class BM25HistoricRetrievalResponse:
    """Retrieves the historic human agent response from the top BM25 matched training interaction."""

    def __init__(self, bm25_index: BM25Index, fallback_response: str = GENERIC_DEFAULT_RESPONSE):
        self.bm25_index = bm25_index
        self.fallback_response = fallback_response

    def generate(self, customer_query: str, predicted_intent: Optional[str] = None) -> str:
        hits = self.bm25_index.search(customer_query, top_k=1)
        if hits and hits[0].get("bm25_score", 0.0) > 0.0:
            return hits[0].get("support_response", self.fallback_response)
        return self.fallback_response
