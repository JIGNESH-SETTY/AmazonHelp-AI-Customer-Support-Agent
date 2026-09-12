"""
retrieval.py
------------
STAGE 9: Historical Support Knowledge Retrieval

Integrates:
  - BM25 retrieval over training split support corpus
  - Cleans and sanitizes historical agent responses (removes dead links, Twitter handles, agent initials)
  - Treats retrieved data as untrusted context/evidence rather than authoritative policy
"""

import re
from typing import Any, Dict, List, Optional

from src.agent.schemas import RetrievalItem
from src.retrieval.bm25 import BM25Index


def sanitize_historical_response(response_text: str) -> str:
    """
    Cleans Twitter handles, agent signatures, and dead links from historical tweets.
    """
    # Remove @mentions
    cleaned = re.sub(r"@\w+", "", response_text)
    # Remove agent signatures (e.g. ^JZ, ^SK, 1/2)
    cleaned = re.sub(r"\^[A-Z]{2,3}\b", "", cleaned)
    cleaned = re.sub(r"\b\d/\d\b", "", cleaned)
    # Standardize URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", "<URL>", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class KnowledgeRetriever:
    """Retrieves relevant historical conversations from BM25 index."""

    def __init__(self, bm25_index: Optional[BM25Index] = None, top_k: int = 3):
        self.bm25_index = bm25_index
        self.top_k = top_k

    def set_index(self, bm25_index: BM25Index) -> None:
        self.bm25_index = bm25_index

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalItem]:
        """
        Retrieves top-k sanitized historical support interactions.
        """
        k = top_k if top_k is not None else self.top_k
        if not self.bm25_index or self.bm25_index.corpus_size == 0:
            return []

        hits = self.bm25_index.search(query, top_k=k)
        retrieval_items: List[RetrievalItem] = []

        for hit in hits:
            raw_resp = hit.get("support_response", "")
            clean_resp = sanitize_historical_response(raw_resp)
            score = hit.get("bm25_score", 0.0)

            # Filter out generic deflection noise
            if re.search(r"^\s*(dm\s+sent|check\s+dm|replied)\b", clean_resp, re.I):
                continue

            retrieval_items.append(
                RetrievalItem(
                    example_id=hit.get("example_id", "hist_unknown"),
                    relevance_score=round(score, 3),
                    customer_query=hit.get("customer_message", ""),
                    historical_response=clean_resp,
                    source_dataset="amazonhelp_historical_evidence",
                )
            )

        return retrieval_items
