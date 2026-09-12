"""
bm25.py
-------
Okapi BM25 retrieval engine implemented in pure Python and NumPy.

Provides:
  - Text normalization and tokenization
  - Incremental corpus indexing
  - Okapi BM25 scoring with k1=1.5, b=0.75
  - Fast top-k nearest neighbor retrieval
"""

from collections import Counter, defaultdict
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Cleans and tokenizes text into lowercase words."""
    cleaned = re.sub(r"https?://\S+|www\.\S+|<URL>", " ", text)
    cleaned = re.sub(r"@\w+", " ", cleaned)
    tokens = re.findall(r"\b[a-z]{2,}\b", cleaned.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Index:
    """Okapi BM25 search index for text retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: List[int] = []
        self.corpus_metadata: List[Dict[str, Any]] = []
        # Inverted index: term -> list of (doc_id, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # IDF table: term -> idf
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Indexes a list of documents with optional metadata dicts.
        """
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        self.doc_lengths = []
        self.corpus_metadata = metadata if metadata is not None else [{} for _ in range(self.corpus_size)]
        self.inverted_index = defaultdict(list)
        doc_freq: Counter = Counter()

        total_length = 0
        for doc_id, doc_text in enumerate(documents):
            tokens = tokenize(doc_text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            counts = Counter(tokens)
            for term, freq in counts.items():
                self.inverted_index[term].append((doc_id, freq))
                doc_freq[term] += 1

        self.avg_doc_len = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Calculate Okapi BM25 IDF: ln((N - n + 0.5) / (n + 0.5) + 1.0)
        self.idf = {}
        for term, df in doc_freq.items():
            val = (self.corpus_size - df + 0.5) / (df + 0.5) + 1.0
            self.idf[term] = math.log(val)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Scores corpus documents for a query and returns top_k matches.
        """
        if self.corpus_size == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[int, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            idf_val = self.idf.get(term, 0.0)
            if idf_val <= 0:
                continue

            for doc_id, freq in self.inverted_index[term]:
                doc_len = self.doc_lengths[doc_id]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf_val * ((freq * (self.k1 + 1.0)) / denom)
                scores[doc_id] += term_score

        if not scores:
            return []

        # Sort top-k by score descending
        sorted_hits = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_hits:
            meta = dict(self.corpus_metadata[doc_id])
            meta["bm25_score"] = float(score)
            meta["doc_id"] = doc_id
            results.append(meta)

        return results
