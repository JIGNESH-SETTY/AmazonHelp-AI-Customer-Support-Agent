"""
discover_intents.py
-------------------
STAGE 6: Intent Discovery Pipeline

Discovers customer-support themes empirically from the actual AmazonHelp
conversations using TF-IDF feature extraction, frequency analysis, and K-Means
clustering on a stratified sample of English customer queries from the training split.

Outputs:
  - `reports/stage6_discovery.json`: Machine-readable discovery artifact
    detailing candidate clusters, top keywords, representative dialogues,
    and discovery metrics.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# Configure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "processed" / "splits"
REPORTS_DIR = REPO_ROOT / "reports"
SELECTED_BRAND_PATH = REPO_ROOT / "data" / "selected_brand.json"
DISCOVERY_REPORT_PATH = REPORTS_DIR / "stage6_discovery.json"

RANDOM_SEED = 42
SAMPLE_SIZE = 15_000
N_CLUSTERS = 14

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn",
    "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each",
    "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "isn", "it", "its", "itself", "just",
    "ll", "m", "me", "more", "most", "my", "myself", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "re", "s", "same", "shan", "she", "should", "shouldn",
    "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "we",
    "were", "weren", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "won", "would", "wouldn", "y", "you", "your", "yours",
    "yourself", "yourselves", "amazon", "amazonhelp", "help", "please", "thanks",
    "thank", "hello", "hey", "hi", "get", "got", "one", "two", "see", "say", "said",
    "tell", "told", "also", "even", "still", "always", "never", "really", "much",
    "well", "way", "via", "amp", "url", "make", "made", "let", "know", "back",
    "tried", "trying", "times", "time", "day", "days", "week", "today", "yesterday",
    "tomorrow", "first", "last", "new", "good", "bad", "great"
}


def load_selected_brand() -> str:
    """Loads selected brand from data/selected_brand.json (single source of truth)."""
    if not SELECTED_BRAND_PATH.exists():
        raise FileNotFoundError(f"Missing {SELECTED_BRAND_PATH}")
    with open(SELECTED_BRAND_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["selected_brand"]


def clean_text_for_discovery(text: str) -> str:
    """Conservative cleaning preserving domain keywords while removing noise."""
    if not text:
        return ""
    # Normalize unicode apostrophes/quotes
    t = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # Remove URLs and @mentions
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    # Lowercase & normalize spaces
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def run_intent_discovery() -> Dict[str, Any]:
    """Executes the unsupervised theme discovery pipeline on training data."""
    t_start = time.time()
    brand_name = load_selected_brand()
    print("=" * 75)
    print(f"STAGE 6: Intent Discovery for '{brand_name}'")
    print("=" * 75)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load candidate population from Stage 4 train split
    train_dir = SPLITS_DIR / "train"
    source_files = {
        "resolution": train_dir / "amazon_resolution_pairs_train.jsonl",
        "escalation": train_dir / "amazon_escalation_pairs_train.jsonl",
        "clarification": train_dir / "amazon_clarification_pairs_train.jsonl",
    }

    population_by_type: Dict[str, List[Dict[str, Any]]] = {}
    total_population = 0

    print("Loading candidate customer messages from Stage 4 train split...")
    for resp_type, path in source_files.items():
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    # Exclude empty or very short messages (< 10 chars)
                    if len(rec.get("customer_message", "").strip()) >= 10:
                        records.append(rec)
        population_by_type[resp_type] = records
        total_population += len(records)
        print(f"  {resp_type:<15}: {len(records):,} candidate interactions")

    # 2. Stratified sampling across response types
    print(f"\nDrawing stratified sample of {SAMPLE_SIZE:,} interactions (seed={RANDOM_SEED})...")
    sampled_records: List[Dict[str, Any]] = []
    # Allocation: 5,000 from resolution, 5,000 from escalation, 5,000 from clarification
    per_type_sample = SAMPLE_SIZE // 3

    for resp_type, records in population_by_type.items():
        k_sample = min(per_type_sample, len(records))
        chosen = random.sample(records, k_sample)
        sampled_records.extend(chosen)

    print(f"  Total sampled: {len(sampled_records):,} customer messages")

    # 3. Tokenization and N-Gram extraction
    print("\nTokenizing and building TF-IDF vocabulary...")
    doc_tokens: List[List[str]] = []
    doc_raw_texts: List[str] = []
    doc_metadata: List[Dict[str, Any]] = []
    df_counts: Counter = Counter()

    for rec in sampled_records:
        raw_text = rec["customer_message"]
        cleaned = clean_text_for_discovery(raw_text)
        words = [w for w in re.findall(r"[a-z]{3,}", cleaned) if w not in STOPWORDS]

        # Extract unigrams and bigrams
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        all_terms = words + bigrams

        doc_tokens.append(all_terms)
        doc_raw_texts.append(raw_text)
        doc_metadata.append({
            "example_id": rec["example_id"],
            "conversation_id": rec["conversation_id"],
            "response_type": rec["response_type"],
            "text": raw_text,
        })
        df_counts.update(set(all_terms))

    N = len(doc_tokens)
    # Filter vocabulary: term must appear in at least 15 documents and at most 30% of documents
    vocab = {term: idx for idx, (term, count) in enumerate(df_counts.most_common(3_500)) if 15 <= count <= 0.30 * N}
    inv_vocab = {idx: term for term, idx in vocab.items()}
    vocab_size = len(vocab)
    print(f"  Vocabulary size: {vocab_size:,} distinct unigrams and bigrams")

    # 4. Construct Sparse TF-IDF Matrix
    print("Constructing normalized TF-IDF matrix...")
    X = np.zeros((N, vocab_size), dtype=np.float32)
    idf = {term: math.log((N + 1) / (df_counts[term] + 1)) + 1.0 for term in vocab}

    for i, terms in enumerate(doc_tokens):
        term_counts = Counter(terms)
        for term, cnt in term_counts.items():
            if term in vocab:
                col = vocab[term]
                X[i, col] = (1.0 + math.log(cnt)) * idf[term]

    # L2 normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    # 5. K-Means Clustering (Deterministic)
    print(f"Running K-Means clustering (k={N_CLUSTERS}, max_iter=25, seed={RANDOM_SEED})...")
    init_indices = np.random.choice(N, N_CLUSTERS, replace=False)
    centroids = X[init_indices].copy()

    clusters = np.zeros(N, dtype=np.int32)
    for iteration in range(25):
        # Cosine similarity is dot product of L2 normalized vectors
        dots = np.dot(X, centroids.T)
        clusters = np.argmax(dots, axis=1)

        # Recompute centroids
        for c in range(N_CLUSTERS):
            members = X[clusters == c]
            if len(members) > 0:
                c_mean = np.mean(members, axis=0)
                c_norm = np.linalg.norm(c_mean)
                centroids[c] = c_mean / (c_norm if c_norm > 0 else 1.0)

    # 6. Analyze Discovered Candidate Themes
    print("\nAnalyzing candidate discovered themes...")
    cluster_counts = Counter(clusters)
    discovered_themes = []

    for c in range(N_CLUSTERS):
        member_indices = np.where(clusters == c)[0]
        n_members = len(member_indices)
        if n_members == 0:
            continue

        # Top TF-IDF keywords for centroid
        top_col_indices = np.argsort(centroids[c])[::-1][:15]
        top_keywords = [inv_vocab[idx] for idx in top_col_indices]

        # Representative dialogue samples closest to centroid
        member_sims = np.dot(X[member_indices], centroids[c])
        best_local_indices = np.argsort(member_sims)[::-1][:5]
        representative_samples = [
            {
                "conversation_id": doc_metadata[member_indices[idx]]["conversation_id"],
                "example_id": doc_metadata[member_indices[idx]]["example_id"],
                "response_type": doc_metadata[member_indices[idx]]["response_type"],
                "customer_text": doc_metadata[member_indices[idx]]["text"],
                "similarity": round(float(member_sims[idx]), 4),
            }
            for idx in best_local_indices
        ]

        # Response type distribution in this theme
        theme_resp_types = Counter(doc_metadata[idx]["response_type"] for idx in member_indices)

        theme_payload = {
            "cluster_id": c,
            "sample_count": n_members,
            "sample_percentage": round((n_members / N) * 100, 2),
            "top_keywords": top_keywords,
            "response_type_breakdown": dict(theme_resp_types),
            "representative_samples": representative_samples,
        }
        discovered_themes.append(theme_payload)

    # Sort themes by size descending
    discovered_themes.sort(key=lambda x: -x["sample_count"])

    elapsed = time.time() - t_start
    print(f"\nDiscovered {len(discovered_themes)} candidate themes in {elapsed:.2f}s.")

    # 7. Write Discovery Artifact
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    discovery_artifact = {
        "stage": 6,
        "selected_brand": brand_name,
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "population_statistics": {
            "total_available_train_interactions": total_population,
            "by_response_type": {k: len(v) for k, v in population_by_type.items()},
        },
        "sampling_methodology": {
            "sample_size": N,
            "random_seed": RANDOM_SEED,
            "sampling_strategy": "Stratified sampling across resolution, escalation, and clarification interactions in Stage 4 train split",
            "per_response_type_sample": per_type_sample,
        },
        "feature_extraction": {
            "vocabulary_size": vocab_size,
            "min_doc_freq": 15,
            "max_doc_freq_ratio": 0.30,
            "ngram_range": [1, 2],
        },
        "clustering": {
            "algorithm": "K-Means (Cosine Metric via L2-normalized TF-IDF)",
            "n_clusters": N_CLUSTERS,
            "max_iterations": 25,
        },
        "candidate_themes": discovered_themes,
        "execution_time_seconds": round(elapsed, 2),
    }

    with open(DISCOVERY_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(discovery_artifact, fh, indent=2, ensure_ascii=False)

    print(f"✓ Discovery report saved to: {DISCOVERY_REPORT_PATH.relative_to(REPO_ROOT)}")
    return discovery_artifact


if __name__ == "__main__":
    run_intent_discovery()
