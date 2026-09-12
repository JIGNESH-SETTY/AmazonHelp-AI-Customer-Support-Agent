"""
profile_brands.py
-----------------
STAGE 5: Brand Selection & Dataset Profiling

Discovers and quantitatively profiles all customer-support brands represented
in the Twitter Customer Support dataset (`data/raw/twcs.csv`).

Key Features:
  - Memory-efficient streaming in chunks (handles 2.8M rows without loading all into RAM).
  - Graph-based conversation reconstruction using Union-Find on tweet_id and response links.
  - Conversation-level analysis: conversation_id is the primary unit of aggregation.
  - Guarantees: Every discovered conversation belongs to exactly one brand (Check 1).
  - Support behavior classification consistent with Stages 2 & 3 (resolution, escalation,
    clarification, acknowledgement, uncertain).
  - Language inference consistent with Stages 2 & 3 (English, Japanese, German, Spanish, French, etc.).
  - Intent diversity proxy: Normalized Shannon entropy across domain topics combined with
    lexical vocabulary richness.
"""

from collections import Counter, defaultdict
import math
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Configure UTF-8 stdout on Windows
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths and Defaults
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "twcs.csv"
DEFAULT_CHUNK_SIZE = 250_000

# ---------------------------------------------------------------------------
# Regexes: Language & Classification (aligned with Stages 2 & 3)
# ---------------------------------------------------------------------------
RE_JA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
RE_DE = re.compile(
    r"\b(wir|haben|danke|bitte|hilfe|kontakt|können|für|nicht|hallo|schönen|gruß|oder|deine|dein|einen|eine|einem|einer|unsere|unser)\b",
    re.I,
)
RE_ES = re.compile(
    r"\b(hola|gracias|podemos|ayudar|por favor|pedido|cuenta|envío|disculpa|para|con|sobre|está|nuestro|nuestra|servicio|puedes|ayuda)\b",
    re.I,
)
RE_FR = re.compile(
    r"\b(bonjour|merci|votre|nous|pour|commande|livraison|compte|pouvez|contacter|cordialement|désolé|avec|vous|notre|cette|dans)\b",
    re.I,
)
RE_EN = re.compile(
    r"\b(the|to|you|we|for|is|in|and|our|please|sorry|help|dm|order|link|have|with|your|can|not|this|my|it|on|at|me)\b",
    re.I,
)

RE_SUBSTANTIVE = re.compile(
    r"\b(restart|reboot|reinstall|troubleshoot|settings|steps|refund|return|replacement|estimated delivery|carrier|delivered|package|policy|warranty|update your|clear cache|cancel your|charge|billing|force stop|instructions|dispatch|tracking|transit|re-order|prime membership|download|app store|browser|account settings|sign out|sign in|unplug|plug|wifi|connection|firmware|try to|you can (?:check|find|see)|order status|delivery date|days for the refund|business days|booked|flight|ticket|boarding|gate|seat|reservation|baggage|luggage)\b",
    re.I,
)
RE_DEFLECT = re.compile(
    r"\b(dm us|direct message|reach (?:us|out) (?:via|by|at|here)|contact (?:us|our support|our team|customer service)|call us|phone or chat|use this link|via this link|at this link|get in touch (?:with us|here)|fill out (?:this|the) form|support page|unable to (?:access|affect|view|check)|security (?:reasons|measures)|personal info(?:rmation)?)\b|https?://",
    re.I,
)
RE_CLARIFY = re.compile(
    r"\b(could you (?:provide|clarify|confirm|tell)|can you (?:provide|clarify|confirm|tell)|what (?:device|error|happens|order|country|item|marketplace)|which (?:device|marketplace|item|app)|mind (?:providing|letting|sharing)|are you (?:seeing|using|able|trying)|more details|describe the issue|share with us|any error)\b|\?",
    re.I,
)
RE_ACK = re.compile(
    r"\b(you(\')?re welcome|happy to help|glad to help|anytime|have a (?:great|nice|wonderful|good) (?:day|evening|weekend|night|rest)|thank you|thanks for (?:reaching|contacting|letting|providing|sharing)|our pleasure|hope this helps|wir haben zu danken|de rien|un placer)\b",
    re.I,
)

# Domain Topic Patterns for Diversity Proxy
TOPIC_PATTERNS = {
    "orders": re.compile(r"\b(order|orders|ordered|ordering)\b", re.I),
    "delivery": re.compile(r"\b(deliver|delivering|delivered|delivery|deliveries)\b", re.I),
    "refunds": re.compile(r"\b(refund|refunds|refunded|refunding)\b", re.I),
    "returns": re.compile(r"\b(return|returns|returned|returning|exchange)\b", re.I),
    "payments": re.compile(r"\b(pay|payment|payments|paid|charge|charged|card|billing|invoice|fee)\b", re.I),
    "account": re.compile(r"\b(account|login|password|sign in|locked|username|profile)\b", re.I),
    "technical": re.compile(r"\b(bug|glitch|crash|error|update|app|website|loading|broken|down|server)\b", re.I),
    "hardware_device": re.compile(r"\b(device|phone|tv|screen|battery|kindle|iphone|ipad|console|controller)\b", re.I),
    "subscriptions": re.compile(r"\b(subscription|subscriptions|subscribe|membership|prime|renew|cancel)\b", re.I),
    "shipping_transit": re.compile(r"\b(ship|shipping|shipped|carrier|ups|fedex|usps|tracking|courier|transit|dispatch)\b", re.I),
    "booking_travel": re.compile(r"\b(flight|ticket|airport|plane|delay|seat|baggage|luggage|reservation|train)\b", re.I),
    "policies_general": re.compile(r"\b(policy|warranty|terms|guarantee|claim|legal|protection)\b", re.I),
}

STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "me", "more", "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you",
    "your", "yours"
}


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------
class UnionFind:
    """Disjoint Set Union (Union-Find) with path compression."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        path = []
        while x in self.parent and self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        for node in path:
            self.parent[node] = x
        return x

    def union(self, x: int, y: int) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[rx] = ry


# ---------------------------------------------------------------------------
# Language & Response Helpers
# ---------------------------------------------------------------------------
def infer_language(text: str) -> str:
    """Transparent heuristic language classifier consistent with Stages 2 & 3."""
    if not text or pd.isna(text):
        return "Unknown"
    if RE_JA.search(text):
        return "Japanese"
    en_m = len(RE_EN.findall(text))
    de_m = len(RE_DE.findall(text))
    es_m = len(RE_ES.findall(text))
    fr_m = len(RE_FR.findall(text))
    max_m = max(en_m, de_m, es_m, fr_m)
    if max_m == 0:
        return "Unknown"
    if en_m == max_m and en_m > de_m and en_m > es_m and en_m > fr_m:
        return "English"
    if de_m == max_m and de_m > en_m and de_m > es_m and de_m > fr_m:
        return "German"
    if es_m == max_m and es_m > en_m and es_m > de_m and es_m > fr_m:
        return "Spanish"
    if fr_m == max_m and fr_m > en_m and fr_m > de_m and fr_m > es_m:
        return "French"
    return "Other"


def classify_response(text: str) -> str:
    """Rule-based support response classifier consistent with Stages 2 & 3."""
    if not text or pd.isna(text):
        return "uncertain"
    has_sub = bool(RE_SUBSTANTIVE.search(text))
    has_def = bool(RE_DEFLECT.search(text))
    has_cla = bool(RE_CLARIFY.search(text))
    has_ack = bool(RE_ACK.search(text))

    if has_ack and not has_sub and not has_cla and not has_def:
        return "acknowledgement"
    if has_sub and not (has_def and len(text) < 120 and not has_cla):
        return "resolution"
    if has_cla and not has_def:
        return "clarification"
    if has_def:
        return "escalation"
    return "uncertain"


# ---------------------------------------------------------------------------
# Profiling Pipeline
# ---------------------------------------------------------------------------
def run_brand_profiling(
    raw_csv_path: Path = RAW_DATA_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, Any]:
    """
    Executes the complete brand profiling pipeline on the raw dataset.

    Returns:
      A dictionary containing:
        - "metadata": Execution metadata and summary
        - "brands": List of profiled brand dictionaries
    """
    t_start = time.time()
    print("=" * 75)
    print("STAGE 5: Brand Profiling & Discovery Pipeline")
    print("=" * 75)
    print(f"Source file: {raw_csv_path}")
    print(f"Chunk size : {chunk_size:,} rows\n")

    # -----------------------------------------------------------------------
    # Pass 1: Build Conversation Components (Union-Find)
    # -----------------------------------------------------------------------
    print("[Pass 1/3] Building conversation graph with Union-Find...")
    t0 = time.time()
    uf = UnionFind()

    for chunk in pd.read_csv(
        raw_csv_path,
        chunksize=chunk_size,
        usecols=["tweet_id", "in_response_to_tweet_id", "response_tweet_id"],
        dtype={"tweet_id": "Int64", "in_response_to_tweet_id": "Int64", "response_tweet_id": str},
    ):
        for tid, pid, resp in zip(chunk["tweet_id"], chunk["in_response_to_tweet_id"], chunk["response_tweet_id"]):
            if pd.notna(pid):
                uf.union(int(tid), int(pid))
            if pd.notna(resp):
                for r in str(resp).split(","):
                    r = r.strip()
                    if r and r.isnumeric():
                        uf.union(int(tid), int(r))

    print(f"  Graph built in {time.time()-t0:.2f}s (UnionFind entries: {len(uf.parent):,})")

    # -----------------------------------------------------------------------
    # Pass 2: Assign Brand to Each Conversation Component
    # -----------------------------------------------------------------------
    print("[Pass 2/3] Mapping conversations to brand support accounts...")
    t1 = time.time()
    conv_brand_votes: Dict[int, Counter] = defaultdict(Counter)

    for chunk in pd.read_csv(
        raw_csv_path,
        chunksize=chunk_size,
        usecols=["tweet_id", "author_id", "inbound"],
        dtype={"tweet_id": "Int64", "author_id": str, "inbound": bool},
    ):
        outbound = chunk[~chunk["inbound"]]
        for tid, auth in zip(outbound["tweet_id"], outbound["author_id"]):
            root = uf.find(int(tid))
            conv_brand_votes[root][auth] += 1

    # Every conversation belongs to EXACTLY ONE brand: majority brand vote
    conv_to_brand: Dict[int, str] = {}
    for root, counter in conv_brand_votes.items():
        majority_brand = counter.most_common(1)[0][0]
        conv_to_brand[root] = majority_brand

    discovered_brands = sorted(list(set(conv_to_brand.values())))
    print(
        f"  Mapped {len(conv_to_brand):,} conversations to {len(discovered_brands)} official brands in {time.time()-t1:.2f}s"
    )

    # -----------------------------------------------------------------------
    # Pass 3: Quantitative Aggregation Across All Brands
    # -----------------------------------------------------------------------
    print("[Pass 3/3] Streaming quantitative profiling across all brands...")
    t2 = time.time()

    # Per-conversation trackers
    conv_total_tweets: Counter = Counter()
    conv_cust_tweets: Counter = Counter()
    conv_supp_tweets: Counter = Counter()
    conv_lang_votes: Dict[int, Counter] = defaultdict(Counter)
    conv_has_resolution: Dict[int, bool] = defaultdict(bool)

    # Per-brand aggregates
    brand_resp_counts: Dict[str, Counter] = defaultdict(Counter)
    brand_topic_counts: Dict[str, Counter] = defaultdict(Counter)
    brand_customer_vocab: Dict[str, Counter] = defaultdict(Counter)

    row_count = 0
    for chunk in pd.read_csv(
        raw_csv_path,
        chunksize=chunk_size,
        usecols=["tweet_id", "inbound", "text"],
        dtype={"tweet_id": "Int64", "inbound": bool, "text": str},
    ):
        row_count += len(chunk)
        for tid, inb, text_val in zip(chunk["tweet_id"], chunk["inbound"], chunk["text"]):
            root = uf.find(int(tid))
            brand = conv_to_brand.get(root)
            if not brand:
                continue

            conv_total_tweets[root] += 1
            text_str = str(text_val) if pd.notna(text_val) else ""

            if inb:
                conv_cust_tweets[root] += 1
                # Language detection vote
                lang = infer_language(text_str)
                conv_lang_votes[root][lang] += 1

                # Diversity proxy topic matching
                for topic_key, topic_re in TOPIC_PATTERNS.items():
                    if topic_re.search(text_str):
                        brand_topic_counts[brand][topic_key] += 1

                # Lexical vocabulary tracking (subsample to keep memory lean)
                if len(brand_customer_vocab[brand]) < 25_000:
                    words = re.findall(r"\b[a-zA-Z]{3,}\b", text_str.lower())
                    filtered = [w for w in words if w not in STOPWORDS]
                    brand_customer_vocab[brand].update(filtered)
            else:
                conv_supp_tweets[root] += 1
                # Support response classification
                rtype = classify_response(text_str)
                brand_resp_counts[brand][rtype] += 1
                if rtype == "resolution":
                    conv_has_resolution[root] = True

        print(f"  Processed {row_count:,} rows...", flush=True)

    print(f"  Aggregation complete in {time.time()-t2:.2f}s")

    # -----------------------------------------------------------------------
    # Post-Processing: Compute Final Brand Metrics
    # -----------------------------------------------------------------------
    print("\nComputing final conversation-level metrics per brand...")

    # Group conversation roots by brand
    brand_roots: Dict[str, List[int]] = defaultdict(list)
    for root, brand in conv_to_brand.items():
        brand_roots[brand].append(root)

    profiled_brands: List[Dict[str, Any]] = []

    for brand in discovered_brands:
        roots = brand_roots[brand]
        n_convs = len(roots)
        if n_convs == 0:
            continue

        # Conversation lengths
        lengths = [conv_total_tweets[r] for r in roots]
        total_tweets = sum(lengths)
        cust_messages = sum(conv_cust_tweets[r] for r in roots)
        supp_messages = sum(conv_supp_tweets[r] for r in roots)

        avg_turns = round(statistics.mean(lengths), 2) if lengths else 0.0
        med_turns = float(statistics.median(lengths)) if lengths else 0.0
        max_turns = max(lengths) if lengths else 0

        # Multi-turn defined as >= 3 turns (customer inquiry -> support response -> customer follow-up)
        multiturn_convs = sum(1 for l in lengths if l >= 3)
        multiturn_pct = round((multiturn_convs / n_convs) * 100, 2) if n_convs else 0.0

        # Language calculation
        en_convs = 0
        for r in roots:
            votes = conv_lang_votes[r]
            if not votes:
                conv_lang = "Unknown"
            elif "Japanese" in votes:
                conv_lang = "Japanese"
            else:
                conv_lang = votes.most_common(1)[0][0]
            if conv_lang == "English":
                en_convs += 1

        non_en_convs = n_convs - en_convs
        en_pct = round((en_convs / n_convs) * 100, 2) if n_convs else 0.0

        # Support behavior metrics
        res_count = brand_resp_counts[brand]["resolution"]
        esc_count = brand_resp_counts[brand]["escalation"]
        cla_count = brand_resp_counts[brand]["clarification"]
        ack_count = brand_resp_counts[brand]["acknowledgement"]
        unc_count = brand_resp_counts[brand]["uncertain"]
        total_support = supp_messages if supp_messages > 0 else (res_count + esc_count + cla_count + ack_count + unc_count)

        res_pct = round((res_count / total_support) * 100, 2) if total_support else 0.0
        esc_pct = round((esc_count / total_support) * 100, 2) if total_support else 0.0
        cla_pct = round((cla_count / total_support) * 100, 2) if total_support else 0.0
        ack_pct = round((ack_count / total_support) * 100, 2) if total_support else 0.0
        unc_pct = round((unc_count / total_support) * 100, 2) if total_support else 0.0

        substantive_convs = sum(1 for r in roots if conv_has_resolution[r])

        # Intent Diversity Proxy: Topic Shannon Entropy + Lexical Richness
        topic_counts = brand_topic_counts[brand]
        total_topics = sum(topic_counts.values())
        k_topics = len(TOPIC_PATTERNS)
        max_entropy = math.log2(k_topics)

        if total_topics > 0:
            entropy = -sum((c / total_topics) * math.log2(c / total_topics) for c in topic_counts.values() if c > 0)
            norm_topic_entropy = entropy / max_entropy
        else:
            norm_topic_entropy = 0.0

        vocab = brand_customer_vocab[brand]
        total_vocab_tokens = sum(vocab.values())
        unique_vocab_words = len(vocab)
        lexical_richness = (unique_vocab_words / total_vocab_tokens) if total_vocab_tokens > 0 else 0.0
        # Diversity proxy composite (0.0 to 1.0)
        diversity_proxy = round(0.70 * norm_topic_entropy + 0.30 * min(1.0, lexical_richness * 10.0), 4)

        profiled_brands.append(
            {
                "brand": brand,
                "conversation_count": n_convs,
                "tweet_count": total_tweets,
                "customer_message_count": cust_messages,
                "support_message_count": supp_messages,
                "english_conversation_count": en_convs,
                "non_english_conversation_count": non_en_convs,
                "english_percentage": en_pct,
                "resolution_count": res_count,
                "resolution_percentage": res_pct,
                "escalation_count": esc_count,
                "escalation_percentage": esc_pct,
                "clarification_count": cla_count,
                "clarification_percentage": cla_pct,
                "acknowledgement_count": ack_count,
                "acknowledgement_percentage": ack_pct,
                "uncertain_count": unc_count,
                "uncertain_percentage": unc_pct,
                "substantive_resolution_conversations": substantive_convs,
                "multiturn_conversation_count": multiturn_convs,
                "multiturn_percentage": multiturn_pct,
                "average_turns": avg_turns,
                "median_turns": med_turns,
                "maximum_turns": max_turns,
                "diversity_proxy": diversity_proxy,
            }
        )

    # Sort deterministically by conversation_count descending, then brand name
    profiled_brands.sort(key=lambda b: (-b["conversation_count"], b["brand"]))

    elapsed_total = time.time() - t_start
    print(f"\nSuccessfully profiled {len(profiled_brands)} brands in {elapsed_total:.2f}s.")

    return {
        "metadata": {
            "stage": 5,
            "pipeline": "profile_brands.py",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_brands_discovered": len(profiled_brands),
            "total_conversations_profiled": len(conv_to_brand),
            "total_tweets_profiled": row_count,
            "execution_time_seconds": round(elapsed_total, 2),
        },
        "brands": profiled_brands,
    }


def main():
    results = run_brand_profiling()
    print("\nTop 15 Profiled Brands by Conversation Count:")
    print("-" * 75)
    print(f"{'Brand':<20} | {'Convs':>8} | {'Tweets':>8} | {'En %':>6} | {'Res %':>6} | {'Esc %':>6} | {'Multi %':>7} | {'DivProxy':>8}")
    print("-" * 75)
    for b in results["brands"][:15]:
        print(
            f"{b['brand']:<20} | {b['conversation_count']:>8,} | {b['tweet_count']:>8,} | "
            f"{b['english_percentage']:>5.1f}% | {b['resolution_percentage']:>5.1f}% | "
            f"{b['escalation_percentage']:>5.1f}% | {b['multiturn_percentage']:>6.1f}% | "
            f"{b['diversity_proxy']:>8.4f}"
        )
    print("-" * 75)


if __name__ == "__main__":
    main()
