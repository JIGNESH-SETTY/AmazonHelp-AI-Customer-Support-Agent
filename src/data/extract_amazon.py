"""
extract_amazon.py
-----------------
Memory-efficient two-pass extraction pipeline for AmazonHelp conversations
from the raw Twitter Customer Support dataset (`data/raw/twcs.csv`).

Outputs:
  1. `data/processed/amazonhelp_tweets.csv`:
     All tweets authored by AmazonHelp (support agent tweets).
  2. `data/processed/amazonhelp_conversations.csv`:
     All customer and AmazonHelp tweets needed to reconstruct conversation threads.

Design:
  - Pass 1: Reads dataset in chunks (default 50,000 rows). Extracts AmazonHelp tweets
    and collects relevant conversation tweet IDs (AmazonHelp IDs, parent IDs, response IDs).
    Streams AmazonHelp tweets to disk to keep memory usage minimal.
  - Pass 2: Reads dataset in chunks. Extracts all rows whose tweet_id matches any
    collected conversation tweet ID. Streams matched rows to disk.
  - Validation: Performs integrity and conversation reconstruction checks, calculates
    broken references, duplicate counts, author distributions, and mixed conversations.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Ensure Windows consoles handle emojis and unicode characters cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Default paths and configuration constants
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "twcs.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
AMAZON_TWEETS_PATH = PROCESSED_DIR / "amazonhelp_tweets.csv"
AMAZON_CONVERSATIONS_PATH = PROCESSED_DIR / "amazonhelp_conversations.csv"

DEFAULT_CHUNK_SIZE = 50_000
PROGRESS_INTERVAL = 500_000
BRAND_AUTHOR = "AmazonHelp"

COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

DTYPES = {
    "tweet_id": "Int64",
    "author_id": "str",
    "inbound": "bool",
    "created_at": "str",
    "text": "str",
    "response_tweet_id": "str",
    "in_response_to_tweet_id": "Int64",
}


class UnionFind:
    """Disjoint Set Union (Union-Find) for grouping tweets into conversation components."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y


def parse_response_ids(val: Optional[str]) -> List[int]:
    """Parses response_tweet_id field which may contain comma-separated IDs."""
    if pd.isna(val) or not val:
        return []
    res = []
    for part in str(val).split(","):
        cleaned = part.strip()
        if cleaned:
            try:
                res.append(int(float(cleaned)))
            except (ValueError, TypeError):
                pass
    return res


def run_pass_1(
    raw_path: Path,
    amazon_tweets_output_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tuple[Set[int], int]:
    """
    Pass 1:
      - Streams twcs.csv in chunks.
      - Filters rows where author_id == BRAND_AUTHOR ('AmazonHelp').
      - Collects relevant IDs: AmazonHelp tweet_ids, parent IDs, and response IDs.
      - Streams AmazonHelp tweets to amazonhelp_tweets.csv.

    Returns:
      conversation_ids: Set of all tweet IDs required to reconstruct conversations.
      amazon_tweets_count: Total number of AmazonHelp tweets extracted.
    """
    print("\n" + "=" * 70)
    print(" PASS 1/2: Identifying AmazonHelp tweets and conversation tweet IDs")
    print("=" * 70)
    print(f"Reading: {raw_path}")
    print(f"Chunk size: {chunk_size:,} rows")
    print(f"Streaming AmazonHelp tweets to: {amazon_tweets_output_path}\n", flush=True)

    conversation_ids: Set[int] = set()
    amazon_tweets_count = 0
    total_raw_rows = 0
    start_time = time.time()
    first_chunk_written = False

    for chunk in pd.read_csv(
        raw_path,
        chunksize=chunk_size,
        usecols=COLUMNS,
        dtype=DTYPES,
    ):
        total_raw_rows += len(chunk)

        # Filter AmazonHelp tweets
        ah_chunk = chunk[chunk["author_id"] == BRAND_AUTHOR]
        ah_len = len(ah_chunk)

        if ah_len > 0:
            amazon_tweets_count += ah_len

            # 1. Collect AmazonHelp tweet_ids
            ah_ids = ah_chunk["tweet_id"].dropna().astype(int)
            conversation_ids.update(ah_ids)

            # 2. Collect AmazonHelp parent tweet_ids (in_response_to_tweet_id)
            parent_ids = ah_chunk["in_response_to_tweet_id"].dropna().astype(int)
            conversation_ids.update(parent_ids)

            # 3. Collect AmazonHelp response_tweet_ids
            for resp_val in ah_chunk["response_tweet_id"].dropna():
                for r_id in parse_response_ids(resp_val):
                    conversation_ids.add(r_id)

            # Stream AmazonHelp tweets directly to disk
            mode = "w" if not first_chunk_written else "a"
            header = not first_chunk_written
            ah_chunk.to_csv(amazon_tweets_output_path, mode=mode, header=header, index=False)
            first_chunk_written = True

        if total_raw_rows % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(
                f"Pass 1: Processed {total_raw_rows:,} raw rows... "
                f"({amazon_tweets_count:,} AmazonHelp tweets, "
                f"{len(conversation_ids):,} conversation IDs collected, {elapsed:.1f}s)",
                flush=True,
            )

    elapsed_total = time.time() - start_time
    print(
        f"\nPass 1 complete in {elapsed_total:.2f}s."
        f"\nTotal raw rows scanned: {total_raw_rows:,}"
        f"\nAmazonHelp support tweets found: {amazon_tweets_count:,}"
        f"\nUnique conversation tweet IDs collected: {len(conversation_ids):,}",
        flush=True,
    )

    return conversation_ids, amazon_tweets_count


def run_pass_2(
    raw_path: Path,
    conversation_output_path: Path,
    target_ids: Set[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """
    Pass 2:
      - Streams twcs.csv in chunks.
      - Filters rows where tweet_id is in target_ids.
      - Streams matched conversation rows to amazonhelp_conversations.csv.

    Returns:
      conversation_tweets_count: Total number of conversation tweets extracted.
    """
    print("\n" + "=" * 70)
    print(" PASS 2/2: Extracting conversation tweets (Customer & AmazonHelp)")
    print("=" * 70)
    print(f"Reading: {raw_path}")
    print(f"Target tweet IDs to extract: {len(target_ids):,}")
    print(f"Streaming conversation tweets to: {conversation_output_path}\n", flush=True)

    conversation_tweets_count = 0
    total_raw_rows = 0
    start_time = time.time()
    first_chunk_written = False

    for chunk in pd.read_csv(
        raw_path,
        chunksize=chunk_size,
        usecols=COLUMNS,
        dtype=DTYPES,
    ):
        total_raw_rows += len(chunk)

        # Match rows with collected conversation IDs
        matched = chunk[chunk["tweet_id"].isin(target_ids)]
        matched_len = len(matched)

        if matched_len > 0:
            conversation_tweets_count += matched_len
            mode = "w" if not first_chunk_written else "a"
            header = not first_chunk_written
            matched.to_csv(conversation_output_path, mode=mode, header=header, index=False)
            first_chunk_written = True

        if total_raw_rows % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(
                f"Pass 2: Processed {total_raw_rows:,} raw rows... "
                f"({conversation_tweets_count:,} conversation tweets matched, {elapsed:.1f}s)",
                flush=True,
            )

    elapsed_total = time.time() - start_time
    print(
        f"\nPass 2 complete in {elapsed_total:.2f}s."
        f"\nTotal conversation tweets extracted: {conversation_tweets_count:,}",
        flush=True,
    )

    return conversation_tweets_count


def validate_extraction(
    amazon_tweets_path: Path,
    conversations_path: Path,
) -> None:
    """
    Performs full conversation reconstruction check, duplicate analysis,
    and integrity validation on the extracted datasets.
    """
    print("\n" + "=" * 70)
    print(" VALIDATING EXTRACTION RESULTS & RECONSTRUCTING CONVERSATIONS")
    print("=" * 70, flush=True)

    # 1. Load AmazonHelp tweets dataset
    print(f"Loading {amazon_tweets_path.name} for validation...")
    df_ah = pd.read_csv(amazon_tweets_path, dtype=DTYPES)

    # 2. Load AmazonHelp conversations dataset
    print(f"Loading {conversations_path.name} for validation...")
    df_conv = pd.read_csv(conversations_path, dtype=DTYPES)

    # --- Duplicate Analysis ---
    ah_total_rows = len(df_ah)
    ah_unique_ids = df_ah["tweet_id"].nunique()
    ah_duplicates = ah_total_rows - ah_unique_ids

    conv_total_rows = len(df_conv)
    conv_unique_ids = df_conv["tweet_id"].nunique()
    conv_duplicates = conv_total_rows - conv_unique_ids

    # --- Author & Message Distributions ---
    unique_authors = df_conv["author_id"].nunique()
    support_mask = df_conv["author_id"] == BRAND_AUTHOR
    customer_mask = ~support_mask

    support_tweets = int(support_mask.sum())
    customer_tweets = int(customer_mask.sum())

    tweets_with_parent = int(df_conv["in_response_to_tweet_id"].notna().sum())
    tweets_with_responses = int(df_conv["response_tweet_id"].notna().sum())

    # --- Conversation Integrity (Broken Parent References) ---
    conv_tweet_id_set = set(df_conv["tweet_id"].dropna().astype(int))

    # Parent IDs that were referenced in conversations dataset
    referenced_parents = df_conv["in_response_to_tweet_id"].dropna().astype(int)
    broken_mask = ~referenced_parents.isin(conv_tweet_id_set)
    broken_parent_refs = int(broken_mask.sum())
    broken_parent_unique_ids = int((~pd.Series(list(set(referenced_parents))).isin(conv_tweet_id_set)).sum())
    broken_pct = (broken_parent_refs / tweets_with_parent * 100) if tweets_with_parent > 0 else 0.0

    # --- Conversation Reconstruction & Mixed Conversations ---
    uf = UnionFind()
    for tid in conv_tweet_id_set:
        uf.find(tid)

    for _, row in df_conv[["tweet_id", "in_response_to_tweet_id"]].dropna().iterrows():
        tid = int(row["tweet_id"])
        pid = int(row["in_response_to_tweet_id"])
        if pid in conv_tweet_id_set:
            uf.union(tid, pid)

    # Build connected components
    components: Dict[int, List[Tuple[str, bool]]] = defaultdict(list)
    for tid, author, inbound in zip(
        df_conv["tweet_id"], df_conv["author_id"], df_conv["inbound"]
    ):
        root = uf.find(int(tid))
        components[root].append((str(author), bool(inbound)))

    total_conversations = len(components)
    mixed_conversations = 0

    for tweets in components.values():
        has_support = any(auth == BRAND_AUTHOR for auth, _ in tweets)
        has_customer = any(auth != BRAND_AUTHOR for auth, _ in tweets)
        if has_support and has_customer:
            mixed_conversations += 1

    mixed_pct = (mixed_conversations / total_conversations * 100) if total_conversations > 0 else 0.0

    # --- Print Summary (Section 6 format) ---
    print("\n" + "=" * 50)
    print("AMAZONHELP EXTRACTION SUMMARY")
    print("=" * 29)
    print(f"AmazonHelp tweets: {ah_total_rows:,}")
    print(f"Conversation tweets: {conv_total_rows:,}")
    print(f"Unique authors: {unique_authors:,}")
    print(f"Customer tweets: {customer_tweets:,}")
    print(f"Support tweets: {support_tweets:,}")
    print(f"Tweets with parent: {tweets_with_parent:,}")
    print(f"Tweets with responses: {tweets_with_responses:,}")
    print(f"Broken parent references: {broken_parent_refs:,} ({broken_pct:.2f}%)")
    print(f"Mixed customer/support conversations: {mixed_conversations:,} ({mixed_pct:.2f}%)")
    print("=" * 50 + "\n")

    # --- Duplicate Analysis Report ---
    print("=" * 50)
    print("DUPLICATE TWEET ID ANALYSIS")
    print("=" * 50)
    print(f"AmazonHelp Tweets File ({amazon_tweets_path.name}):")
    print(f"  - Total rows: {ah_total_rows:,}")
    print(f"  - Unique tweet IDs: {ah_unique_ids:,}")
    print(f"  - Duplicate tweet IDs: {ah_duplicates:,}")
    print(f"Conversation Tweets File ({conversations_path.name}):")
    print(f"  - Total rows: {conv_total_rows:,}")
    print(f"  - Unique tweet IDs: {conv_unique_ids:,}")
    print(f"  - Duplicate tweet IDs: {conv_duplicates:,}")
    if conv_duplicates == 0 and ah_duplicates == 0:
        print("  - Result: No duplicates found. Every tweet ID is globally unique.")
    else:
        print(f"  - Notice: Found duplicate rows in output files.")
    print("=" * 50 + "\n")

    # --- Integrity Report ---
    print("=" * 50)
    print("CONVERSATION INTEGRITY ANALYSIS")
    print("=" * 50)
    print(f"Total conversation threads / trees: {total_conversations:,}")
    print(f"Mixed customer/support threads: {mixed_conversations:,} ({mixed_pct:.2f}%)")
    print(f"Total tweets referencing a parent: {tweets_with_parent:,}")
    print(f"Broken parent references (missing parent): {broken_parent_refs:,} ({broken_pct:.2f}%)")
    print(f"Unique missing parent IDs: {broken_parent_unique_ids:,}")
    print("Explanation: Broken references occur when a customer's original tweet was deleted,")
    print("posted outside the dataset's collection window, or omitted from the Kaggle dump.")
    print("All rows with broken parent references have been preserved intact.")
    print("=" * 50 + "\n")

    # --- Sample Rows Display ---
    with pd.option_context("display.max_columns", None, "display.width", 1000, "display.max_colwidth", 80):
        print("=" * 70)
        print(f"FIRST 10 ROWS: {amazon_tweets_path.name}")
        print("=" * 70)
        print(df_ah.head(10).to_string())
        print("\n")

        print("=" * 70)
        print(f"FIRST 10 ROWS: {conversations_path.name}")
        print("=" * 70)
        print(df_conv.head(10).to_string())
        print("\n")


    # --- Sanity Checks ---
    assert (df_ah["author_id"] == BRAND_AUTHOR).all(), "Error: non-AmazonHelp author in amazonhelp_tweets.csv!"
    assert (df_conv["author_id"] == BRAND_AUTHOR).any(), "Error: No AmazonHelp tweets in conversations dataset!"
    assert (df_conv["author_id"] != BRAND_AUTHOR).any(), "Error: No customer tweets in conversations dataset!"
    print("[SUCCESS] All validation assertions passed successfully!\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract AmazonHelp tweets and conversation threads from TWCS dataset."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_DATA_PATH,
        help=f"Path to raw twcs.csv (default: {RAW_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Directory to save processed datasets (default: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Pandas chunksize for streaming (default: {DEFAULT_CHUNK_SIZE})",
    )

    args = parser.parse_args()

    # Verify input exists
    if not args.raw_path.exists():
        print(f"Error: Raw dataset not found at '{args.raw_path}'.", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    amazon_tweets_path = args.output_dir / "amazonhelp_tweets.csv"
    conversations_path = args.output_dir / "amazonhelp_conversations.csv"

    overall_start = time.time()

    # Pass 1: Collect IDs & stream AmazonHelp support tweets
    conversation_ids, _ = run_pass_1(
        raw_path=args.raw_path,
        amazon_tweets_output_path=amazon_tweets_path,
        chunk_size=args.chunksize,
    )

    # Pass 2: Stream all conversation tweets (customer & AmazonHelp)
    run_pass_2(
        raw_path=args.raw_path,
        conversation_output_path=conversations_path,
        target_ids=conversation_ids,
        chunk_size=args.chunksize,
    )

    # Validation Stage
    validate_extraction(
        amazon_tweets_path=amazon_tweets_path,
        conversations_path=conversations_path,
    )

    total_time = time.time() - overall_start
    print(f"Extraction and validation pipeline completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
