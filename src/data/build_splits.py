"""
build_splits.py
---------------
Stage 4: Dataset Split & Evaluation Foundation

Creates reproducible train / validation / test splits of the canonical
AmazonHelp AI dataset while ensuring:
  - Global conversation isolation: all examples sharing a conversation_id
    end up in the exact same split across ALL datasets (zero leakage).
  - Stratified split across response_type and language.
  - Separate evaluation benchmark files for human and automated evaluation.

Outputs (written to data/processed/splits/):
  train/
    amazon_resolution_pairs_train.jsonl
    amazon_escalation_pairs_train.jsonl
    amazon_clarification_pairs_train.jsonl
    amazon_multiturn_examples_train.jsonl
  val/
    amazon_resolution_pairs_val.jsonl
    amazon_escalation_pairs_val.jsonl
    amazon_clarification_pairs_val.jsonl
    amazon_multiturn_examples_val.jsonl
  test/
    amazon_resolution_pairs_test.jsonl
    amazon_escalation_pairs_test.jsonl
    amazon_clarification_pairs_test.jsonl
    amazon_multiturn_examples_test.jsonl
  amazon_eval_benchmark.jsonl   (merged English test-set for evaluation)
  amazon_splits_manifest.json   (reproducibility metadata)

Design Principles:
  - Deterministic: fixed random seed = 42.
  - Conversation-safe: conversation boundaries are globally preserved across all files.
  - Stratified: splits preserve response_type + language distribution.
  - No data modification: source JSONL files are never altered.
"""

import hashlib
import json
from pathlib import Path
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Configure UTF-8 stdout on Windows
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

SOURCE_FILES = {
    "resolution": PROCESSED_DIR / "amazon_resolution_pairs.jsonl",
    "escalation": PROCESSED_DIR / "amazon_escalation_pairs.jsonl",
    "clarification": PROCESSED_DIR / "amazon_clarification_pairs.jsonl",
    "multiturn": PROCESSED_DIR / "amazon_multiturn_examples.jsonl",
}

SPLITS_MANIFEST_PATH = SPLITS_DIR / "amazon_splits_manifest.json"
EVAL_BENCHMARK_PATH = SPLITS_DIR / "amazon_eval_benchmark.jsonl"

# ---------------------------------------------------------------------------
# Split ratios
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load an entire JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    """Return hex SHA-256 of a file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Global Conversation-Safe Stratified Split
# ---------------------------------------------------------------------------
def partition_conversations_globally(
    all_datasets: Dict[str, List[Dict[str, Any]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Partitions all unique conversation_ids across all source datasets globally
    so that every conversation belongs to exactly one split (zero leakage).
    Stratification is performed over conversation primary response type and language.
    """
    rng = random.Random(seed)

    # Collect metadata per conversation_id
    conv_metadata: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"response_types": Counter(), "languages": Counter()})
    for ds_name, records in all_datasets.items():
        for r in records:
            cid = r["conversation_id"]
            rtype = r.get("response_type", ds_name)
            lang = r.get("language", "English")
            conv_metadata[cid]["response_types"][rtype] += 1
            conv_metadata[cid]["languages"][lang] += 1

    # Assign each conversation to a stratum
    stratum_to_convs: Dict[str, List[str]] = defaultdict(list)
    for cid, meta in conv_metadata.items():
        primary_rtype = meta["response_types"].most_common(1)[0][0]
        primary_lang = meta["languages"].most_common(1)[0][0]
        stratum_key = f"{primary_rtype}|{primary_lang}"
        stratum_to_convs[stratum_key].append(cid)

    train_ids: Set[str] = set()
    val_ids: Set[str] = set()
    test_ids: Set[str] = set()

    for stratum_key in sorted(stratum_to_convs.keys()):
        conv_ids = stratum_to_convs[stratum_key]
        shuffled = sorted(conv_ids)
        rng.shuffle(shuffled)
        n = len(shuffled)

        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio)) if n > 2 else 0
        n_test = n - n_train - n_val

        if n_test <= 0 and n >= 3:
            n_train -= 1
            n_test = 1
        elif n_test <= 0:
            n_train = n
            n_val = 0
            n_test = 0

        train_ids.update(shuffled[:n_train])
        val_ids.update(shuffled[n_train : n_train + n_val])
        test_ids.update(shuffled[n_train + n_val :])

    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Evaluation Benchmark Construction
# ---------------------------------------------------------------------------
def build_eval_benchmark(
    split_test_records: Dict[str, List[Dict]]
) -> List[Dict]:
    """
    Build a unified evaluation benchmark from the test split.
    Only includes English examples.
    """
    benchmark: List[Dict] = []
    bm_index = 0

    for dataset_name in sorted(split_test_records.keys()):
        records = split_test_records[dataset_name]
        for rec in records:
            if rec.get("language", "") != "English":
                continue

            bm_index += 1
            bm_id = f"bm_{bm_index:07d}"

            if dataset_name == "multiturn":
                query = " | ".join(
                    turn["text"]
                    for turn in rec.get("context", [])
                    if turn["role"] == "customer"
                )
                reference_response = rec.get("target_response", "")
                reference_response_original = rec.get("target_response_original", "")
            else:
                query = rec.get("customer_message", "")
                reference_response = rec.get("support_response", "")
                reference_response_original = rec.get("support_response_original", "")

            benchmark.append(
                {
                    "benchmark_id": bm_id,
                    "source_example_id": rec["example_id"],
                    "source_dataset": dataset_name,
                    "conversation_id": rec["conversation_id"],
                    "response_type": rec.get("response_type", ""),
                    "language": rec.get("language", "English"),
                    "query": query,
                    "reference_response": reference_response,
                    "reference_response_original": reference_response_original,
                    "predicted_response": None,
                    "retrieval_candidate_ids": [],
                    "eval_scores": {},
                }
            )

    return benchmark


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    print("=" * 70)
    print("Stage 4: Dataset Split & Evaluation Foundation")
    print("=" * 70)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    (SPLITS_DIR / "train").mkdir(exist_ok=True)
    (SPLITS_DIR / "val").mkdir(exist_ok=True)
    (SPLITS_DIR / "test").mkdir(exist_ok=True)

    source_checksums: Dict[str, str] = {}
    all_datasets: Dict[str, List[Dict[str, Any]]] = {}

    for dataset_name, source_path in SOURCE_FILES.items():
        print(f"Loading {source_path.name} ...")
        records = load_jsonl(source_path)
        source_checksums[source_path.name] = file_sha256(source_path)
        all_datasets[dataset_name] = records
        print(f"  Loaded {len(records):,} records.")

    # Partition all conversations globally
    print("\nPartitioning conversations globally (zero leakage)...")
    train_convs, val_convs, test_convs = partition_conversations_globally(
        all_datasets, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
    )

    # Verify global disjointness
    assert train_convs.isdisjoint(val_convs), "Global leakage: TRAIN/VAL overlap!"
    assert train_convs.isdisjoint(test_convs), "Global leakage: TRAIN/TEST overlap!"
    assert val_convs.isdisjoint(test_convs), "Global leakage: VAL/TEST overlap!"
    print(f"  ✓ Total unique conversations partitioned: {len(train_convs) + len(val_convs) + len(test_convs):,}")
    print(f"  ✓ Train: {len(train_convs):,} | Val: {len(val_convs):,} | Test: {len(test_convs):,}")
    print("  ✓ Zero conversation leakage verified across all splits and datasets.")

    all_split_stats: Dict[str, Any] = {}
    all_test_records: Dict[str, List[Dict]] = {}

    for dataset_name, source_path in SOURCE_FILES.items():
        records = all_datasets[dataset_name]
        train = [r for r in records if r["conversation_id"] in train_convs]
        val = [r for r in records if r["conversation_id"] in val_convs]
        test = [r for r in records if r["conversation_id"] in test_convs]
        all_test_records[dataset_name] = test

        stem = source_path.stem
        write_jsonl(train, SPLITS_DIR / "train" / f"{stem}_train.jsonl")
        write_jsonl(val, SPLITS_DIR / "val" / f"{stem}_val.jsonl")
        write_jsonl(test, SPLITS_DIR / "test" / f"{stem}_test.jsonl")

        print(f"\n[{dataset_name}] Split -> train={len(train):,} | val={len(val):,} | test={len(test):,}")

        def split_stats(recs: List[Dict]) -> Dict[str, Any]:
            lang_counts = Counter(r.get("language", "Unknown") for r in recs)
            rtype_counts = Counter(r.get("response_type", "unknown") for r in recs)
            conv_ids = {r["conversation_id"] for r in recs}
            return {
                "n_records": len(recs),
                "n_conversations": len(conv_ids),
                "by_language": dict(lang_counts.most_common()),
                "by_response_type": dict(rtype_counts.most_common()),
            }

        all_split_stats[dataset_name] = {
            "source_file": source_path.name,
            "total_records": len(records),
            "train": split_stats(train),
            "val": split_stats(val),
            "test": split_stats(test),
        }

    # Build evaluation benchmark
    print("\n[benchmark] Building evaluation benchmark ...")
    benchmark = build_eval_benchmark(all_test_records)
    write_jsonl(benchmark, EVAL_BENCHMARK_PATH)
    bm_rtype = Counter(r["response_type"] for r in benchmark)
    bm_ds = Counter(r["source_dataset"] for r in benchmark)
    print(f"  Total benchmark examples: {len(benchmark):,}")
    print(f"  ✓ Written to {EVAL_BENCHMARK_PATH.name}")

    elapsed = time.time() - t_start
    manifest = {
        "manifest_version": "1.0",
        "stage": "Stage 4 -- Dataset Split & Evaluation Foundation",
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "split_strategy": (
            "Global conversation-safe stratified split across all datasets. "
            "Every conversation_id is assigned to exactly one split globally (zero leakage)."
        ),
        "source_file_checksums_sha256": source_checksums,
        "dataset_splits": all_split_stats,
        "eval_benchmark": {
            "path": str(EVAL_BENCHMARK_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "total_examples": len(benchmark),
            "language_filter": "English only",
            "by_response_type": dict(bm_rtype.most_common()),
            "by_source_dataset": dict(bm_ds.most_common()),
            "schema": {
                "benchmark_id": "Unique ID for this benchmark example.",
                "source_example_id": "Original example_id from source JSONL.",
                "source_dataset": "resolution | escalation | clarification | multiturn",
                "conversation_id": "Originating conversation cluster.",
                "response_type": "Inferred support response category.",
                "language": "Inferred language (English for benchmark).",
                "query": "Customer message (or concatenated customer turns for multiturn).",
                "reference_response": "Normalized ground-truth support response.",
                "reference_response_original": "Raw (un-normalized) support response.",
                "predicted_response": "Placeholder for future RAG/LLM output.",
                "retrieval_candidate_ids": "Placeholder for future retrieval results.",
                "eval_scores": "Placeholder for future BLEU/ROUGE/BERTScore/LLM-judge scores.",
            },
        },
        "processing_time_seconds": round(elapsed, 2),
        "known_limitations": [
            "Evaluation benchmark is English-only; multilingual evaluation is not yet included.",
            "predicted_response and eval_scores fields are empty placeholders for the RAG/LLM stage.",
        ],
    }

    with open(SPLITS_MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"\n✓ Splits manifest written to {SPLITS_MANIFEST_PATH.name}")
    print(f"\n{'=' * 70}")
    print(f"Stage 4 complete in {elapsed:.1f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
