"""
validate_golden_set.py
----------------------
STAGE 7: Dedicated Golden Evaluation Set Validator

Performs automated verification of the Golden Evaluation Set:
  1. Artifact existence (.jsonl, .csv, manifest.json)
  2. Schema validity and presence of all required fields
  3. Unique example_id across all records
  4. Unique conversation_id across all records
  5. No empty required strings or missing values
  6. Valid intent IDs matching Stage 6 taxonomy
  7. Valid difficulty labels (easy, medium, hard)
  8. Zero duplicate customer messages
  9. Target-size constraints (e.g. 200 examples)
  10. Canonical intent coverage (10/10 canonical intents covered)
  11. Split integrity (zero overlap with training or validation splits)

Usage:
  python -m src.evaluation.validate_golden_set
"""

from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

# UTF-8 stdout configuration for Windows environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"

GOLDEN_JSONL_PATH = GOLDEN_DIR / "golden_evaluation_set.jsonl"
GOLDEN_CSV_PATH = GOLDEN_DIR / "golden_evaluation_set.csv"
GOLDEN_MANIFEST_PATH = GOLDEN_DIR / "golden_set_manifest.json"
TAXONOMY_PATH = DATA_DIR / "intent_taxonomy.json"
LABELS_PATH = PROCESSED_DIR / "amazonhelp_intent_labels.jsonl"

REQUIRED_FIELDS = [
    "example_id",
    "conversation_id",
    "brand",
    "customer_message",
    "primary_intent",
    "primary_intent_name",
    "difficulty",
    "expected_behavior",
    "gold_response_guidance",
    "reference_support_response",
    "response_type",
    "source_reference",
    "split",
]

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_RESPONSE_TYPES = {"resolution", "escalation", "clarification"}


def run_validation(
    expected_total: int = 200,
    expected_canonical_intents: int = 10,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Validates Golden Evaluation Set artifacts against all quality and integrity constraints.
    Returns (is_valid, summary_metrics, error_messages).
    """
    errors: List[str] = []
    metrics: Dict[str, Any] = {}

    # 1. File existence
    if not GOLDEN_JSONL_PATH.exists():
        errors.append(f"Missing JSONL file: {GOLDEN_JSONL_PATH}")
    if not GOLDEN_CSV_PATH.exists():
        errors.append(f"Missing CSV file: {GOLDEN_CSV_PATH}")
    if not GOLDEN_MANIFEST_PATH.exists():
        errors.append(f"Missing Manifest file: {GOLDEN_MANIFEST_PATH}")

    if errors:
        return False, metrics, errors

    # 2. Load taxonomy
    with open(TAXONOMY_PATH, encoding="utf-8") as fh:
        tax = json.load(fh)
    canonical_intent_ids = {i["intent_id"] for i in tax["intents"]}
    all_valid_intent_ids = set(canonical_intent_ids)
    if "fallback_category" in tax:
        all_valid_intent_ids.add(tax["fallback_category"]["intent_id"])

    # 3. Load manifest
    with open(GOLDEN_MANIFEST_PATH, encoding="utf-8") as fh:
        manifest = json.load(fh)

    # 4. Load train and val conversation IDs to verify zero leakage
    train_val_convs: Set[str] = set()
    with open(LABELS_PATH, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("split") in ("train", "val"):
                train_val_convs.add(row["conversation_id"])

    # 5. Read JSONL records
    records: List[Dict[str, Any]] = []
    with open(GOLDEN_JSONL_PATH, encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                records.append(rec)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_idx}: Invalid JSON: {exc}")

    metrics["total_records"] = len(records)

    # 6. Check size constraint
    if len(records) != expected_total:
        errors.append(f"Expected {expected_total} records, but found {len(records)}.")

    # 7. Check field completeness and schema
    example_ids: List[str] = []
    conv_ids: List[str] = []
    customer_texts: List[str] = []
    primary_intents: List[str] = []
    difficulties: List[str] = []
    missing_fields_count = 0
    invalid_intents_count = 0
    invalid_diff_count = 0
    leakage_count = 0

    for idx, rec in enumerate(records, start=1):
        # Required fields presence and non-emptiness
        for field in REQUIRED_FIELDS:
            val = rec.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing_fields_count += 1
                errors.append(f"Record {idx} ({rec.get('example_id', 'unknown')}): Missing required field '{field}'.")

        ex_id = rec.get("example_id", "")
        conv_id = rec.get("conversation_id", "")
        cust_text = rec.get("customer_message", "")
        p_intent = rec.get("primary_intent", "")
        diff = rec.get("difficulty", "")
        split_val = rec.get("split", "")

        example_ids.append(ex_id)
        conv_ids.append(conv_id)
        customer_texts.append(cust_text.strip().lower())
        primary_intents.append(p_intent)
        difficulties.append(diff)

        # Intent validation
        if p_intent not in all_valid_intent_ids:
            invalid_intents_count += 1
            errors.append(f"Record {idx}: Invalid primary_intent '{p_intent}'.")

        # Difficulty validation
        if diff not in VALID_DIFFICULTIES:
            invalid_diff_count += 1
            errors.append(f"Record {idx}: Invalid difficulty '{diff}'.")

        # Split & Leakage validation
        if split_val != "test":
            errors.append(f"Record {idx}: Invalid split '{split_val}', expected 'test'.")

        if conv_id in train_val_convs:
            leakage_count += 1
            errors.append(f"Record {idx}: Leakage detected! Conv ID '{conv_id}' exists in train/val splits.")

    metrics["unique_examples"] = len(set(example_ids))
    metrics["unique_conversations"] = len(set(conv_ids))
    metrics["duplicate_examples"] = len(records) - len(set(example_ids))
    metrics["duplicate_conversations"] = len(records) - len(set(conv_ids))
    metrics["duplicate_customer_texts"] = len(records) - len(set(customer_texts))
    metrics["missing_required_fields"] = missing_fields_count
    metrics["invalid_intents"] = invalid_intents_count
    metrics["invalid_difficulties"] = invalid_diff_count
    metrics["train_val_leakage"] = leakage_count

    if metrics["duplicate_examples"] > 0:
        errors.append(f"Found {metrics['duplicate_examples']} duplicate example IDs.")
    if metrics["duplicate_conversations"] > 0:
        errors.append(f"Found {metrics['duplicate_conversations']} duplicate conversation IDs.")
    if metrics["duplicate_customer_texts"] > 0:
        errors.append(f"Found {metrics['duplicate_customer_texts']} duplicate customer messages.")

    # 8. Intent coverage check
    covered_intents = set(primary_intents)
    covered_canonical = covered_intents.intersection(canonical_intent_ids)
    metrics["canonical_intents_covered"] = len(covered_canonical)
    metrics["total_canonical_intents"] = len(canonical_intent_ids)
    metrics["all_intents_covered"] = len(covered_intents)

    if len(covered_canonical) < expected_canonical_intents:
        errors.append(
            f"Canonical intent coverage shortfall: {len(covered_canonical)}/{expected_canonical_intents} covered."
        )

    # 9. Verify CSV row count matches JSONL
    with open(GOLDEN_CSV_PATH, encoding="utf-8") as fh:
        csv_reader = csv.DictReader(fh)
        csv_rows = list(csv_reader)
    metrics["csv_row_count"] = len(csv_rows)
    if len(csv_rows) != len(records):
        errors.append(f"CSV row count ({len(csv_rows)}) does not match JSONL count ({len(records)}).")

    is_valid = len(errors) == 0
    return is_valid, metrics, errors


def main() -> None:
    is_valid, metrics, errors = run_validation()

    print("========================================")
    print("GOLDEN SET VALIDATION")
    print("========================================")
    print(f"Total examples: {metrics.get('total_records', 0)}")
    print(f"Unique conversations: {metrics.get('unique_conversations', 0)}")
    print(
        f"Intents covered: {metrics.get('canonical_intents_covered', 0)}/"
        f"{metrics.get('total_canonical_intents', 0)} canonical "
        f"({metrics.get('all_intents_covered', 0)} total)"
    )
    print(f"Duplicate examples: {metrics.get('duplicate_examples', 0)}")
    print(f"Duplicate conversations: {metrics.get('duplicate_conversations', 0)}")
    print(f"Missing required fields: {metrics.get('missing_required_fields', 0)}")
    print(f"Invalid intents: {metrics.get('invalid_intents', 0)}")
    print(f"Invalid difficulty labels: {metrics.get('invalid_difficulties', 0)}")
    print(f"Train/Val leakage: {metrics.get('train_val_leakage', 0)}")
    print("----------------------------------------")
    if is_valid:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")
        print("Validation Errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")
    print("========================================")

    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
