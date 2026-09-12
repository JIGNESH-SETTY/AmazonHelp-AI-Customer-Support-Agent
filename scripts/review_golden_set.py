"""
scripts/review_golden_set.py
----------------------------
STAGE 15A: Manual Golden Evaluation Set Review Helper

Provides a fast, batch-oriented interactive workflow for manual review
of the 200-example Golden Evaluation Set.

Features:
  - Loads the canonical 11-intent taxonomy directly from data/intent_taxonomy.json.
  - Fast batch review mode: displays 10 pending examples at once with full text and taxonomy.
  - Compact batch input syntax:
      * '1-10'        -> Confirm all 10 displayed labels as verified
      * '1,3,5,7'     -> Confirm selected examples
      * '3=11'        -> Change example 3 to intent 11
      * '2=10,5=3'    -> Change multiple examples
      * '1-6,8-10;7=5'-> Mixed confirmation + corrections
      * '7=5#note'    -> Optional reviewer notes via '#'
      * 's'           -> Skip current batch
      * 'q'           -> Safe save and quit
  - Real-time progress reporting (reviewed count, pending count, % complete).
  - Safe persistence: auto-saves to CSV safely using atomic write pattern.
  - Strict validation: ensures verified intents belong to the valid taxonomy.
  - Resume support: preserves existing reviewed rows and continues from first pending row.
  - Single-step mode still available via --single-step.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

# Windows console UTF-8 configuration
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
GOLDEN_DIR = DATA_DIR / "golden"
DEFAULT_MANIFEST_PATH = GOLDEN_DIR / "golden_human_review_manifest.csv"
DEFAULT_TAXONOMY_PATH = DATA_DIR / "intent_taxonomy.json"

REQUIRED_COLUMNS = [
    "example_id",
    "customer_message",
    "current_intent",
    "human_verified_intent",
    "expected_behavior",
    "gold_guidance",
    "human_review_status",
    "reviewer_notes",
]


def load_taxonomy_intents(taxonomy_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Loads canonical and fallback intents from the intent taxonomy JSON artifact.
    Returns list of dicts with 'intent_id', 'name', and 'definition'.
    """
    path = taxonomy_path or DEFAULT_TAXONOMY_PATH
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    intents: List[Dict[str, str]] = []
    for item in data.get("intents", []):
        intents.append({
            "intent_id": item["intent_id"],
            "name": item.get("name", item["intent_id"]),
            "definition": item.get("definition", "").strip(),
        })

    fallback = data.get("fallback_category")
    if fallback:
        intents.append({
            "intent_id": fallback["intent_id"],
            "name": fallback.get("name", fallback["intent_id"]),
            "definition": fallback.get("definition", "").strip(),
        })

    return intents


def load_manifest(manifest_path: Optional[Path] = None) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Loads human review manifest CSV.
    Returns (fieldnames, rows).
    """
    path = manifest_path or DEFAULT_MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    return fieldnames, rows


def save_manifest(
    manifest_path: Path,
    fieldnames: List[str],
    rows: List[Dict[str, str]],
) -> None:
    """
    Safely writes manifest rows to CSV using atomic rename pattern.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = manifest_path.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Atomic replace
    os.replace(temp_file, manifest_path)


def get_pending_and_reviewed_counts(rows: List[Dict[str, str]]) -> Tuple[int, int]:
    """
    Returns (reviewed_count, pending_count).
    A row is reviewed if human_review_status == 'VERIFIED' and human_verified_intent is non-empty.
    """
    reviewed = sum(
        1 for r in rows
        if r.get("human_review_status", "").strip() == "VERIFIED"
        and bool(r.get("human_verified_intent", "").strip())
    )
    pending = len(rows) - reviewed
    return reviewed, pending


def validate_manifest_state(
    rows: List[Dict[str, str]],
    valid_intent_ids: Set[str],
) -> Dict[str, Any]:
    """
    Validates manifest rows:
      - Valid example IDs
      - No duplicate example IDs
      - Customer messages and current intents present
      - Verified intents must belong to valid taxonomy
      - Distinction between VERIFIED and PENDING_REVIEW rows
    """
    errors: List[str] = []
    seen_ids: Set[str] = set()

    for idx, r in enumerate(rows, start=1):
        eid = r.get("example_id", "").strip()
        if not eid:
            errors.append(f"Row {idx}: missing example_id")
        elif eid in seen_ids:
            errors.append(f"Row {idx}: duplicate example_id '{eid}'")
        else:
            seen_ids.add(eid)

        msg = r.get("customer_message", "").strip()
        if not msg:
            errors.append(f"Row {idx} ({eid}): missing customer_message")

        curr_intent = r.get("current_intent", "").strip()
        if not curr_intent:
            errors.append(f"Row {idx} ({eid}): missing current_intent")
        elif curr_intent not in valid_intent_ids:
            errors.append(f"Row {idx} ({eid}): invalid current_intent '{curr_intent}'")

        status = r.get("human_review_status", "").strip()
        verified_intent = r.get("human_verified_intent", "").strip()

        if status == "VERIFIED":
            if not verified_intent:
                errors.append(f"Row {idx} ({eid}): status is VERIFIED but human_verified_intent is empty")
            elif verified_intent not in valid_intent_ids:
                errors.append(f"Row {idx} ({eid}): human_verified_intent '{verified_intent}' not in taxonomy")
        elif status == "PENDING_REVIEW":
            if verified_intent:
                errors.append(f"Row {idx} ({eid}): status is PENDING_REVIEW but human_verified_intent is set to '{verified_intent}'")
        else:
            errors.append(f"Row {idx} ({eid}): unknown human_review_status '{status}'")

    reviewed_cnt, pending_cnt = get_pending_and_reviewed_counts(rows)
    return {
        "is_valid": len(errors) == 0,
        "total_rows": len(rows),
        "reviewed_count": reviewed_cnt,
        "pending_count": pending_cnt,
        "errors": errors,
    }


def parse_compact_batch_input(
    command: str,
    batch_len: int,
    taxonomy: List[Dict[str, str]],
) -> Tuple[Dict[int, Tuple[str, str]], List[str]]:
    """
    Parses compact reviewer command for a batch of batch_len displayed examples.

    Supported syntax:
      - '1-10'         -> Confirms all 10 displayed items as CURRENT
      - '1,3,5,7'      -> Confirms selected items as CURRENT
      - '3=11'         -> Changes item 3 to intent 11
      - '2=10,5=3'     -> Changes multiple items
      - '1-6,8-10;7=5' -> Confirms 1-6 and 8-10, changes 7 to intent 5
      - '7=5#note'     -> Assigns intent 5 with reviewer note
      - 'all' or 'c'   -> Confirms all items as CURRENT
      - 'q' or 'quit'  -> Exits session safely
      - 's' or 'skip'  -> Skips current batch

    Returns:
      (decisions, errors)
      decisions: {1-based_item_idx: (target_intent_or_CURRENT, note)}
      errors: List of error strings (empty if valid)
    """
    raw = command.strip()
    if not raw:
        return {}, ["Empty input. Enter a command (e.g. '1-10', '1-6,8-10;7=5', 's' to skip, 'q' to quit)."]

    if raw.lower() in ("q", "quit", "exit"):
        return {}, ["QUIT"]
    if raw.lower() in ("s", "skip"):
        return {}, ["SKIP"]

    valid_by_num: Dict[str, str] = {str(i): item["intent_id"] for i, item in enumerate(taxonomy, start=1)}
    valid_ids: Set[str] = {item["intent_id"] for item in taxonomy}

    if raw.lower() in ("all", "c", "confirm"):
        return {i: ("CURRENT", "") for i in range(1, batch_len + 1)}, []

    decisions: Dict[int, Tuple[str, str]] = {}
    errors: List[str] = []

    # Split into clauses by semicolon
    clauses = [c.strip() for c in raw.split(";") if c.strip()]
    subtokens: List[Tuple[str, str]] = []

    for clause in clauses:
        note = ""
        if "#" in clause:
            main_part, note = clause.split("#", 1)
            main_part = main_part.strip()
            note = note.strip()
        else:
            main_part = clause

        items = [x.strip() for x in main_part.split(",") if x.strip()]
        for idx_item, itm in enumerate(items):
            # Apply note to the specific item if present
            if idx_item == len(items) - 1:
                subtokens.append((itm, note))
            else:
                subtokens.append((itm, ""))

    for token, note in subtokens:
        if "=" in token:
            parts = token.split("=", 1)
            item_str = parts[0].strip()
            intent_str = parts[1].strip()

            if not item_str.isdigit():
                errors.append(f"Invalid item index '{item_str}' in '{token}'. Expected a number 1–{batch_len}.")
                continue

            item_idx = int(item_str)
            if item_idx < 1 or item_idx > batch_len:
                errors.append(f"Item index {item_idx} out of range (must be 1–{batch_len}).")
                continue

            resolved_intent: Optional[str] = None
            if intent_str in valid_by_num:
                resolved_intent = valid_by_num[intent_str]
            elif intent_str in valid_ids:
                resolved_intent = intent_str
            else:
                errors.append(f"Invalid intent '{intent_str}' for item {item_idx}. Must be 1–{len(taxonomy)} or valid intent ID.")
                continue

            decisions[item_idx] = (resolved_intent, note)

        elif "-" in token:
            parts = token.split("-")
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                s = int(parts[0].strip())
                e = int(parts[1].strip())
                if s < 1 or e > batch_len or s > e:
                    errors.append(f"Invalid range '{token}'. Must be 1–{batch_len} with start <= end.")
                    continue
                for i in range(s, e + 1):
                    decisions[i] = ("CURRENT", note)
            else:
                errors.append(f"Malformed range expression '{token}'. Example: '1-6' or '8-10'.")

        elif token.isdigit():
            idx = int(token)
            if idx < 1 or idx > batch_len:
                errors.append(f"Item index {idx} out of range (must be 1–{batch_len}).")
                continue
            decisions[idx] = ("CURRENT", note)

        else:
            errors.append(f"Unrecognized token '{token}'. Expected range ('1-10'), index ('3'), or assignment ('7=5').")

    if errors:
        return {}, errors

    if not decisions:
        return {}, ["No valid item decisions specified."]

    return decisions, []


def apply_batch_decisions(
    rows: List[Dict[str, str]],
    batch_indices: List[int],
    decisions: Dict[int, Tuple[str, str]],
) -> int:
    """
    Applies parsed decisions to manifest rows.
    Returns number of updated rows.
    """
    applied = 0
    for item_idx, (target_intent, note) in decisions.items():
        if 1 <= item_idx <= len(batch_indices):
            row_idx = batch_indices[item_idx - 1]
            r = rows[row_idx]

            if target_intent == "CURRENT":
                r["human_verified_intent"] = r["current_intent"]
            else:
                r["human_verified_intent"] = target_intent

            r["human_review_status"] = "VERIFIED"
            if note:
                r["reviewer_notes"] = note

            applied += 1

    return applied


def display_taxonomy_reference(taxonomy: List[Dict[str, str]]) -> None:
    """Prints compact taxonomy reference table."""
    print("================================================================================")
    print("VALID INTENT TAXONOMY REFERENCE (11 INTENTS):")
    print("--------------------------------------------------------------------------------")
    col1 = taxonomy[:6]
    col2 = taxonomy[6:]
    max_len = max(len(col1), len(col2))
    for i in range(max_len):
        t1 = f" [{i+1:>2}] {col1[i]['intent_id']}" if i < len(col1) else ""
        t2 = f" [{i+7:>2}] {col2[i]['intent_id']}" if i < len(col2) else ""
        print(f"  {t1:<38} {t2}")
    print("================================================================================")


def review_fast_batch(
    manifest_path: Path,
    taxonomy_path: Path,
    batch_size: int = 10,
) -> None:
    """Runs high-efficiency batch review workflow."""
    taxonomy = load_taxonomy_intents(taxonomy_path)
    valid_intents = {item["intent_id"] for item in taxonomy}

    fieldnames, rows = load_manifest(manifest_path)
    val_res = validate_manifest_state(rows, valid_intents)
    if not val_res["is_valid"]:
        print("Manifest validation errors found before starting:")
        for err in val_res["errors"][:10]:
            print(f"  [ERROR] {err}")
        return

    reviewed_cnt, pending_cnt = get_pending_and_reviewed_counts(rows)
    total_rows = len(rows)

    print("================================================================================")
    print("STAGE 15A: FAST BATCH MANUAL REVIEW HELPER")
    print("================================================================================")
    print(f"Manifest Path: {manifest_path.relative_to(REPO_ROOT) if manifest_path.is_relative_to(REPO_ROOT) else manifest_path}")
    print(f"Total Examples: {total_rows}")
    print(f"Already Reviewed: {reviewed_cnt} ({reviewed_cnt/total_rows:.1%})")
    print(f"Pending Review:   {pending_cnt} ({pending_cnt/total_rows:.1%})")
    print(f"Batch Size:       {batch_size}")
    print("================================================================================")

    if pending_cnt == 0:
        print("\nAll 200 examples in the manifest are already marked as VERIFIED!")
        return

    try:
        while True:
            # Recompute pending indices at each iteration for clean resume
            pending_indices = [
                i for i, r in enumerate(rows)
                if r.get("human_review_status", "").strip() != "VERIFIED"
                or not r.get("human_verified_intent", "").strip()
            ]

            if not pending_indices:
                print("\n================================================================================")
                print("✓ ALL 200 GOLDEN EXAMPLES HAVE BEEN VERIFIED!")
                print("================================================================================")
                break

            batch_slice = pending_indices[:batch_size]
            cur_reviewed, cur_pending = get_pending_and_reviewed_counts(rows)
            pct = (cur_reviewed / total_rows) * 100.0

            print("\n")
            display_taxonomy_reference(taxonomy)
            print(f"\n>>> CURRENT BATCH: {len(batch_slice)} PENDING EXAMPLES | {cur_reviewed}/{total_rows} REVIEWED ({pct:.1f}%) | {cur_pending} PENDING <<<")
            print("--------------------------------------------------------------------------------")

            for item_num, row_idx in enumerate(batch_slice, start=1):
                r = rows[row_idx]
                eid = r["example_id"]
                curr_intent = r["current_intent"]
                msg = r["customer_message"].replace("\n", " ").strip()
                print(f" [{item_num:>2}] {eid} | Current Intent: {curr_intent}")
                print(f"      \"{msg}\"")

            print("--------------------------------------------------------------------------------")
            print("Input Syntax:")
            print("  1-10           -> Confirm all 10 displayed intents as verified")
            print("  1,3,5,7        -> Confirm only selected examples")
            print("  3=11           -> Change example 3 to intent 11 (unknown_or_ambiguous)")
            print("  2=10,5=3       -> Change multiple examples")
            print("  1-6,8-10;7=5   -> Confirm 1-6 and 8-10, change 7 to intent 5")
            print("  7=5#note       -> Optional reviewer note using '#' (blank notes by default)")
            print("  s              -> Skip this batch")
            print("  q              -> Safe save & exit")
            print("--------------------------------------------------------------------------------")

            while True:
                user_cmd = input("Batch decision: ").strip()
                decisions, errors = parse_compact_batch_input(user_cmd, len(batch_slice), taxonomy)

                if errors == ["QUIT"]:
                    print("\nSaving progress and quitting...")
                    save_manifest(manifest_path, fieldnames, rows)
                    rev, pend = get_pending_and_reviewed_counts(rows)
                    print(f"Progress saved: {rev}/{total_rows} reviewed ({(rev/total_rows):.1%}) | {pend} pending.")
                    return

                if errors == ["SKIP"]:
                    print("Skipping batch. Moving to next examples...")
                    # Temporarily rotate slice to end of pending_indices for this session
                    pending_indices = pending_indices[batch_size:] + pending_indices[:batch_size]
                    break

                if errors:
                    print("  [ERROR(S)]")
                    for err in errors:
                        print(f"    - {err}")
                    print("  Please enter a valid command or 'q' to quit.")
                    continue

                # Apply valid decisions
                applied = apply_batch_decisions(rows, batch_slice, decisions)
                save_manifest(manifest_path, fieldnames, rows)

                rev, pend = get_pending_and_reviewed_counts(rows)
                pct_done = (rev / total_rows) * 100.0
                print(f"\n✓ Verified {applied} example(s) in this batch.")
                print(f"Progress: {rev} reviewed / {total_rows} total ({pct_done:.1f}% complete) | {pend} pending.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving manifest safely...")
        save_manifest(manifest_path, fieldnames, rows)
        rev, pend = get_pending_and_reviewed_counts(rows)
        print(f"Progress safely preserved: {rev}/{total_rows} reviewed | {pend} pending.")
        sys.exit(0)


def review_single_step(
    manifest_path: Path,
    taxonomy_path: Path,
    batch_size: int = 10,
) -> None:
    """Runs legacy single-step review mode."""
    taxonomy = load_taxonomy_intents(taxonomy_path)
    valid_intents = {item["intent_id"] for item in taxonomy}
    intent_by_number = {str(i): item["intent_id"] for i, item in enumerate(taxonomy, start=1)}

    fieldnames, rows = load_manifest(manifest_path)
    val_res = validate_manifest_state(rows, valid_intents)
    if not val_res["is_valid"]:
        print("Manifest validation errors found before starting:")
        for err in val_res["errors"][:10]:
            print(f"  [ERROR] {err}")
        return

    reviewed_cnt, pending_cnt = get_pending_and_reviewed_counts(rows)
    total_rows = len(rows)

    if pending_cnt == 0:
        print("\nAll 200 examples in the manifest are already marked as VERIFIED!")
        return

    display_taxonomy_reference(taxonomy)

    pending_indices = [
        i for i, r in enumerate(rows)
        if r.get("human_review_status", "").strip() != "VERIFIED"
        or not r.get("human_verified_intent", "").strip()
    ]

    try:
        for idx_in_pending, row_idx in enumerate(pending_indices, start=1):
            r = rows[row_idx]
            eid = r["example_id"]
            msg = r["customer_message"]
            curr_intent = r["current_intent"]

            print("-" * 80)
            print(f"Item {idx_in_pending}/{len(pending_indices)} | Example ID: {eid}")
            print(f"Customer Message: \"{msg}\"")
            print(f"Current Intent:   {curr_intent}")
            print("-" * 80)
            print("Options: 1-11 | 'c' to confirm current | intent_id | 's' to skip | 'q' to quit")

            while True:
                choice = input("Verified Intent: ").strip().lower()

                if choice == "q":
                    print("\nSaving progress and quitting...")
                    save_manifest(manifest_path, fieldnames, rows)
                    rev, pend = get_pending_and_reviewed_counts(rows)
                    print(f"Progress saved: {rev}/{total_rows} reviewed | {pend} pending.")
                    return

                if choice == "s":
                    print(f"Skipping {eid} for now.")
                    break

                selected_intent: Optional[str] = None
                if choice in intent_by_number:
                    selected_intent = intent_by_number[choice]
                elif choice in ("c", "current"):
                    selected_intent = curr_intent
                elif choice in valid_intents:
                    selected_intent = choice

                if selected_intent and selected_intent in valid_intents:
                    notes = input("Reviewer notes (optional, press Enter to skip): ").strip()
                    r["human_verified_intent"] = selected_intent
                    r["human_review_status"] = "VERIFIED"
                    if notes:
                        r["reviewer_notes"] = notes

                    save_manifest(manifest_path, fieldnames, rows)
                    rev, pend = get_pending_and_reviewed_counts(rows)
                    print(f"✓ Saved {eid} -> '{selected_intent}' | Progress: {rev}/{total_rows} ({(rev/total_rows):.1%})")
                    break
                else:
                    print("Invalid input. Enter 1-11, 'c', intent_id, 's', or 'q'.")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving manifest safely...")
        save_manifest(manifest_path, fieldnames, rows)
        rev, pend = get_pending_and_reviewed_counts(rows)
        print(f"Progress safely preserved: {rev}/{total_rows} reviewed | {pend} pending.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Stage 15A Manual Golden Set Review Helper")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to human review manifest CSV (default: data/golden/golden_human_review_manifest.csv)",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
        help="Path to intent taxonomy JSON (default: data/intent_taxonomy.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of examples per review batch (default: 10)",
    )
    parser.add_argument(
        "--fast-batch",
        action="store_true",
        help="Run high-efficiency fast batch review mode (default)",
    )
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="Run step-by-step single example review mode",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check manifest status without starting interactive review",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate manifest integrity and taxonomy adherence",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy_intents(args.taxonomy_path)
    valid_intents = {item["intent_id"] for item in taxonomy}
    _, rows = load_manifest(args.manifest_path)
    val_res = validate_manifest_state(rows, valid_intents)

    if args.validate or args.status:
        total_rows = val_res["total_rows"]
        rev = val_res["reviewed_count"]
        pend = val_res["pending_count"]
        pct = (rev / total_rows) * 100.0 if total_rows else 0.0

        print("==================================================")
        print("GOLDEN SET REVIEW MANIFEST STATUS")
        print("==================================================")
        print(f"Manifest File:  {args.manifest_path.name}")
        print(f"Total Rows:     {total_rows}")
        print(f"Reviewed:       {rev} ({pct:.1f}%)")
        print(f"Pending:        {pend} ({(100.0 - pct):.1f}%)")
        print(f"Valid Taxonomy: {len(valid_intents)} intents")
        print(f"Status:         {'VALID' if val_res['is_valid'] else 'ERRORS FOUND'}")
        if not val_res["is_valid"]:
            print("\nValidation Errors:")
            for e in val_res["errors"]:
                print(f"  - {e}")
            sys.exit(1)
        print("==================================================")
        return

    if args.single_step:
        review_single_step(
            manifest_path=args.manifest_path,
            taxonomy_path=args.taxonomy_path,
            batch_size=args.batch_size,
        )
    else:
        review_fast_batch(
            manifest_path=args.manifest_path,
            taxonomy_path=args.taxonomy_path,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
