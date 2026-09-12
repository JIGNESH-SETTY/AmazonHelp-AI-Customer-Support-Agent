"""
build_golden_set.py
-------------------
STAGE 7: Golden Evaluation Set Builder

Constructs the curated, high-quality, trusted evaluation benchmark (200 examples)
from the AmazonHelp test split, covering all 10 Stage 6 canonical support intents
plus controlled ambiguous/edge cases.

Evaluation Dimensions:
  - Intent classification coverage and accuracy
  - Domain-grounded expected behavior
  - Gold response guidance (anti-hallucination semantic criteria)
  - Graded difficulty levels (easy, medium, hard)
  - Response type diversity (resolution, escalation, clarification)

Outputs:
  - `data/golden/golden_evaluation_set.jsonl`: Primary JSONL dataset.
  - `data/golden/golden_evaluation_set.csv`: Tabular CSV export.
  - `data/golden/golden_set_manifest.json`: Metadata, distributions, filtering audit.
"""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# UTF-8 stdout configuration for Windows environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
TEST_SPLITS_DIR = SPLITS_DIR / "test"
GOLDEN_DIR = DATA_DIR / "golden"

SELECTED_BRAND_PATH = DATA_DIR / "selected_brand.json"
TAXONOMY_PATH = DATA_DIR / "intent_taxonomy.json"
INTENT_LABELS_PATH = PROCESSED_DIR / "amazonhelp_intent_labels.jsonl"

GOLDEN_JSONL_PATH = GOLDEN_DIR / "golden_evaluation_set.jsonl"
GOLDEN_CSV_PATH = GOLDEN_DIR / "golden_evaluation_set.csv"
GOLDEN_MANIFEST_PATH = GOLDEN_DIR / "golden_set_manifest.json"

# Configurable constants
RANDOM_SEED = 42
TARGET_TOTAL = 200
TARGET_CANONICAL_PER_INTENT = 18  # 18 * 10 = 180
TARGET_AMBIGUOUS = 20             # 20 ambiguous / edge cases
MIN_CUSTOMER_TEXT_LEN = 25
MAX_JACCARD_SIMILARITY = 0.80

NOISE_PATTERNS = [
    re.compile(r"^\s*(dm\s+sent|sent\s+dm|check\s+dm|check\s+your\s+dm|sent\s+you\s+a\s+dm|replied\s+in\s+dm)\b", re.I),
    re.compile(r"^\s*(thanks|thank\s+you|ok|okay|sure|will\s+do|done)\b", re.I),
    re.compile(r"^\s*(@\w+\s*)+$", re.I),  # Only mentions
]

HIGH_FRUSTRATION_KEYWORDS = {
    "worst", "terrible", "furious", "unacceptable", "pathetic", "useless",
    "scam", "crooks", "lawyer", "police", "legal", "sue", "fraud", "disgusted",
    "horrible", "robbery", "stealing", "stole", "shameful", "ridiculous"
}

# Domain guidance mapping for each intent
INTENT_GUIDANCE_MAP: Dict[str, Dict[str, str]] = {
    "delivery_delay": {
        "expected_behavior": (
            "Acknowledge shipping delay with professional empathy, refrain from asserting unverified "
            "carrier arrival times without account access, and direct customer to tracking link or request secure order details."
        ),
        "gold_response_guidance": (
            "Must acknowledge late delivery; must NOT fabricate specific carrier dates/times; "
            "should provide order tracking self-service link or invite secure details via private message."
        ),
    },
    "missing_delivered_package": {
        "expected_behavior": (
            "Acknowledge delivery discrepancy, advise checking safe spots, mailroom, and neighbors, "
            "and offer carrier inquiry or replacement next steps if still not found."
        ),
        "gold_response_guidance": (
            "Must acknowledge package marked as delivered; advise checking porch, mailroom, or neighbors; "
            "refrain from guaranteeing immediate reshipment without verification."
        ),
    },
    "returns_and_refunds": {
        "expected_behavior": (
            "Explain return procedure and refund timeline clearly, provide link to Online Returns Center, "
            "and clarify that funds are returned to original payment method upon return processing."
        ),
        "gold_response_guidance": (
            "Guide customer to Online Returns Center; state standard refund processing window (typically 3-5 business days after receipt); "
            "avoid stating a refund has already posted."
        ),
    },
    "order_cancellation": {
        "expected_behavior": (
            "Assess order dispatch stage, direct customer to 'Your Orders' for immediate cancellation if not shipped, "
            "or explain return-upon-delivery workflow if already dispatched."
        ),
        "gold_response_guidance": (
            "Direct customer to Your Orders to check if cancellation is still possible; clarify that shipped orders cannot be cancelled directly and must be returned."
        ),
    },
    "damaged_defective_item": {
        "expected_behavior": (
            "Express genuine regret for damaged or defective item, reassure customer that replacement or refund is available, "
            "and guide them to Returns Center for a prepaid replacement label."
        ),
        "gold_response_guidance": (
            "Apologize for damaged condition; outline replacement or return steps; "
            "do not request the customer pay for return shipping on defective goods."
        ),
    },
    "prime_membership": {
        "expected_behavior": (
            "Clarify Prime membership terms, benefits, or renewal charge, guide to 'Manage Your Prime Membership' page, "
            "and explain refund eligibility for unused renewal periods."
        ),
        "gold_response_guidance": (
            "Provide clear directions to Manage Prime Membership; explain unused benefit refund policy; "
            "refrain from altering account settings without authenticated customer action."
        ),
    },
    "payment_and_billing": {
        "expected_behavior": (
            "Acknowledge billing or payment discrepancy, advise verifying pending authorizations with bank, "
            "and provide secure customer service billing link while warning against sharing sensitive card details publicly."
        ),
        "gold_response_guidance": (
            "Advise checking order invoice and pending bank charges; provide official payment support link; "
            "strictly warn against posting card or bank numbers on public social channels."
        ),
    },
    "account_access_security": {
        "expected_behavior": (
            "Provide secure password reset or Two-Step Verification recovery instructions, direct to official account recovery page, "
            "and emphasize account security."
        ),
        "gold_response_guidance": (
            "Direct user to official Password Assistance / Account Recovery portal; never ask for password or OTP in response; "
            "reassure user regarding account safety."
        ),
    },
    "digital_services_technical": {
        "expected_behavior": (
            "Identify digital device or streaming service (Kindle, Fire TV, Alexa, Prime Video), offer initial troubleshooting steps "
            "(restart device, check app update, verify network), and link to device support hub."
        ),
        "gold_response_guidance": (
            "Offer relevant first-line troubleshooting step for device/app; avoid promising unverified hardware replacements; "
            "point to official device help page."
        ),
    },
    "service_complaint_escalation": {
        "expected_behavior": (
            "Acknowledge customer frustration and poor experience with sincere de-escalation, avoid defensive arguments, "
            "and escalate to specialized support via private channel or callback."
        ),
        "gold_response_guidance": (
            "De-escalate with empathetic, non-defensive language; apologize for the prior negative experience; "
            "provide direct channel to speak with an escalated support specialist."
        ),
    },
    "unknown_or_ambiguous": {
        "expected_behavior": (
            "Identify that the customer message lacks specific order or issue details, refrain from guessing or hallucinating an issue, "
            "and ask a focused clarifying question."
        ),
        "gold_response_guidance": (
            "Ask clarifying questions to identify whether inquiry relates to an order, account, or digital service; "
            "do NOT invent problem details or claim an action was taken."
        ),
    },
}


def load_selected_brand() -> str:
    """Loads selected brand from data/selected_brand.json."""
    with open(SELECTED_BRAND_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["selected_brand"]


def load_intent_taxonomy() -> Tuple[Dict[str, str], Set[str]]:
    """Loads intent metadata from data/intent_taxonomy.json."""
    with open(TAXONOMY_PATH, encoding="utf-8") as fh:
        taxonomy = json.load(fh)
    id_to_name = {intent["intent_id"]: intent["name"] for intent in taxonomy["intents"]}
    fallback = taxonomy.get("fallback_category", {})
    if fallback:
        id_to_name[fallback["intent_id"]] = fallback["name"]
    return id_to_name, set(id_to_name.keys())


def tokenize(text: str) -> Set[str]:
    """Simple whitespace + punctuation tokenizer for Jaccard similarity."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = {w for w in cleaned.split() if len(w) > 2}
    return tokens


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Calculates Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    union_len = len(set_a.union(set_b))
    if union_len == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / union_len


def is_noise_or_boilerplate(text: str) -> bool:
    """Detects boilerplate noise like 'DM sent', 'check DM', purely handle mentions."""
    for pattern in NOISE_PATTERNS:
        if pattern.search(text.strip()):
            return True
    return False


def detect_secondary_intent(customer_text: str, primary_intent: str) -> Optional[str]:
    """Detects possible secondary intent when multiple issues are expressed."""
    lower = customer_text.lower()
    
    rules = [
        ("returns_and_refunds", [r"\b(refund|money back|return|send back)\b"]),
        ("order_cancellation", [r"\b(cancel|cancellation)\b"]),
        ("damaged_defective_item", [r"\b(damaged|broken|shattered|cracked|defective)\b"]),
        ("missing_delivered_package", [r"\b(marked delivered|says delivered|never arrived|stolen)\b"]),
        ("prime_membership", [r"\b(prime|membership|annual fee)\b"]),
        ("payment_and_billing", [r"\b(charged twice|double charge|gift card|billing|charged me)\b"]),
        ("service_complaint_escalation", [r"\b(supervisor|manager|complaint|worst service|lawyer|sue)\b"]),
    ]
    
    for intent_id, regexes in rules:
        if intent_id == primary_intent:
            continue
        for rgx in regexes:
            if re.search(rgx, lower):
                return intent_id
    return None


def assign_difficulty(
    customer_text: str,
    primary_intent: str,
    secondary_intent: Optional[str],
    response_type: str,
    confidence: str,
) -> str:
    """
    Deterministically assigns difficulty label:
      - hard: multi-intent conflict, high frustration/anger, escalation response, or ambiguous request posing hallucination risk.
      - medium: clarification needed, long query with multiple details, or medium confidence.
      - easy: clear single intent, high confidence, resolution response, no conflicting signals.
    """
    lower = customer_text.lower()
    tokens = set(re.findall(r"\b\w+\b", lower))

    has_frustration = bool(tokens.intersection(HIGH_FRUSTRATION_KEYWORDS))
    is_multi_intent = secondary_intent is not None
    is_ambiguous = primary_intent == "unknown_or_ambiguous"
    is_escalation = response_type == "escalation" or primary_intent == "service_complaint_escalation"

    if is_multi_intent or has_frustration or is_escalation or (is_ambiguous and len(customer_text) < 50):
        return "hard"
    
    if response_type == "clarification" or len(customer_text) > 180 or confidence == "medium":
        return "medium"

    return "easy"


def load_test_pairs(brand: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Loads candidate test pairs strictly from Stage 4 test split files.
    Cross-references with Stage 6 intent labels.
    """
    # 1. Load test intent labels
    test_labels: Dict[str, Dict[str, Any]] = {}
    with open(INTENT_LABELS_PATH, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("split") == "test":
                test_labels[row["example_id"]] = row

    print(f"[Stage 7] Loaded {len(test_labels):,} intent labels for test split.")

    # 2. Load test pairs
    candidate_pairs: List[Dict[str, Any]] = []
    stats = {
        "raw_test_pairs_read": 0,
        "matched_with_intent_label": 0,
        "rejected_length": 0,
        "rejected_noise": 0,
        "rejected_missing_response": 0,
    }

    test_files = [
        TEST_SPLITS_DIR / "amazon_resolution_pairs_test.jsonl",
        TEST_SPLITS_DIR / "amazon_clarification_pairs_test.jsonl",
        TEST_SPLITS_DIR / "amazon_escalation_pairs_test.jsonl",
    ]

    for test_file in test_files:
        if not test_file.exists():
            print(f"Warning: Test file {test_file} not found.")
            continue

        with open(test_file, encoding="utf-8") as fh:
            for line in fh:
                stats["raw_test_pairs_read"] += 1
                row = json.loads(line)
                ex_id = row.get("example_id")
                cust_msg = row.get("customer_message", "").strip()
                supp_resp = row.get("support_response", "").strip()

                if not cust_msg or len(cust_msg) < MIN_CUSTOMER_TEXT_LEN:
                    stats["rejected_length"] += 1
                    continue

                if is_noise_or_boilerplate(cust_msg):
                    stats["rejected_noise"] += 1
                    continue

                if not supp_resp:
                    stats["rejected_missing_response"] += 1
                    continue

                label_info = test_labels.get(ex_id)
                if not label_info:
                    continue

                stats["matched_with_intent_label"] += 1
                candidate_pairs.append({
                    "example_id": ex_id,
                    "conversation_id": row.get("conversation_id"),
                    "brand": brand,
                    "customer_message": cust_msg,
                    "customer_message_original": row.get("customer_message_original", cust_msg),
                    "support_response": supp_resp,
                    "support_response_original": row.get("support_response_original", supp_resp),
                    "response_type": row.get("response_type", label_info.get("response_type", "resolution")),
                    "primary_intent": label_info.get("intent_id", "unknown_or_ambiguous"),
                    "primary_intent_name": label_info.get("intent_name", "Unknown or Ambiguous"),
                    "confidence": label_info.get("confidence", "high"),
                    "source_reference": f"test/{test_file.name}#{ex_id}",
                    "split": "test",
                })

    print(f"[Stage 7] Candidate pool after quality filtering: {len(candidate_pairs):,} candidates.")
    return candidate_pairs, stats


def build_golden_dataset() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes stratified sampling with duplicate and near-duplicate protection.
    Returns golden examples and run statistics.
    """
    brand = load_selected_brand()
    intent_names, valid_intents = load_intent_taxonomy()

    candidates, filter_stats = load_test_pairs(brand)

    # Group candidates by intent
    by_intent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        by_intent[cand["primary_intent"]].append(cand)

    # Deterministic sampling
    rng = random.Random(RANDOM_SEED)

    selected_examples: List[Dict[str, Any]] = []
    selected_conv_ids: Set[str] = set()
    selected_texts: Set[str] = set()
    intent_token_sets: Dict[str, List[Set[str]]] = defaultdict(list)

    duplicate_stats = {
        "exact_duplicates_rejected": 0,
        "near_duplicates_rejected": 0,
        "conv_id_collisions_rejected": 0,
    }

    canonical_intents = [
        "delivery_delay",
        "missing_delivered_package",
        "returns_and_refunds",
        "damaged_defective_item",
        "order_cancellation",
        "payment_and_billing",
        "prime_membership",
        "account_access_security",
        "digital_services_technical",
        "service_complaint_escalation",
    ]

    # 1. Sample canonical intents
    for intent_id in canonical_intents:
        pool = by_intent.get(intent_id, [])
        rng.shuffle(pool)
        
        target = TARGET_CANONICAL_PER_INTENT
        count_for_intent = 0

        for cand in pool:
            if count_for_intent >= target:
                break

            conv_id = cand["conversation_id"]
            cust_text = cand["customer_message"]

            if conv_id in selected_conv_ids:
                duplicate_stats["conv_id_collisions_rejected"] += 1
                continue

            if cust_text.lower() in selected_texts:
                duplicate_stats["exact_duplicates_rejected"] += 1
                continue

            toks = tokenize(cust_text)
            is_near_dup = False
            for existing_toks in intent_token_sets[intent_id]:
                sim = jaccard_similarity(toks, existing_toks)
                if sim >= MAX_JACCARD_SIMILARITY:
                    is_near_dup = True
                    break

            if is_near_dup:
                duplicate_stats["near_duplicates_rejected"] += 1
                continue

            # Accept candidate
            selected_conv_ids.add(conv_id)
            selected_texts.add(cust_text.lower())
            intent_token_sets[intent_id].append(toks)
            
            sec_intent = detect_secondary_intent(cust_text, intent_id)
            diff = assign_difficulty(
                cust_text, intent_id, sec_intent, cand["response_type"], cand["confidence"]
            )
            guidance = INTENT_GUIDANCE_MAP.get(intent_id, INTENT_GUIDANCE_MAP["unknown_or_ambiguous"])

            golden_record = {
                "example_id": f"golden_{len(selected_examples) + 1:03d}",
                "conversation_id": conv_id,
                "brand": brand,
                "customer_message": cust_text,
                "primary_intent": intent_id,
                "primary_intent_name": intent_names.get(intent_id, intent_id),
                "secondary_intent": sec_intent,
                "secondary_intent_name": intent_names.get(sec_intent) if sec_intent else None,
                "difficulty": diff,
                "expected_behavior": guidance["expected_behavior"],
                "gold_response_guidance": guidance["gold_response_guidance"],
                "reference_support_response": cand["support_response"],
                "response_type": cand["response_type"],
                "source_reference": cand["source_reference"],
                "split": "test",
            }
            selected_examples.append(golden_record)
            count_for_intent += 1

        print(f"[Stage 7] Intent '{intent_id}': sampled {count_for_intent}/{target} examples.")

    # 2. Sample ambiguous / edge cases
    ambiguous_pool = by_intent.get("unknown_or_ambiguous", [])
    rng.shuffle(ambiguous_pool)
    ambiguous_target = TARGET_AMBIGUOUS
    ambiguous_count = 0

    for cand in ambiguous_pool:
        if ambiguous_count >= ambiguous_target:
            break

        conv_id = cand["conversation_id"]
        cust_text = cand["customer_message"]

        if conv_id in selected_conv_ids:
            duplicate_stats["conv_id_collisions_rejected"] += 1
            continue

        if cust_text.lower() in selected_texts:
            duplicate_stats["exact_duplicates_rejected"] += 1
            continue

        toks = tokenize(cust_text)
        is_near_dup = False
        for existing_toks in intent_token_sets["unknown_or_ambiguous"]:
            sim = jaccard_similarity(toks, existing_toks)
            if sim >= MAX_JACCARD_SIMILARITY:
                is_near_dup = True
                break

        if is_near_dup:
            duplicate_stats["near_duplicates_rejected"] += 1
            continue

        selected_conv_ids.add(conv_id)
        selected_texts.add(cust_text.lower())
        intent_token_sets["unknown_or_ambiguous"].append(toks)

        sec_intent = detect_secondary_intent(cust_text, "unknown_or_ambiguous")
        diff = assign_difficulty(
            cust_text, "unknown_or_ambiguous", sec_intent, cand["response_type"], cand["confidence"]
        )
        guidance = INTENT_GUIDANCE_MAP["unknown_or_ambiguous"]

        golden_record = {
            "example_id": f"golden_{len(selected_examples) + 1:03d}",
            "conversation_id": conv_id,
            "brand": brand,
            "customer_message": cust_text,
            "primary_intent": "unknown_or_ambiguous",
            "primary_intent_name": intent_names.get("unknown_or_ambiguous", "Unknown or Ambiguous"),
            "secondary_intent": sec_intent,
            "secondary_intent_name": intent_names.get(sec_intent) if sec_intent else None,
            "difficulty": diff,
            "expected_behavior": guidance["expected_behavior"],
            "gold_response_guidance": guidance["gold_response_guidance"],
            "reference_support_response": cand["support_response"],
            "response_type": cand["response_type"],
            "source_reference": cand["source_reference"],
            "split": "test",
        }
        selected_examples.append(golden_record)
        ambiguous_count += 1

    print(f"[Stage 7] Sampled {ambiguous_count}/{ambiguous_target} ambiguous / edge case examples.")
    print(f"[Stage 7] Total Golden Evaluation Set size: {len(selected_examples)} examples.")

    manifest_metadata = {
        "stage": 7,
        "name": "AmazonHelp Golden Evaluation Set",
        "schema_version": "1.0",
        "brand": brand,
        "creation_timestamp": "2026-09-12 06:45:00 UTC",
        "random_seed": RANDOM_SEED,
        "target_size": TARGET_TOTAL,
        "actual_size": len(selected_examples),
        "source_split": "Stage 4 test split (zero leakage from train/validation)",
        "intents_covered_count": len({ex["primary_intent"] for ex in selected_examples}),
        "canonical_intents_covered": len([
            k for k in canonical_intents if any(ex["primary_intent"] == k for ex in selected_examples)
        ]),
        "total_canonical_intents": len(canonical_intents),
        "intent_distribution": dict(Counter(ex["primary_intent"] for ex in selected_examples)),
        "difficulty_distribution": dict(Counter(ex["difficulty"] for ex in selected_examples)),
        "response_type_distribution": dict(Counter(ex["response_type"] for ex in selected_examples)),
        "secondary_intent_count": sum(1 for ex in selected_examples if ex["secondary_intent"] is not None),
        "filter_statistics": filter_stats,
        "duplicate_statistics": duplicate_stats,
    }

    return selected_examples, manifest_metadata


def save_golden_artifacts(examples: List[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    """Saves JSONL, CSV, and manifest artifacts into data/golden/."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save JSONL
    with open(GOLDEN_JSONL_PATH, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[Stage 7] Saved JSONL: {GOLDEN_JSONL_PATH} ({len(examples)} rows)")

    # 2. Save CSV
    fieldnames = [
        "example_id",
        "conversation_id",
        "brand",
        "primary_intent",
        "primary_intent_name",
        "secondary_intent",
        "secondary_intent_name",
        "difficulty",
        "response_type",
        "customer_message",
        "expected_behavior",
        "gold_response_guidance",
        "reference_support_response",
        "source_reference",
        "split",
    ]
    with open(GOLDEN_CSV_PATH, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ex in examples:
            writer.writerow(ex)
    print(f"[Stage 7] Saved CSV: {GOLDEN_CSV_PATH}")

    # 3. Save Manifest
    with open(GOLDEN_MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[Stage 7] Saved Manifest: {GOLDEN_MANIFEST_PATH}")


def main() -> None:
    print("==================================================")
    print("STAGE 7: Building AmazonHelp Golden Evaluation Set")
    print("==================================================")
    examples, manifest = build_golden_dataset()
    save_golden_artifacts(examples, manifest)
    print("==================================================")
    print("Golden Evaluation Set generation completed successfully.")
    print("==================================================")


if __name__ == "__main__":
    main()
