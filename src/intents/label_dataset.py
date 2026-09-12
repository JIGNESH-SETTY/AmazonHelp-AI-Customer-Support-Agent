"""
label_dataset.py
----------------
STAGE 6: Intent Labeling Pipeline

Labels all AmazonHelp interactions across Stage 4 train, validation, and test splits
with the canonical 10-intent taxonomy.

Key Properties:
  - Preserves Stage 4 conversation-level splits (zero leakage).
  - Keeps support response behavior (`response_type`) separate from customer intent.
  - Transparent deterministic multi-tier rule matching with explicit priority ordering.
  - Assigns ambiguous, low-information, or unclassifiable queries to `unknown_or_ambiguous`.
  - Outputs:
    `data/processed/amazonhelp_intent_labels.jsonl`
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Configure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "processed" / "splits"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DATA_DIR = REPO_ROOT / "data"

SELECTED_BRAND_PATH = DATA_DIR / "selected_brand.json"
TAXONOMY_JSON_PATH = DATA_DIR / "intent_taxonomy.json"
OUTPUT_LABELS_PATH = PROCESSED_DIR / "amazonhelp_intent_labels.jsonl"


def load_selected_brand() -> str:
    """Loads selected brand from data/selected_brand.json."""
    if not SELECTED_BRAND_PATH.exists():
        raise FileNotFoundError(f"Missing {SELECTED_BRAND_PATH}")
    with open(SELECTED_BRAND_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["selected_brand"]


def load_taxonomy() -> Dict[str, Any]:
    """Loads canonical taxonomy from data/intent_taxonomy.json."""
    if not TAXONOMY_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing {TAXONOMY_JSON_PATH}")
    with open(TAXONOMY_JSON_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------
def normalize_customer_text(text: str) -> str:
    """Normalizes text for robust intent matching."""
    if not text or not isinstance(text, str):
        return ""
    t = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Comprehensive Intent Pattern Rules
# ---------------------------------------------------------------------------
INTENT_PATTERNS = {
    "missing_delivered_package": re.compile(
        r"\b(says delivered|marked (?:as )?delivered|shows delivered|said delivered|delivered but|delivered today but|delivered Friday but|delivered yesterday but|don't have (?:it|the package|my package)|never received|didn't receive (?:it|my package)|nowhere to be found|driver left|front porch|front door|stolen|missing package|package.*was delivered.*haven't got)\b",
        re.I,
    ),
    "damaged_defective_item": re.compile(
        r"\b(damaged|broken|defective|faulty|cracked|shattered|smashed|scratched|dented|poor quality|wrong item|different item|missing parts|missing pieces|box was open|package arrived open|tampered with|leaking|expired)\b",
        re.I,
    ),
    "order_cancellation": re.compile(
        r"\b(cancel|canceling|cancelling|cancelled|cancellation|cancel (?:my )?order|stop (?:the )?order|accidental (?:order|purchase)|bought by mistake|ordered by mistake|change (?:the )?shipping address|change delivery address|don't want (?:it|this order) anymore)\b",
        re.I,
    ),
    "prime_membership": re.compile(
        r"\b(prime membership|prime member|pay(?:ing)? for prime|charged for prime|cancel prime|renew(?:al)? prime|prime subscription|free trial|prime video|prime student|prime fee|monthly prime|prime delivery)\b",
        re.I,
    ),
    "account_access_security": re.compile(
        r"\b(log\s*in|sign\s*in|locked out|password|reset password|verification code|otp|compromised|hacked|two factor|2fa|account access|email address|access my account|suspend(?:ed)?|on hold|unauthorized access)\b",
        re.I,
    ),
    "returns_and_refunds": re.compile(
        r"\b(refund|refunds|refunded|refunding|return|returns|returned|returning|send (?:it )?back|money back|replacement|replace|exchange|drop off|return label|return pickup|credited back|reimburse)\b",
        re.I,
    ),
    "payment_and_billing": re.compile(
        r"\b(charged|charge|charges|overcharged|double charge|unauthorized charge|credit card|debit card|bank account|billing|invoice|payment failed|declined|gift card|promo code|voucher|balance|payment method)\b",
        re.I,
    ),
    "digital_services_technical": re.compile(
        r"\b(kindle|paperwhite|fire\s*tv|firestick|fire\s*stick|alexa|echo|amazon app|website|loading|error code|crash|crashing|bug|glitch|download|ebook|e-book|stream|streaming|music app)\b",
        re.I,
    ),
    "delivery_delay": re.compile(
        r"\b(late|delay|delayed|still waiting|not arrived|haven't (?:received|got|arrived)|hasn't (?:arrived|come|delivered)|supposed to (?:arrive|deliver|be here)|where is my (?:order|package)|expected delivery|delivery date|running late|still in transit|due date|shipping status|tracking number|carrier|usps|ups|hermes|dpd|amzl|dispatch|shipped yet)\b",
        re.I,
    ),
    "service_complaint_escalation": re.compile(
        r"\b(worst customer service|terrible service|horrible service|rude|useless|speak to (?:a )?supervisor|manager|talk to a human|customer care number|phone number|call me|complaint|escalate|spoke to 3 different|giving run around|disgusted|unacceptable)\b",
        re.I,
    ),
}

# Explicit deterministic priority ordering for multi-intent resolution:
# High specificity intents take precedence over general delay or complaint
INTENT_PRIORITY_ORDER = [
    "missing_delivered_package",
    "damaged_defective_item",
    "order_cancellation",
    "prime_membership",
    "account_access_security",
    "returns_and_refunds",
    "payment_and_billing",
    "digital_services_technical",
    "delivery_delay",
    "service_complaint_escalation",
]


def classify_customer_intent(customer_text: str) -> Tuple[str, str]:
    """
    Classifies customer message into an intent_id and confidence level.
    Returns: (intent_id, confidence)
    """
    norm_text = normalize_customer_text(customer_text)
    if not norm_text or len(norm_text) < 5:
        return "unknown_or_ambiguous", "low_ambiguous"

    # Evaluate priority rules
    matched_intents = []
    for intent_id in INTENT_PRIORITY_ORDER:
        pat = INTENT_PATTERNS[intent_id]
        if pat.search(norm_text):
            matched_intents.append(intent_id)

    if not matched_intents:
        # Broad keywords fallback check
        words = set(re.findall(r"[a-z]{3,}", norm_text.lower()))
        if {"deliver", "delivered", "package", "order", "arriving", "delivery"} & words:
            return "delivery_delay", "medium"
        if {"refund", "return"} & words:
            return "returns_and_refunds", "medium"
        if {"cancel"} & words:
            return "order_cancellation", "medium"
        if {"prime"} & words:
            return "prime_membership", "medium"
        if {"broken", "damage"} & words:
            return "damaged_defective_item", "medium"
        return "unknown_or_ambiguous", "low_ambiguous"

    # Primary intent selected via deterministic priority rule
    primary_intent = matched_intents[0]
    confidence = "high" if len(matched_intents) == 1 else "medium"
    return primary_intent, confidence


def run_intent_labeling() -> Dict[str, Any]:
    """Runs the labeling pipeline across all Stage 4 train, val, and test split datasets."""
    t_start = time.time()
    brand_name = load_selected_brand()
    taxonomy_data = load_taxonomy()

    intent_names = {item["intent_id"]: item["name"] for item in taxonomy_data["intents"]}
    intent_names["unknown_or_ambiguous"] = taxonomy_data["fallback_category"]["name"]

    print("=" * 75)
    print(f"STAGE 6: Intent Labeling Pipeline for '{brand_name}'")
    print("=" * 75)

    splits = ["train", "val", "test"]
    dataset_types = ["resolution", "escalation", "clarification"]

    total_records = 0
    labeled_records: List[Dict[str, Any]] = []

    # Distribution tracking
    intent_counter: Counter = Counter()
    split_counter: Counter = Counter()
    resp_type_by_intent: Dict[str, Counter] = defaultdict(Counter)
    split_by_intent: Dict[str, Counter] = defaultdict(Counter)
    seen_example_ids: Set[str] = set()

    # Track conversations per split to verify zero leakage (Check 6)
    convs_per_split: Dict[str, Set[str]] = defaultdict(set)

    print("Processing Stage 4 splits...")
    with open(OUTPUT_LABELS_PATH, "w", encoding="utf-8") as out_fh:
        for split in splits:
            split_dir = SPLITS_DIR / split
            for ds_type in dataset_types:
                filename = f"amazon_{ds_type}_pairs_{split}.jsonl"
                filepath = split_dir / filename
                if not filepath.exists():
                    print(f"Warning: {filepath} does not exist, skipping.")
                    continue

                with open(filepath, encoding="utf-8") as in_fh:
                    for line in in_fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        total_records += 1

                        ex_id = rec["example_id"]
                        conv_id = rec["conversation_id"]
                        cust_text = rec.get("customer_message", "")
                        resp_type = rec.get("response_type", ds_type)

                        seen_example_ids.add(ex_id)
                        convs_per_split[split].add(conv_id)

                        # Skip completely empty customer text (Check 8)
                        if not cust_text or not cust_text.strip():
                            continue

                        intent_id, confidence = classify_customer_intent(cust_text)
                        intent_name = intent_names[intent_id]

                        intent_counter[intent_id] += 1
                        split_counter[split] += 1
                        resp_type_by_intent[intent_id][resp_type] += 1
                        split_by_intent[intent_id][split] += 1

                        output_item = {
                            "conversation_id": conv_id,
                            "example_id": ex_id,
                            "customer_text": cust_text,
                            "intent_id": intent_id,
                            "intent_name": intent_name,
                            "response_type": resp_type,
                            "split": split,
                            "confidence": confidence,
                        }
                        out_fh.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    elapsed = time.time() - t_start
    total_labeled = sum(intent_counter.values())

    print(f"\n✓ Labeled {total_labeled:,} customer interactions in {elapsed:.2f}s.")
    print(f"✓ Written to: {OUTPUT_LABELS_PATH.relative_to(REPO_ROOT)}")

    # Print Distribution Summary
    print("\n" + "=" * 75)
    print(f"{'Intent ID':<30} | {'Name':<30} | {'Count':>7} | {'Pct':>6}")
    print("=" * 75)
    for intent_id, cnt in intent_counter.most_common():
        name = intent_names.get(intent_id, intent_id)
        pct = (cnt / total_labeled) * 100 if total_labeled else 0.0
        print(f"{intent_id:<30} | {name:<30} | {cnt:>7,} | {pct:>5.1f}%")
    print("=" * 75)

    print("\nSplit Distribution:")
    for split_name in splits:
        print(f"  {split_name:<10}: {split_counter[split_name]:,} labeled records ({len(convs_per_split[split_name]):,} unique convs)")

    # Verify split disjointness (Check 6)
    train_convs = convs_per_split["train"]
    val_convs = convs_per_split["val"]
    test_convs = convs_per_split["test"]
    assert train_convs.isdisjoint(val_convs), "Leakage between train and val!"
    assert train_convs.isdisjoint(test_convs), "Leakage between train and test!"
    assert val_convs.isdisjoint(test_convs), "Leakage between val and test!"
    print("✓ Zero conversation leakage verified across train, val, and test splits.")

    return {
        "total_labeled": total_labeled,
        "intent_distribution": dict(intent_counter),
        "split_distribution": dict(split_counter),
        "resp_type_by_intent": {k: dict(v) for k, v in resp_type_by_intent.items()},
        "split_by_intent": {k: dict(v) for k, v in split_by_intent.items()},
        "elapsed_seconds": round(elapsed, 2),
    }


if __name__ == "__main__":
    run_intent_labeling()
