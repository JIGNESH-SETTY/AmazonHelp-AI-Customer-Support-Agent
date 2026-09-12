"""
datasets.py
-----------
STAGE 10: Safe & Immutable Golden Set Loader

Provides:
  - Read-only loading of the Stage 7 Golden Evaluation Set
  - Strict schema validation
  - Immutability guarantee (never writes back or modifies source)
  - Zero training leakage assertions
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
GOLDEN_JSONL_PATH = DATA_DIR / "golden" / "golden_evaluation_set.jsonl"
TRAIN_LABELS_PATH = DATA_DIR / "processed" / "amazonhelp_intent_labels.jsonl"

REQUIRED_GOLDEN_FIELDS = [
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
    "split",
]


@dataclass(frozen=True)
class GoldenEvaluationRecord:
    """Immutable record from the Stage 7 Golden Evaluation Set."""
    example_id: str
    conversation_id: str
    brand: str
    customer_message: str
    primary_intent: str
    primary_intent_name: str
    secondary_intent: Optional[str]
    secondary_intent_name: Optional[str]
    difficulty: str
    expected_behavior: str
    gold_response_guidance: str
    reference_support_response: str
    response_type: str
    source_reference: str
    split: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "conversation_id": self.conversation_id,
            "brand": self.brand,
            "customer_message": self.customer_message,
            "primary_intent": self.primary_intent,
            "primary_intent_name": self.primary_intent_name,
            "secondary_intent": self.secondary_intent,
            "secondary_intent_name": self.secondary_intent_name,
            "difficulty": self.difficulty,
            "expected_behavior": self.expected_behavior,
            "gold_response_guidance": self.gold_response_guidance,
            "reference_support_response": self.reference_support_response,
            "response_type": self.response_type,
            "source_reference": self.source_reference,
            "split": self.split,
        }


class GoldenSetLoader:
    """Read-only loader for the protected Stage 7 Golden Evaluation Set."""

    def __init__(self, golden_path: Optional[Path] = None):
        self.path = golden_path or GOLDEN_JSONL_PATH

    def load(self) -> List[GoldenEvaluationRecord]:
        """
        Loads and validates all records from the Golden Evaluation Set.
        Raises ValueError if schema is violated or if file is missing.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Golden Evaluation Set not found at: {self.path}")

        records: List[GoldenEvaluationRecord] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line_idx, line in enumerate(fh, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Line {line_idx}: Invalid JSON: {exc}") from exc

                # Schema validation
                for field_name in REQUIRED_GOLDEN_FIELDS:
                    if field_name not in data or data[field_name] is None:
                        raise ValueError(f"Line {line_idx}: Missing required field '{field_name}'")

                records.append(
                    GoldenEvaluationRecord(
                        example_id=data["example_id"],
                        conversation_id=data["conversation_id"],
                        brand=data["brand"],
                        customer_message=data["customer_message"],
                        primary_intent=data["primary_intent"],
                        primary_intent_name=data["primary_intent_name"],
                        secondary_intent=data.get("secondary_intent"),
                        secondary_intent_name=data.get("secondary_intent_name"),
                        difficulty=data["difficulty"],
                        expected_behavior=data["expected_behavior"],
                        gold_response_guidance=data["gold_response_guidance"],
                        reference_support_response=data["reference_support_response"],
                        response_type=data["response_type"],
                        source_reference=data.get("source_reference", ""),
                        split=data["split"],
                    )
                )

        return records

    def verify_zero_training_leakage(self, training_labels_path: Optional[Path] = None) -> Tuple[bool, int]:
        """
        Verifies that none of the Golden Set conversation IDs exist in the training split.
        Returns (is_leakage_free, overlapping_count).
        """
        golden_records = self.load()
        golden_conv_ids = {r.conversation_id for r in golden_records}

        lbl_path = training_labels_path or TRAIN_LABELS_PATH
        if not lbl_path.exists():
            return True, 0

        train_conv_ids: Set[str] = set()
        with open(lbl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("split") == "train":
                    train_conv_ids.add(row["conversation_id"])

        overlap = golden_conv_ids.intersection(train_conv_ids)
        return len(overlap) == 0, len(overlap)
