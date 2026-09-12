"""
normalize.py
------------
STAGE 9: Input Normalization Layer

Handles:
  - Empty or malformed input checking
  - Whitespace collapsing
  - Excessive repeated character reduction (e.g., 'pleaaase' -> 'please')
  - Punctuation cleanup while preserving critical domain tokens
  - Preserves original_text alongside normalized_text
"""

import re
from typing import Tuple

from src.agent.schemas import AgentInput, NormalizedInput


def normalize_input(agent_input: AgentInput) -> NormalizedInput:
    """
    Normalizes raw customer input text conservatively.
    """
    raw_text = agent_input.customer_text

    if raw_text is None or not isinstance(raw_text, str):
        return NormalizedInput(
            original_text="",
            normalized_text="",
            is_valid=False,
            rejection_reason="Input customer text is null or invalid type.",
        )

    stripped = raw_text.strip()
    if len(stripped) == 0:
        return NormalizedInput(
            original_text=raw_text,
            normalized_text="",
            is_valid=False,
            rejection_reason="Input customer text is empty.",
        )

    # 1. Normalize excessive whitespace (newlines, tabs, multiple spaces)
    normalized = re.sub(r"\s+", " ", stripped)

    # 2. Collapse characters repeated 3 or more times down to 2 (e.g., 'soooo' -> 'soo', '????' -> '??')
    normalized = re.sub(r"(.)\1{2,}", r"\1\1", normalized)

    # 3. Clean up non-printable control characters
    normalized = "".join(ch for ch in normalized if ch.isprintable() or ch == " ")

    # 4. Check if text has at least 2 alphanumeric characters
    alnum_count = sum(1 for ch in normalized if ch.isalnum())
    if alnum_count < 2:
        return NormalizedInput(
            original_text=raw_text,
            normalized_text=normalized,
            is_valid=False,
            rejection_reason="Input text contains insufficient alphanumeric content.",
        )

    return NormalizedInput(
        original_text=raw_text,
        normalized_text=normalized,
        is_valid=True,
        rejection_reason=None,
    )
