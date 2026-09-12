"""
policy.py
---------
STAGE 9: Policy Grounding & Safety Guardrail Engine

Prevents:
  - Fabricating refund execution claims ("I have issued your refund")
  - Fabricating cancellation claims ("I have cancelled your order")
  - Hallucinating specific arrival dates/times ("will arrive by 2pm today")
  - Insecure credential requests (asking for password, OTP, full credit card)
  - Unsanitized deflection boilerplate or dead links
"""

import re
from typing import List, Optional, Tuple

from src.agent.schemas import ContextEntities, PolicyCheckResult

# Forbidden hallucination and policy violation patterns
FORBIDDEN_PATTERNS = [
    (
        re.compile(r"\b(i\s+have\s+(processed|issued|sent|refunded)\s+your\s+refund)\b", re.I),
        "UNSUPPORTED_REFUND_EXECUTION: Agent must not claim to have processed a refund without external tool confirmation.",
    ),
    (
        re.compile(r"\b(i\s+have\s+(cancelled|canceled)\s+your\s+order)\b", re.I),
        "UNSUPPORTED_CANCELLATION_EXECUTION: Agent must not claim an order is cancelled without verified execution.",
    ),
    (
        re.compile(r"\b(guaranteed\s+to\s+arrive\s+(today|tomorrow|by\s+\d))\b", re.I),
        "HALLUCINATED_DELIVERY_GUARANTEE: Agent must not guarantee specific transit arrival times without carrier lookup.",
    ),
    (
        re.compile(r"\b(tell\s+me\s+your\s+password|send\s+your\s+(password|pin|cvv|card\s*number))\b", re.I),
        "INSECURE_CREDENTIAL_REQUEST: Agent must never request customer passwords, PINs, or card details.",
    ),
    (
        re.compile(r"\b(dm\s+sent|replied\s+in\s+dm)\b", re.I),
        "UNSANITIZED_TWITTER_BOILERPLATE: Agent response contains circular deflection noise.",
    ),
    (
        re.compile(r"\^[A-Z]{2,3}\b"),
        "UNSANITIZED_AGENT_SIGNATURE: Historical agent initials must not be reproduced.",
    ),
]


class PolicyGuardrail:
    """Validates generated agent responses against safety and compliance policies."""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def validate(
        self,
        response_text: str,
        primary_intent: str,
        context: ContextEntities,
    ) -> PolicyCheckResult:
        """
        Validates response for hallucinations, false action claims, and safety violations.
        """
        violations: List[str] = []
        warnings: List[str] = []

        if not response_text or len(response_text.strip()) < 10:
            violations.append("EMPTY_RESPONSE: Response is empty or trivially brief.")
            return PolicyCheckResult(passed=False, violations=violations, warnings=warnings)

        # 1. Regex rule checks against forbidden claims
        for pattern, description in FORBIDDEN_PATTERNS:
            if pattern.search(response_text):
                violations.append(description)

        # 2. Intent-specific policy checks
        lower = response_text.lower()

        if primary_intent == "account_access_security":
            if "password" in lower and not any(w in lower for w in ["recovery", "reset", "assistance", "page", "link"]):
                warnings.append("SECURITY_WARNING: Should guide user to official recovery portal.")

        if primary_intent == "delivery_delay" and context.has_missing_info:
            if not any(w in lower for w in ["order", "track", "tracking", "details"]):
                warnings.append("MISSING_INFO_WARNING: Customer did not provide order ID; agent should prompt for details.")

        passed = len(violations) == 0
        return PolicyCheckResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            remediated=False,
        )

    def remediate(
        self,
        response_text: str,
        primary_intent: str,
        context: ContextEntities,
    ) -> Tuple[str, bool]:
        """
        Applies automated safe remediation to remove minor policy violations.
        Returns (remediated_text, was_remediated).
        """
        remediated = response_text

        # Strip agent signatures
        remediated = re.sub(r"\^[A-Z]{2,3}\b", "", remediated)

        # Replace false execution claims with self-service or escalation guidance
        remediated = re.sub(
            r"\b(i\s+have\s+(processed|issued|sent|refunded)\s+your\s+refund)\b",
            "You can track the status of your refund directly in the Returns Center at <URL>",
            remediated,
            flags=re.I,
        )

        remediated = re.sub(
            r"\b(i\s+have\s+(cancelled|canceled)\s+your\s+order)\b",
            "You can submit a cancellation request in 'Your Orders' at <URL>",
            remediated,
            flags=re.I,
        )

        remediated = re.sub(r"\s+", " ", remediated).strip()
        was_changed = remediated != response_text
        return remediated, was_changed
