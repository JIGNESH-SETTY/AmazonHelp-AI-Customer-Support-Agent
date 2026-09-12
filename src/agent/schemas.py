"""
schemas.py
----------
STAGE 9: Structured Data Schemas & Contracts

Defines all typed dataclasses for the agent pipeline.
Ensures zero chain-of-thought leakage in final outputs while providing
rich structured intermediate decision metadata.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentInput:
    """Raw customer message and optional conversation context."""
    customer_text: str
    conversation_id: str = "conv_default"
    channel: str = "twitter"


@dataclass
class NormalizedInput:
    """Normalized text output with audit tracking."""
    original_text: str
    normalized_text: str
    is_valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class CandidateIntent:
    """Ranked candidate intent with score."""
    intent_id: str
    intent_name: str
    score: float


@dataclass
class IntentResult:
    """Intent classification metadata."""
    primary_intent: str
    primary_intent_name: str
    intent_confidence: float
    secondary_intents: List[str] = field(default_factory=list)
    top_candidates: List[CandidateIntent] = field(default_factory=list)


@dataclass
class ContextEntities:
    """Extracted contextual entities and operational signals."""
    order_id: Optional[str] = None
    product_reference: Optional[str] = None
    carrier_reference: Optional[str] = None
    date_reference: Optional[str] = None
    urgency_level: str = "normal"  # 'normal' | 'urgent' | 'critical'
    sentiment: str = "neutral"      # 'positive' | 'neutral' | 'frustrated' | 'angry'
    has_missing_info: bool = False
    missing_info_details: List[str] = field(default_factory=list)


@dataclass
class RetrievalItem:
    """Retrieved historical context item."""
    example_id: str
    relevance_score: float
    customer_query: str
    historical_response: str
    source_dataset: str = "amazonhelp_historical"


@dataclass
class PolicyCheckResult:
    """Policy guardrail compliance check."""
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    remediated: bool = False


@dataclass
class EscalationSignal:
    """Escalation trigger and severity."""
    required: bool
    reason: Optional[str] = None
    priority: str = "none"  # 'none' | 'medium' | 'high' | 'urgent'


@dataclass
class AgentOutput:
    """
    Final structured agent response payload.
    Contains decision metadata WITHOUT exposing chain-of-thought reasoning.
    """
    conversation_id: str
    primary_intent: str
    primary_intent_name: str
    intent_confidence: float
    secondary_intents: List[str]
    context: Dict[str, Any]
    retrieval: List[Dict[str, Any]]
    policy: Dict[str, Any]
    escalation: Dict[str, Any]
    response: str
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts output to JSON-serializable dictionary."""
        return asdict(self)
