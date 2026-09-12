"""
config.py
---------
STAGE 9: Agent Configuration & Thresholds

Defines operational thresholds, retrieval parameters, safety settings,
and optional LLM provider configurations read from environment variables.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
TAXONOMY_PATH = DATA_DIR / "intent_taxonomy.json"
GOLDEN_SET_PATH = DATA_DIR / "golden" / "golden_evaluation_set.jsonl"
REPORTS_DIR = REPO_ROOT / "reports"


@dataclass
class AgentConfig:
    """Configuration parameters for the AI Support Agent."""

    # Brand identity
    brand: str = "AmazonHelp"

    # Intent classification thresholds
    low_confidence_threshold: float = 0.55
    high_confidence_threshold: float = 0.85
    multi_intent_margin: float = 0.15

    # Escalation risk thresholds
    high_risk_threshold: float = 0.70
    max_retrieval_history: int = 3
    min_query_length_chars: int = 5

    # Feature flags for ablation & testing
    enable_retrieval: bool = True
    enable_policy_guardrails: bool = True
    enable_context_extraction: bool = True

    # Generator settings: 'deterministic' | 'llm'
    generator_type: str = "deterministic"

    # Optional LLM integration settings (reads environment variables safely)
    llm_provider: Optional[str] = os.getenv("LLM_PROVIDER", None)
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY", None)
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = 0.0
    llm_max_tokens: int = 250
    llm_timeout_seconds: int = 10
