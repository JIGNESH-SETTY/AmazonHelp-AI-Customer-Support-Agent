"""
src.agent
=========
Modular AI Support Agent package for AmazonHelp customer service.

Components:
  - `SupportAgent`: Main end-to-end agent orchestrator.
  - `AgentConfig`: Operational configuration and thresholds.
  - `schemas`: Typed dataclasses for input, normalization, intent, context, policy, and output.
  - `normalize`: Conservative input normalization.
  - `context`: Operational entity and context extraction.
  - `intent`: Hybrid probabilistic and keyword intent classifier.
  - `retrieval`: Sanitized BM25 historical support knowledge retrieval.
  - `policy`: Policy guardrail and anti-hallucination compliance engine.
  - `generation`: Dynamic grounded response generator with LLM interface and fallback.
  - `cli`: Interactive demonstration entry point.
"""

from src.agent.agent import SupportAgent
from src.agent.config import AgentConfig
from src.agent.context import extract_context
from src.agent.generation import (
    DeterministicResponseGenerator,
    LLMResponseGenerator,
    ResponseGenerator,
)
from src.agent.intent import HybridIntentClassifier, IntentClassifier
from src.agent.normalize import normalize_input
from src.agent.policy import PolicyGuardrail
from src.agent.retrieval import KnowledgeRetriever
from src.agent.schemas import (
    AgentInput,
    AgentOutput,
    ContextEntities,
    EscalationSignal,
    IntentResult,
    NormalizedInput,
    PolicyCheckResult,
    RetrievalItem,
)

__all__ = [
    "SupportAgent",
    "AgentConfig",
    "AgentInput",
    "NormalizedInput",
    "IntentResult",
    "ContextEntities",
    "RetrievalItem",
    "PolicyCheckResult",
    "EscalationSignal",
    "AgentOutput",
    "normalize_input",
    "extract_context",
    "HybridIntentClassifier",
    "IntentClassifier",
    "KnowledgeRetriever",
    "PolicyGuardrail",
    "ResponseGenerator",
    "DeterministicResponseGenerator",
    "LLMResponseGenerator",
]
