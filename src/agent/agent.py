"""
agent.py
--------
STAGE 9: Main AI Support Agent Orchestrator

Coordinates the modular pipeline:
  INPUT -> NORMALIZATION -> HYBRID INTENT -> CONTEXT -> RETRIEVAL ->
  POLICY CHECK -> RESPONSE GENERATION -> SAFETY GUARDRAIL -> ESCALATION -> STRUCTURED OUTPUT
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from src.agent.config import AgentConfig
from src.agent.context import extract_context
from src.agent.generation import (
    DeterministicResponseGenerator,
    LLMResponseGenerator,
    ResponseGenerator,
)
from src.agent.intent import HybridIntentClassifier
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
from src.evaluation.baselines import TFIDFNaiveBayesBaseline
from src.retrieval.bm25 import BM25Index


class SupportAgent:
    """
    End-to-end AI Support Agent for AmazonHelp customer service conversations.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        intent_classifier: Optional[HybridIntentClassifier] = None,
        retriever: Optional[KnowledgeRetriever] = None,
        policy_guardrail: Optional[PolicyGuardrail] = None,
        generator: Optional[ResponseGenerator] = None,
    ):
        self.config = config or AgentConfig()
        self.intent_classifier = intent_classifier or HybridIntentClassifier()
        self.retriever = retriever or KnowledgeRetriever(top_k=self.config.max_retrieval_history)
        self.policy = policy_guardrail or PolicyGuardrail(strict_mode=True)

        if generator is not None:
            self.generator = generator
        elif self.config.generator_type == "llm":
            self.generator = LLMResponseGenerator(self.config)
        else:
            self.generator = DeterministicResponseGenerator()

    def fit_training_data(
        self,
        train_texts: List[str],
        train_labels: List[str],
        train_meta: List[Dict[str, Any]],
    ) -> None:
        """
        Fits underlying TF-IDF intent model and BM25 historical retrieval index
        strictly on training split data.
        """
        # 1. Fit TF-IDF Naive Bayes
        tfidf_model = TFIDFNaiveBayesBaseline(alpha=1.0, min_df=3)
        tfidf_model.fit(train_texts, train_labels)
        self.intent_classifier.set_tfidf_model(tfidf_model)

        # 2. Fit BM25 Knowledge Retrieval Index
        bm25_index = BM25Index(k1=1.5, b=0.75)
        bm25_index.fit(train_texts, train_meta)
        self.retriever.set_index(bm25_index)

    def process(self, raw_input: Any) -> AgentOutput:
        """
        Executes full agent pipeline for a single customer interaction.
        Accepts str or AgentInput.
        """
        t0 = time.time()

        if isinstance(raw_input, str):
            agent_input = AgentInput(customer_text=raw_input)
        else:
            agent_input = raw_input

        conv_id = agent_input.conversation_id

        # 1. Normalization
        norm_result: NormalizedInput = normalize_input(agent_input)
        if not norm_result.is_valid:
            elapsed_ms = (time.time() - t0) * 1000.0
            return AgentOutput(
                conversation_id=conv_id,
                primary_intent="unknown_or_ambiguous",
                primary_intent_name="Unknown or Ambiguous",
                intent_confidence=0.0,
                secondary_intents=[],
                context={},
                retrieval=[],
                policy={"passed": True, "violations": []},
                escalation={"required": False, "reason": None, "priority": "none"},
                response="Thank you for reaching out to Amazon Customer Support. Please send us your message or inquiry so we can assist you.",
                latency_ms=round(elapsed_ms, 2),
            )

        clean_text = norm_result.normalized_text

        # 2. Intent Classification
        intent_res: IntentResult = self.intent_classifier.predict(clean_text)

        # 3. Context & Entity Extraction
        if self.config.enable_context_extraction:
            context_entities: ContextEntities = extract_context(clean_text)
        else:
            context_entities = ContextEntities()

        # 4. Knowledge Retrieval (Historical Evidence)
        retrieval_items: List[RetrievalItem] = []
        if self.config.enable_retrieval:
            retrieval_items = self.retriever.retrieve(clean_text)

        # 5. Response Generation
        raw_response = self.generator.generate(
            customer_query=clean_text,
            intent_result=intent_res,
            context=context_entities,
            retrieval_items=retrieval_items,
        )

        # 6. Policy Safety Guardrails
        policy_res: PolicyCheckResult
        final_response = raw_response
        if self.config.enable_policy_guardrails:
            policy_res = self.policy.validate(raw_response, intent_res.primary_intent, context_entities)
            if not policy_res.passed:
                # Attempt automated safe remediation
                remediated_text, was_changed = self.policy.remediate(
                    raw_response, intent_res.primary_intent, context_entities
                )
                if was_changed:
                    final_response = remediated_text
                    policy_res.remediated = True
                    # Re-validate remediated response
                    recheck = self.policy.validate(final_response, intent_res.primary_intent, context_entities)
                    policy_res.passed = recheck.passed
        else:
            policy_res = PolicyCheckResult(passed=True, violations=[], warnings=[])

        # 7. Escalation Signaling
        escalation: EscalationSignal = self._evaluate_escalation(
            intent_res=intent_res,
            context=context_entities,
            policy_res=policy_res,
        )

        elapsed_ms = (time.time() - t0) * 1000.0

        return AgentOutput(
            conversation_id=conv_id,
            primary_intent=intent_res.primary_intent,
            primary_intent_name=intent_res.primary_intent_name,
            intent_confidence=intent_res.intent_confidence,
            secondary_intents=intent_res.secondary_intents,
            context={
                "order_id": context_entities.order_id,
                "product_reference": context_entities.product_reference,
                "carrier_reference": context_entities.carrier_reference,
                "sentiment": context_entities.sentiment,
                "urgency": context_entities.urgency_level,
                "has_missing_info": context_entities.has_missing_info,
            },
            retrieval=[
                {
                    "example_id": item.example_id,
                    "relevance_score": item.relevance_score,
                    "matched_query": item.customer_query[:100],
                    "evidence_snippet": item.historical_response[:120],
                }
                for item in retrieval_items
            ],
            policy={
                "passed": policy_res.passed,
                "violations": policy_res.violations,
                "warnings": policy_res.warnings,
                "remediated": policy_res.remediated,
            },
            escalation={
                "required": escalation.required,
                "reason": escalation.reason,
                "priority": escalation.priority,
            },
            response=final_response,
            latency_ms=round(elapsed_ms, 2),
        )

    def _evaluate_escalation(
        self,
        intent_res: IntentResult,
        context: ContextEntities,
        policy_res: PolicyCheckResult,
    ) -> EscalationSignal:
        """
        Determines if an inquiry should be flagged for human escalation.
        """
        # Explicit customer service complaint / legal threat
        if intent_res.primary_intent == "service_complaint_escalation":
            return EscalationSignal(
                required=True,
                reason="Customer service complaint or human specialist requested.",
                priority="high",
            )

        # Critical angry sentiment
        if context.sentiment == "angry" and context.urgency_level == "critical":
            return EscalationSignal(
                required=True,
                reason="Customer expressed severe frustration or critical urgency.",
                priority="urgent",
            )

        # Severe unresolvable policy safety violation
        if not policy_res.passed:
            return EscalationSignal(
                required=True,
                reason=f"Policy safety guardrail violation: {'; '.join(policy_res.violations)}",
                priority="medium",
            )

        # Very low intent confidence
        if intent_res.intent_confidence < self.config.low_confidence_threshold:
            return EscalationSignal(
                required=True,
                reason="Low intent classification confidence.",
                priority="medium",
            )

        return EscalationSignal(required=False, reason=None, priority="none")
