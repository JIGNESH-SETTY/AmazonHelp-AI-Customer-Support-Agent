"""
cli.py
------
STAGE 12: Interactive Agent CLI & Demonstration Entry Point

Provides:
  1. Interactive terminal session for real-time customer support testing.
  2. Synthetic demonstration suite covering 6 canonical support scenarios.
  3. Structured decision metadata display (Intent, Confidence, Context,
     Retrieval Proofs, Policy Status, Escalation, Response).
  4. Zero chain-of-thought exposure.
  5. 100% offline, deterministic execution without external API dependencies.

Usage:
  python -m src.agent.cli
  python -m src.agent.cli --demo
  python -m src.agent.cli "Where is my package? It was supposed to arrive yesterday."
"""

import argparse
import sys
from typing import Any, Dict, List, Optional

# UTF-8 stdout configuration for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.agent.agent import SupportAgent
from src.agent.config import AgentConfig
from src.evaluation.evaluate_baselines import load_training_corpus

# ---------------------------------------------------------------------------
# SYNTHETIC DEMONSTRATION SCENARIOS (Zero PII)
# ---------------------------------------------------------------------------

SYNTHETIC_DEMO_SCENARIOS: List[Dict[str, str]] = [
    {
        "scenario_name": "1. Delayed Delivery Inquiry",
        "description": "Customer asking for updated tracking on an overdue in-transit package.",
        "input": "Hi Amazon, my order was supposed to be delivered yesterday by 8pm but tracking hasn't updated since it left the carrier facility. Where is it?",
    },
    {
        "scenario_name": "2. Damaged Item / Refund Request",
        "description": "Customer received a shattered item and requests a full refund.",
        "input": "I received my order today but the ceramic dinner plates were completely broken and shattered inside the box. I want my money back please.",
    },
    {
        "scenario_name": "3. Order / Subscription Cancellation",
        "description": "Customer requesting to cancel a Prime subscription.",
        "input": "Is there a direct link to cancel my prime membership? The app is confusing and I do not want to renew this month.",
    },
    {
        "scenario_name": "4. Double Billing / Payment Dispute",
        "description": "Customer noticing double charges on their credit card statement.",
        "input": "I was charged twice on my credit card for the exact same order invoice. Please fix this double billing charge immediately.",
    },
    {
        "scenario_name": "5. Ambiguous Multi-Intent Customer Request",
        "description": "Customer inquiry conflating shipment delay with subscription frustration.",
        "input": "My package is 3 days late again. Why do I pay for Prime 2-day delivery if everything is delayed? Should I just cancel prime?",
    },
    {
        "scenario_name": "6. Low-Confidence / Vague Deflection",
        "description": "Extremely brief, ungrounded inquiry requiring structured clarification.",
        "input": "it is not working",
    },
]


def format_support_box(query: str, output: Any) -> str:
    """Formats the agent response into the canonical clean support display."""
    sec_intents = ", ".join(output.secondary_intents) if output.secondary_intents else "None"
    
    # Format retrieved evidence
    if output.retrieval:
        ret_summary = f"{len(output.retrieval)} reference chunk(s) retrieved from knowledge base"
    else:
        ret_summary = "None (Standard policy guidance applied)"

    # Policy status
    pol_passed = output.policy.get("passed", True)
    pol_status = "Passed (Compliant with customer support standards)" if pol_passed else "Violations detected"

    # Escalation
    esc_req = output.escalation.get("required", False)
    esc_reason = output.escalation.get("reason", "None")
    esc_status = f"Escalated ({esc_reason})" if esc_req else "Resolved via automated guidance"

    box = f"""----------------------------------------
CUSTOMER INQUIRY:
"{query}"
----------------------------------------
SUPPORT AGENT DECISION METADATA
----------------------------------------
Intent:             {output.primary_intent} ({output.primary_intent_name})
Confidence:         {output.intent_confidence:.2%}
Secondary intents:  {sec_intents}
Context:            {output.context}
Retrieved evidence: {ret_summary}
Policy status:      {pol_status}
Escalation:         {esc_status}
Latency:            {output.latency_ms:.2f} ms

AGENT RESPONSE:
"{output.response}"
----------------------------------------"""
    return box


def create_production_agent(max_samples: int = 15_000) -> SupportAgent:
    """Initializes and trains the SupportAgent instance."""
    print(f"[*] Initializing AmazonHelp AI Support Agent...")
    train_texts, train_labels, train_meta = load_training_corpus(max_samples=max_samples)
    agent = SupportAgent(config=AgentConfig())
    agent.fit_training_data(train_texts, train_labels, train_meta)
    print(f"[*] Support Agent initialized and ready!\n")
    return agent


def run_demo_suite(agent: SupportAgent) -> None:
    """Runs through all 6 synthetic demo scenarios."""
    print("==================================================")
    print("AMAZONHELP AI SUPPORT AGENT — DEMONSTRATION SUITE")
    print("Running 6 Canonical Synthetic Customer Scenarios")
    print("==================================================\n")

    for idx, sc in enumerate(SYNTHETIC_DEMO_SCENARIOS, start=1):
        print(f"### SCENARIO {idx}: {sc['scenario_name']}")
        print(f"Context: {sc['description']}")
        output = agent.process(sc["input"])
        print(format_support_box(sc["input"], output))
        print()


def run_interactive(agent: SupportAgent) -> None:
    """Interactive loop for manual query evaluation."""
    print("==================================================")
    print("AMAZONHELP AI SUPPORT AGENT — INTERACTIVE MODE")
    print("Type your customer message below (type 'exit' to quit):")
    print("==================================================\n")

    while True:
        try:
            user_msg = input("\nEnter customer message > ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive demo. Goodbye!")
                break

            output = agent.process(user_msg)
            print()
            print(format_support_box(user_msg, output))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive demo.")
            break


def main():
    parser = argparse.ArgumentParser(description="AmazonHelp AI Support Agent CLI")
    parser.add_argument("query", nargs="*", default=[], help="Customer inquiry text to process")
    parser.add_argument("--demo", action="store_true", help="Run the automated synthetic demo suite")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive chat mode")
    args = parser.parse_args()

    agent = create_production_agent()

    if args.demo:
        run_demo_suite(agent)
    elif args.query:
        query_text = " ".join(args.query)
        output = agent.process(query_text)
        print(format_support_box(query_text, output))
    else:
        # Default to demo suite if not in interactive tty, or interactive if terminal
        if sys.stdin.isatty():
            run_interactive(agent)
        else:
            run_demo_suite(agent)


if __name__ == "__main__":
    main()
