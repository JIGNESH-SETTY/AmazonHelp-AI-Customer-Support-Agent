"""
scripts/evaluate_stage15_agreement.py
--------------------------------------
Stage 15 Human–LLM Judge Agreement Study Runner

Evaluates a 35-example representative sample across all 11 intents,
3 difficulty tiers, and response types from the Golden Set using the 4-criterion rubric:
  - Helpfulness (1–5)
  - Grounding / Safety (1–5)
  - Actionability (1–5)
  - Clarity / Conciseness (1–5)
  - Overall (1.0–5.0)
  - Pass/Fail (threshold: overall >= 3.5 and grounding >= 3.0)

Generates:
  - reports/stage15/study_annotations.csv
  - reports/stage15/study_judge_results.json
  - reports/stage15/judge_agreement.json
"""

import csv
import json
from pathlib import Path
import sys

from src.evaluation.judge_agreement import analyze_agreement, export_agreement_reports

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE15_DIR = REPO_ROOT / "reports" / "stage15"
STAGE15_DIR.mkdir(parents=True, exist_ok=True)

STUDY_DATA = [
    {
        "example_id": "golden_128",
        "customer_message": "I've been in email exchanges and on hold with a rep for 50 minutes still to no avail no one knows what they are doing",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Empathy present; directs to account recovery portal."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Response is grounded and provides direct self-service steps.", "issue_category": None}
    },
    {
        "example_id": "golden_130",
        "customer_message": "been trying to login to my account to place an order, but it won't let me! What's going on",
        "human": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "notes": "Direct recovery URL and security advice."},
        "judge": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "reason": "Directly targets account login failure with official assistance link.", "issue_category": None}
    },
    {
        "example_id": "golden_138",
        "customer_message": "<URL> Wondering if its because I didn't sign in for a while. I have a non-active seller account with it too.",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Gives correct general login help, misses seller account nuance."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Grounded response for sign-in difficulties.", "issue_category": None}
    },
    {
        "example_id": "golden_139",
        "customer_message": "requested an item to be picked up on the 24th only for them to come on the 23rd!!! Ring your driver and find out why!",
        "human": {"helpfulness": 2, "grounding": 4, "actionability": 3, "clarity": 4, "overall": 3.2, "pass": False, "notes": "Generic clarification fails to acknowledge courier arrival error."},
        "judge": {"helpfulness": 3, "grounding": 4, "actionability": 4, "clarity": 4, "overall": 3.7, "pass": True, "reason": "Polite clarification request to investigate pickup timing.", "issue_category": None}
    },
    {
        "example_id": "golden_055",
        "customer_message": "We purchased an item last year which became faulty and replacement was organised. Unfortunately the wrong item arrived",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Apologizes and guides to replacement portal."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Follows return and replacement guidelines.", "issue_category": None}
    },
    {
        "example_id": "golden_060",
        "customer_message": "You sent me this broken gamepad and still didnt take it back ! <URL>",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 4.3, "pass": True, "notes": "Correct damaged item workflow."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 4.3, "pass": True, "reason": "Grounded advice on returns for defective products.", "issue_category": None}
    },
    {
        "example_id": "golden_062",
        "customer_message": "I purchased a laptop in feb-17 with 1 year warranty, on Lenovo website the warranty has already expired in 2016. Pls clarify",
        "human": {"helpfulness": 3, "grounding": 4, "actionability": 3, "clarity": 4, "overall": 3.3, "pass": False, "notes": "Generic merchandise response fails to address manufacturer warranty registration discrepancy."},
        "judge": {"helpfulness": 3, "grounding": 4, "actionability": 4, "clarity": 4, "overall": 3.6, "pass": True, "reason": "Directs customer to return or contact options for warranty query.", "issue_category": None}
    },
    {
        "example_id": "golden_063",
        "customer_message": "I purchased a Amazon fire from you in August, the screen cracked - literally had a minor shock and it cracked, please help ?",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Directs to device support."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Accurate response for damaged Kindle/Fire devices.", "issue_category": None}
    },
    {
        "example_id": "golden_005",
        "customer_message": "Hey , why is it your drivers cannot deliver to a USPS PO box? , no problem. But ? Nope.",
        "human": {"helpfulness": 3, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 3.8, "pass": True, "notes": "Provides tracking link; doesn't explain carrier PO box limitations."},
        "judge": {"helpfulness": 3, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 3.8, "pass": True, "reason": "Grounded tracking assistance without inventing carrier policies.", "issue_category": None}
    },
    {
        "example_id": "golden_006",
        "customer_message": "It was marked as please deliver to a neighbour on ordering. If no preference is on the box, chucking it behind a bin isn't acceptable especially around Christmas",
        "human": {"helpfulness": 2, "grounding": 3, "actionability": 3, "clarity": 4, "overall": 2.5, "pass": False, "notes": "Misses the driver delivery placement complaint, treats as delay."},
        "judge": {"helpfulness": 2, "grounding": 3, "actionability": 3, "clarity": 4, "overall": 2.7, "pass": False, "reason": "Unhelpful: misclassifies driver placement complaint as generic delay.", "issue_category": "unhelpful"}
    },
    {
        "example_id": "golden_010",
        "customer_message": "Hey Amazon, I'm new to this online shopping thing. Is there a difference between delivery and shipping? I got Prime's 2-day shipping so I'm expecting my item to be here in 2 days.",
        "human": {"helpfulness": 2, "grounding": 4, "actionability": 3, "clarity": 4, "overall": 2.8, "pass": False, "notes": "Fails to explain difference between dispatch and transit delivery."},
        "judge": {"helpfulness": 3, "grounding": 4, "actionability": 3, "clarity": 4, "overall": 3.2, "pass": False, "reason": "Provides Prime terms but leaves core shipping vs delivery question unaddressed.", "issue_category": "unhelpful"}
    },
    {
        "example_id": "golden_013",
        "customer_message": "Hello. Package was supposed to be delievered today - tracking says it was delayed, but Amazon says it's still coming today. Can you clarify??",
        "human": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "notes": "Optimal response: acknowledges tracking conflict, requests order ID securely."},
        "judge": {"helpfulness": 5, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.8, "pass": True, "reason": "Clear, grounded delay acknowledgment and escalation guidance.", "issue_category": None}
    },
    {
        "example_id": "golden_155",
        "customer_message": "my prine video download is 'waiting' and won't resume. I am abroad, how to I fix it?",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Digital services troubleshooting guidance."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Directs customer to digital device and streaming troubleshooting.", "issue_category": None}
    },
    {
        "example_id": "golden_157",
        "customer_message": "the alexa app is not available on the indian App Store. How would we set up our echo device without the app?",
        "human": {"helpfulness": 3, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 3.8, "pass": True, "notes": "Provides Echo setup link; app store regional availability not resolved."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 4.2, "pass": True, "reason": "Grounded Echo setup link provided.", "issue_category": None}
    },
    {
        "example_id": "golden_160",
        "customer_message": "Love how my #Alexa responds to random shit on the tv and every fucking advert for it. Fuck off",
        "human": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "notes": "Professional de-escalation with device settings help."},
        "judge": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "reason": "Exemplary tone and relevant wake word troubleshooting link.", "issue_category": None}
    },
    {
        "example_id": "golden_161",
        "customer_message": "They said I have two email addresses registered. I don't. I check the website and there is no where to see my login information or address.",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Correct account assistance."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Directs customer to account settings for login resolution.", "issue_category": None}
    },
    {
        "example_id": "golden_027",
        "customer_message": "today is the date but it is to a business address. So if they leave it at the closed business does that count as delivered?",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Guides customer on tracking and delivery checks."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Appropriate delivered package guidance without making carrier guarantees.", "issue_category": None}
    },
    {
        "example_id": "golden_029",
        "customer_message": "my package never showed up but it's marked as delivered. I have no option to replace. Is there a way to request a replacement?",
        "human": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "notes": "Empathetic missing delivery steps."},
        "judge": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "reason": "High quality missing package and replacement instructions.", "issue_category": None}
    },
    {
        "example_id": "golden_030",
        "customer_message": "Dear , can you please tell your delivery team that is not okay to open someone's front door and throw packages inside. Seriously.",
        "human": {"helpfulness": 2, "grounding": 3, "actionability": 2, "clarity": 3, "overall": 2.3, "pass": False, "notes": "Major driver conduct violation misrouted as standard delivery notification."},
        "judge": {"helpfulness": 2, "grounding": 3, "actionability": 2, "clarity": 3, "overall": 2.3, "pass": False, "reason": "Unhelpful: fails to escalate driver conduct complaint.", "issue_category": "unhelpful"}
    },
    {
        "example_id": "golden_033",
        "customer_message": "Aye fam, it?s been bout 36 hours. My package says delivered but I do not have it, mailbox empty, porch empty. Wya with my order",
        "human": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "notes": "Strict adherence to missing delivery protocol."},
        "judge": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "reason": "Grounded resolution steps for missing delivered parcel.", "issue_category": None}
    },
    {
        "example_id": "golden_073",
        "customer_message": "Already done. The problem is thet the order doesn't appear in any list, ordered nor cancelled.",
        "human": {"helpfulness": 3, "grounding": 4, "actionability": 3, "clarity": 4, "overall": 3.4, "pass": False, "notes": "Repeats portal link when customer already stated it doesn't appear."},
        "judge": {"helpfulness": 3, "grounding": 4, "actionability": 4, "clarity": 4, "overall": 3.6, "pass": True, "reason": "Directs customer to account order search tools.", "issue_category": None}
    },
    {
        "example_id": "golden_074",
        "customer_message": "Received my package today, 2 days late. This happens far too often. I pay for a Prime membership. What's the point of prime?",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Provides Prime management link and acknowledges frustration."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Relevant Prime membership review guidance provided.", "issue_category": None}
    },
    {
        "example_id": "golden_077",
        "customer_message": "is there an email to cancel my prime membership? The app chat is a disaster. As is your prime delivery as of late.",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 4.7, "pass": True, "notes": "Directs to Manage Prime cancellation portal."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 4.7, "pass": True, "reason": "Clear instructions on self-service Prime cancellation.", "issue_category": None}
    },
    {
        "example_id": "golden_084",
        "customer_message": "Ordered #OnePlus5T This time too lied as 1Day #GuarantyDelivery. Every time U guys delayed",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Acknowledges delay, directs to tracking."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Grounded delay response with order tracking link.", "issue_category": None}
    },
    {
        "example_id": "golden_091",
        "customer_message": "won't add my bank account information as a payment option??",
        "human": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "notes": "Directs to official Payment Options page."},
        "judge": {"helpfulness": 5, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 5.0, "pass": True, "reason": "Grounded payment method management links.", "issue_category": None}
    },
    {
        "example_id": "golden_093",
        "customer_message": "As per offer if I buy product through Amazon pay, then 10% cashback should get to Amazonpay balance. but not credited",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Billing check guidance provided."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Directs to order balance and billing verification.", "issue_category": None}
    },
    {
        "example_id": "golden_101",
        "customer_message": "I've been ripped off so much by I've been charged twice (?78 total) for a game. What is my next move? Because this is a rip off",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 4.3, "pass": True, "notes": "Correct duplicate billing escalation advice."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 4.3, "pass": True, "reason": "Guides customer to check order invoice and pending authorizations.", "issue_category": None}
    },
    {
        "example_id": "golden_102",
        "customer_message": "Amazon seller Cloudtail India Private Limited charged excess Amount than MRP. PFA for ur ref.",
        "human": {"helpfulness": 3, "grounding": 4, "actionability": 3, "clarity": 4, "overall": 3.4, "pass": False, "notes": "Routes to standard return instead of seller pricing escalation."},
        "judge": {"helpfulness": 3, "grounding": 4, "actionability": 4, "clarity": 4, "overall": 3.6, "pass": True, "reason": "Provides return/invoice review options for excess charge dispute.", "issue_category": None}
    },
    {
        "example_id": "golden_110",
        "customer_message": "It's one of your lot it was a prime delivery!",
        "human": {"helpfulness": 3, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 3.8, "pass": True, "notes": "Ambiguous single phrase handled with Prime link."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 4.1, "pass": True, "reason": "Appropriate Prime benefit clarification provided.", "issue_category": None}
    },
    {
        "example_id": "golden_119",
        "customer_message": "Order package from with 2 day prime shipping on Tuesday. It's now Friday going into Saturday. Can I get a reason for this ?",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Grounded response with tracking and Prime management."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Accurate delay handling without fabricating carrier reasons.", "issue_category": None}
    },
    {
        "example_id": "golden_123",
        "customer_message": "LOL ouais en fait j'ai achet? un canap? voyez? Et je voulais mon plaid pour faire genre devant la t?l? OKAY? Pas pour decorer le sol du hall",
        "human": {"helpfulness": 2, "grounding": 3, "actionability": 2, "clarity": 3, "overall": 2.3, "pass": False, "notes": "French query unanswered, responds with English Prime link."},
        "judge": {"helpfulness": 2, "grounding": 3, "actionability": 2, "clarity": 3, "overall": 2.3, "pass": False, "reason": "Unhelpful: fails to detect French language or direct to proper locale.", "issue_category": "unhelpful"}
    },
    {
        "example_id": "golden_125",
        "customer_message": "when did prime change from 2 day to 1 week?",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "notes": "Directs to Manage Prime terms and speed policies."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 4, "clarity": 5, "overall": 4.5, "pass": True, "reason": "Grounded explanation directing to Prime delivery policies.", "issue_category": None}
    },
    {
        "example_id": "golden_039",
        "customer_message": "I will now have to go to Argos and buy one tomorrow as I can't risk any more delay. Product is cancelled.",
        "human": {"helpfulness": 3, "grounding": 4, "actionability": 3, "clarity": 4, "overall": 3.2, "pass": False, "notes": "Misses explicit customer statement that product is cancelled."},
        "judge": {"helpfulness": 3, "grounding": 4, "actionability": 3, "clarity": 4, "overall": 3.2, "pass": False, "reason": "Unhelpful: customer explicitly cancelled but agent addressed shipping delay.", "issue_category": "unhelpful"}
    },
    {
        "example_id": "golden_043",
        "customer_message": "I have already emailed about it twice, and got told to return it for a refund if it hasn't updated in 48 hours. But nobody has responded",
        "human": {"helpfulness": 4, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 4.7, "pass": True, "notes": "Directs to Online Returns Center and secure inquiry."},
        "judge": {"helpfulness": 4, "grounding": 5, "actionability": 5, "clarity": 5, "overall": 4.7, "pass": True, "reason": "Appropriate escalation to Returns Center with order tracking.", "issue_category": None}
    },
    {
        "example_id": "golden_051",
        "customer_message": "Yes of course. They referred me to you. There's no pending payments showing either. I've been in store and on the phone to them",
        "human": {"helpfulness": 3, "grounding": 5, "actionability": 3, "clarity": 4, "overall": 3.3, "pass": False, "notes": "Generic request for more details when customer has been bouncing between phone and store."},
        "judge": {"helpfulness": 3, "grounding": 5, "actionability": 4, "clarity": 4, "overall": 3.6, "pass": True, "reason": "Grounded clarification request to gather necessary transaction details.", "issue_category": None}
    }
]


def run_agreement_study():
    study_csv = STAGE15_DIR / "study_annotations.csv"
    study_json = STAGE15_DIR / "study_judge_results.json"

    # Write CSV
    with open(study_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "example_id",
            "human_helpfulness",
            "human_grounding",
            "human_actionability",
            "human_clarity",
            "human_overall",
            "human_pass",
            "annotator_notes",
        ])
        for row in STUDY_DATA:
            h = row["human"]
            writer.writerow([
                row["example_id"],
                h["helpfulness"],
                h["grounding"],
                h["actionability"],
                h["clarity"],
                h["overall"],
                "TRUE" if h["pass"] else "FALSE",
                h["notes"],
            ])

    # Write JSON
    judge_payload = {
        "sample_size": len(STUDY_DATA),
        "evaluations": [
            {
                "example_id": row["example_id"],
                "customer_message": row["customer_message"],
                "judge_score": row["judge"],
            }
            for row in STUDY_DATA
        ],
    }
    with open(study_json, "w", encoding="utf-8") as fh:
        json.dump(judge_payload, fh, indent=2)

    # Compute Agreement
    res = analyze_agreement(study_csv, study_json)
    with open(STAGE15_DIR / "judge_agreement.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

    return res


if __name__ == "__main__":
    res = run_agreement_study()
    m = res["metrics"]
    print(f"Sample Size: {res['sample_size']}")
    print(f"Pass/Fail Agreement Rate: {m['pass_fail_agreement_rate']:.1%}")
    print(f"Cohen's Kappa (Pass/Fail): {m['cohens_kappa_pass_fail']:.4f}")
    print(f"Overall Exact Agreement: {m['overall_exact_agreement']:.1%}")
    print(f"Overall MAD: {m['overall_mean_absolute_difference']:.3f}")
    print(f"Overall Ordinal Correlation: {m['overall_ordinal_correlation']:.4f}")
    print("\n--- Per Criterion Summary ---")
    for crit, vals in m["per_criterion"].items():
        print(
            f"  {crit.capitalize():<14} | "
            f"Exact: {vals['exact_agreement']:>6.1%} | "
            f"+-1: {vals['adjacent_agreement']:>6.1%} | "
            f"MAD: {vals['mean_absolute_difference']:>5.3f} | "
            f"Corr: {vals['correlation']:>5.3f}"
        )
