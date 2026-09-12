"""
context.py
----------
STAGE 9: Context & Entity Extraction

Extracts:
  - Order numbers & tracking identifiers
  - Product & hardware device mentions (Kindle, Echo, Fire TV, etc.)
  - Carrier references (USPS, UPS, FedEx, AMZL, etc.)
  - Temporal & date markers
  - Sentiment & urgency classification
  - Missing information diagnosis (e.g. unstated order ID)
"""

import re
from typing import List, Optional

from src.agent.schemas import ContextEntities

ORDER_ID_PATTERNS = [
    re.compile(r"\b\d{3}-\d{7}-\d{7}\b"),          # Standard Amazon Order ID
    re.compile(r"\b(order\s*#?\s*([0-9A-Z]{8,}))\b", re.I),
    re.compile(r"#\b(\d{7,})\b"),
]

PRODUCT_PATTERNS = [
    (re.compile(r"\b(kindle|paperwhite|oasis|voyage)\b", re.I), "Kindle E-reader"),
    (re.compile(r"\b(fire\s*stick|firestick|fire\s*tv)\b", re.I), "Fire TV Device"),
    (re.compile(r"\b(echo\s*dot|echo\s*plus|echo\s*show|echo|alexa)\b", re.I), "Echo & Alexa Device"),
    (re.compile(r"\b(prime\s*video|prime\s*music|audible)\b", re.I), "Digital Streaming / Audible"),
    (re.compile(r"\b(gift\s*card|voucher|amazon\s*pay)\b", re.I), "Gift Card / Amazon Pay"),
    (re.compile(r"\b(book|shoes|phone|laptop|headphone|camera|game|toy)\b", re.I), "Physical Goods"),
]

CARRIER_PATTERNS = [
    (re.compile(r"\b(amzl|amazon\s*logistics)\b", re.I), "Amazon Logistics (AMZL)"),
    (re.compile(r"\b(usps|post\s*office)\b", re.I), "USPS"),
    (re.compile(r"\b(ups)\b", re.I), "UPS"),
    (re.compile(r"\b(fedex)\b", re.I), "FedEx"),
    (re.compile(r"\b(royal\s*mail|hermes|dpd|yodel)\b", re.I), "UK Carrier"),
]

URGENCY_PATTERNS = [
    re.compile(r"\b(urgent|urgently|emergency|asap|immediately|right\s*now|need\s*it\s*today|birthday|flight)\b", re.I)
]

ANGRY_KEYWORDS = {
    "furious", "disgusted", "scam", "fraud", "crooks", "stealing", "robbery",
    "police", "lawyer", "sue", "legal", "pathetic", "worst", "unacceptable",
    "shameful", "horrible", "terrible", "useless"
}

FRUSTRATED_KEYWORDS = {
    "annoyed", "ridiculous", "frustrated", "tired", "waiting", "again",
    "still", "joke", "upset", "disappointed", "poor"
}


def extract_context(text: str) -> ContextEntities:
    """
    Extracts structured operational context, entities, and missing information signals.
    """
    lower = text.lower()

    # 1. Order ID extraction
    order_id: Optional[str] = None
    for pattern in ORDER_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            order_id = match.group(0)
            break

    # 2. Product reference extraction
    product: Optional[str] = None
    for pattern, name in PRODUCT_PATTERNS:
        if pattern.search(text):
            product = name
            break

    # 3. Carrier extraction
    carrier: Optional[str] = None
    for pattern, name in CARRIER_PATTERNS:
        if pattern.search(text):
            carrier = name
            break

    # 4. Temporal references
    date_ref: Optional[str] = None
    date_match = re.search(r"\b(yesterday|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\w+))\b", lower)
    if date_match:
        date_ref = date_match.group(0)

    # 5. Urgency level
    urgency = "normal"
    for pattern in URGENCY_PATTERNS:
        if pattern.search(lower):
            urgency = "urgent"
            break

    # 6. Sentiment & Frustration
    words = set(re.findall(r"\b[a-z]{3,}\b", lower))
    if words.intersection(ANGRY_KEYWORDS):
        sentiment = "angry"
        urgency = "critical" if urgency == "urgent" else "urgent"
    elif words.intersection(FRUSTRATED_KEYWORDS):
        sentiment = "frustrated"
    elif any(w in words for w in ["thank", "thanks", "please", "appreciate"]):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    # 7. Missing information check
    missing_info_details = []
    # If customer is talking about order/delivery/refund/cancellation but gave no order number:
    order_related_terms = {"order", "package", "delivery", "refund", "cancel", "item", "shipped", "tracking"}
    if words.intersection(order_related_terms) and not order_id:
        missing_info_details.append("order_id")

    if "broken" in words or "damaged" in words and not product:
        missing_info_details.append("item_details")

    has_missing_info = len(missing_info_details) > 0

    return ContextEntities(
        order_id=order_id,
        product_reference=product,
        carrier_reference=carrier,
        date_reference=date_ref,
        urgency_level=urgency,
        sentiment=sentiment,
        has_missing_info=has_missing_info,
        missing_info_details=missing_info_details,
    )
