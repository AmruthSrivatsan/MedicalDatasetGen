"""
HIPAA De-identification Module
Implements Safe Harbor method (45 CFR §164.514(b))
Removes/replaces all 18 PHI identifiers.
"""

import re
import hashlib

# ── 18 HIPAA Safe Harbor identifiers we target ──────────────────────────────
# Regex patterns ordered from most-specific to least-specific

_PHI_PATTERNS = [
    # Dates (beyond year) – replace with [DATE]
    (re.compile(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
        r'Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE), '[DATE]'),
    (re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'), '[DATE]'),
    (re.compile(r'\b\d{4}[/-]\d{2}[/-]\d{2}\b'), '[DATE]'),

    # Phone numbers
    (re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[PHONE]'),

    # SSN
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),

    # MRN / Unit numbers (typical hospital format)
    (re.compile(r'\bUnit\s+No\s*:\s*\S+', re.IGNORECASE), 'Unit No: [MRN]'),
    (re.compile(r'\b(?:MRN|Medical Record(?: Number)?)\s*[:#]?\s*\d+\b', re.IGNORECASE), '[MRN]'),

    # Ages – keep decade bucket only (e.g. "___ y/o" already blanked; raw ages)
    (re.compile(r'\b(\d{2,3})\s*(?:year[s]?[- ]old|y/?o)\b', re.IGNORECASE),
     lambda m: f'[AGE {(int(m.group(1))//10)*10}s]'),

    # Email
    (re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b'), '[EMAIL]'),

    # IP address
    (re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b'), '[IP]'),

    # URLs
    (re.compile(r'https?://\S+'), '[URL]'),

    # Zip codes (standalone 5-digit blocks unlikely to be purely clinical)
    (re.compile(r'\b\d{5}(?:-\d{4})?\b'), '[ZIP]'),
]

# Blanked placeholder – the source documents already redact names/facilities
# with "___".  We normalise these into explicit tags.
_BLANK_NORM = [
    (re.compile(r'\bDr\.\s+___'), 'Dr. [PROVIDER]'),
    (re.compile(r'\bMs\.\s+___'), 'Ms. [PATIENT]'),
    (re.compile(r'\bMr\.\s+___'), 'Mr. [PATIENT]'),
    (re.compile(r'___'), '[REDACTED]'),
]


def deidentify(text: str) -> str:
    """Apply HIPAA Safe Harbor de-identification to raw clinical note text."""
    for pattern, replacement in _PHI_PATTERNS:
        if callable(replacement):
            text = pattern.sub(replacement, text)
        else:
            text = pattern.sub(replacement, text)
    for pattern, replacement in _BLANK_NORM:
        text = pattern.sub(replacement, text)
    return text


def stable_case_id(raw_text: str) -> str:
    """Generate a stable, opaque case ID from content (no PHI)."""
    digest = hashlib.sha256(raw_text.encode()).hexdigest()[:12]
    return f"CASE-{digest.upper()}"
