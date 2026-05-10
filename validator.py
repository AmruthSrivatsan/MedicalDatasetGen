"""
Quality validation for generated training examples.
Checks for PHI leakage, minimum content length, and abbreviation residue.
"""

import re
from typing import Tuple

# Patterns that should NOT appear in clean output (residual PHI indicators)
_PHI_LEAK_PATTERNS = [
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),          # SSN
    re.compile(r'\b\d{10,}\b'),                      # Long number strings (MRN etc.)
    re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),  # phone
    re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b'),  # email
]

# High-frequency abbreviations that must NOT survive into training text
_RESIDUAL_ABBREV_PATTERNS = [
    re.compile(r'\bCOPD\b'),
    re.compile(r'\bCAD\b'),
    re.compile(r'\bHTN\b'),
    re.compile(r'\bSOB\b'),
    re.compile(r'\bRVR\b'),
    re.compile(r'\bCTA\b'),
    re.compile(r'\bOSH\b'),
    re.compile(r'\bTIA\b'),
    re.compile(r'\bMCA\b'),
    re.compile(r'\bNIHSS\b'),
    re.compile(r'\bEMS\b'),
    re.compile(r'\bED\b(?!\s+spoke)'),   # "ED" alone (not "ED spoke with")
    re.compile(r'\bBID\b'),
    re.compile(r'\bQHS\b'),
    re.compile(r'\bPRN\b'),
    re.compile(r'\bPO\b'),
    re.compile(r'\bIV\b'),
]

_MIN_ASSISTANT_LENGTH = 40   # characters
_MIN_SHORT_QA_LENGTH = 5


def validate(example: dict) -> Tuple[bool, list[str]]:
    """
    Validate a single training example.
    Returns (is_valid, list_of_issues).
    """
    issues = []

    messages = example.get("messages", [])
    task = example.get("task", "")
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

    # 1. Minimum length
    short_answer_task = task.startswith("qa_") or task == "allergy_profile"
    min_length = _MIN_SHORT_QA_LENGTH if short_answer_task else _MIN_ASSISTANT_LENGTH
    if len(assistant_msg.strip()) < min_length:
        issues.append(f"Assistant response too short ({len(assistant_msg)} chars)")

    # 2. PHI leak check (all message fields)
    full_text = " ".join(m["content"] for m in messages)
    for pattern in _PHI_LEAK_PATTERNS:
        if pattern.search(full_text):
            issues.append(f"Potential PHI detected: pattern {pattern.pattern[:40]}")

    # 3. Residual unexpanded abbreviations in assistant response
    for pattern in _RESIDUAL_ABBREV_PATTERNS:
        if pattern.search(assistant_msg):
            issues.append(f"Unexpanded abbreviation in assistant text: {pattern.pattern}")

    # 4. Blank/placeholder-only content
    if re.fullmatch(r'[\s\[REDACTED\]]+', assistant_msg):
        issues.append("Assistant response contains only redacted placeholders")

    # 5. Must have all three roles
    roles = {m["role"] for m in messages}
    for required in ("system", "user", "assistant"):
        if required not in roles:
            issues.append(f"Missing '{required}' message")

    return (len(issues) == 0, issues)


def validate_batch(examples: list[dict]) -> Tuple[list[dict], list[dict]]:
    """Split examples into (valid, rejected) lists, printing a validation report."""
    valid, rejected = [], []
    for ex in examples:
        ok, issues = validate(ex)
        if ok:
            valid.append(ex)
        else:
            ex["_validation_issues"] = issues
            rejected.append(ex)
    return valid, rejected
