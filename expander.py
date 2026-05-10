"""
Abbreviation expander.
Uses longest-match, word-boundary-aware substitution.
Context rules prevent over-expansion (e.g. "F" only in structured fields).
"""

import re
from abbreviations import ABBREVIATIONS


# Build a sorted list: longest phrase first so multi-word abbreviations match before singles
_SORTED_ABBREVS = sorted(ABBREVIATIONS.items(), key=lambda x: len(x[0]), reverse=True)

# Abbreviations that are too short/ambiguous to expand in free text
_FREETEXT_EXCLUDE = {"F", "M", "K", "Na", "Ca", "CO2"}
_SKIP_IF_FOLLOWED_BY = {
    "QT": r"\s+interval",
}


def expand_abbreviations(text: str, structured_field: bool = False) -> str:
    """
    Expand medical abbreviations in text.
    structured_field=True allows single-letter expansions (Sex: F → female).
    """
    text = re.sub(r'(?<=\d)(?=(?:mg|mcg|mL|ml|g)\b)', ' ', text)
    protected: dict[str, str] = {}

    for abbrev, expansion in _SORTED_ABBREVS:
        if abbrev in _FREETEXT_EXCLUDE and not structured_field:
            continue
        # Build a regex that matches the abbreviation as a whole token.
        # Escape special chars, allow optional trailing punctuation.
        escaped = re.escape(abbrev)
        skip_following = _SKIP_IF_FOLLOWED_BY.get(abbrev, "")
        # Word boundary on both sides; handle slash-delimited abbrevs
        pattern = re.compile(
            r'(?<![A-Za-z])' + escaped + r'(?![A-Za-z0-9])'
            + (rf'(?!{skip_following})' if skip_following else ''),
        )
        def protect_match(_: re.Match) -> str:
            placeholder = f"@@EXP{len(protected)}@@"
            protected[placeholder] = expansion
            return placeholder

        text = pattern.sub(protect_match, text)

    for placeholder, expansion in protected.items():
        text = text.replace(placeholder, expansion)
    text = re.sub(r':(?=\S)', ': ', text)
    return text


def expand_text(text: str) -> str:
    return expand_abbreviations(text, structured_field=False)


def expand_structured(text: str) -> str:
    return expand_abbreviations(text, structured_field=True)
