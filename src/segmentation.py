"""Module 1: Claim Segmentation for LitScope-Lite.

Provides rule-based sentence splitting to convert unstructured or semi-structured
scientific input text into individual atomic claim candidate strings.
"""

import re
from typing import List


# Common scientific and general abbreviations to avoid false positive sentence splits
ABBREVIATIONS = [
    r"e\.g\.",
    r"i\.e\.",
    r"et\s+al\.",
    r"vs\.",
    r"approx\.",
    r"ca\.",
    r"fig\.",
    r"figs\.",
    r"ref\.",
    r"refs\.",
    r"no\.",
    r"vol\.",
    r"pp\.",
    r"dept\.",
    r"dr\.",
    r"prof\.",
    r"mr\.",
    r"mrs\.",
    r"ms\.",
    r"st\.",
    r"gen\.",
    r"sp\.",
    r"spp\.",
    r"eq\.",
    r"eqs\.",
    r"wt\.",
    r"mol\.",
]

# Regex pattern matching leading bullet points / list numbers
BULLET_PREFIX_PATTERN = re.compile(r"^(?:[-*•–—]|\(?\d+\)?[.)]|\(?[a-zA-Z]\)[.)])\s+")


def _protect_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    """Replace dots in known abbreviations with temporary placeholders."""
    protected_text = text
    placeholder_map = {}
    
    # 1. Protect decimal numbers (e.g., 3.14, 0.05, 1.2e-3)
    decimal_pattern = re.compile(r"(\d+)\.(\d+)")
    def decimal_sub(m):
        key = f"__DECIMAL_{len(placeholder_map)}__"
        placeholder_map[key] = f"{m.group(1)}.{m.group(2)}"
        return key
    protected_text = decimal_pattern.sub(decimal_sub, protected_text)

    # 2. Protect known abbreviations (case-insensitive)
    for i, abbr_pattern in enumerate(ABBREVIATIONS):
        pattern = re.compile(r"\b" + abbr_pattern, re.IGNORECASE)
        for match in pattern.finditer(protected_text):
            matched_str = match.group(0)
            key = f"__ABBR_{i}_{len(placeholder_map)}__"
            placeholder_map[key] = matched_str
            protected_text = protected_text.replace(matched_str, key, 1)

    # 3. Protect single capital initials (e.g., J. Smith, A. B. C.)
    initial_pattern = re.compile(r"\b([A-Z])\.\s(?=[A-Z])")
    def initial_sub(m):
        key = f"__INITIAL_{len(placeholder_map)}__"
        placeholder_map[key] = f"{m.group(1)}. "
        return key
    protected_text = initial_pattern.sub(initial_sub, protected_text)

    return protected_text, placeholder_map


def _restore_abbreviations(text: str, placeholder_map: dict[str, str]) -> str:
    """Restore original abbreviation tokens from placeholders."""
    restored = text
    for key, orig in placeholder_map.items():
        restored = restored.replace(key, orig)
    return restored


def segment_claims(text: str) -> List[str]:
    """Segment an input text into individual claim strings using rule-based sentence splitting.
    
    Handles scientific abbreviations (e.g., et al., i.e., e.g., fig., vs.), decimal numbers,
    multi-line paragraphs, bullet lists, and quotation marks.
    
    Args:
        text: Input string containing one or more claims or paragraphs.
        
    Returns:
        List of non-empty, stripped claim strings.
    """
    if not text or not isinstance(text, str):
        return []

    # First split across hard newlines
    raw_lines = text.splitlines()
    claims: List[str] = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        # Strip bullet points or numbered lists at start of line
        cleaned_line = BULLET_PREFIX_PATTERN.sub("", line).strip()
        if not cleaned_line:
            continue

        # Protect known abbreviations & decimals from splitting
        protected_line, placeholder_map = _protect_abbreviations(cleaned_line)

        # Split on sentence boundaries: punctuation (. ! ?) followed by whitespace and capital/quote/digit
        sentence_split_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])')
        raw_sentences = sentence_split_pattern.split(protected_line)

        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Restore abbreviations
            restored_sent = _restore_abbreviations(sent, placeholder_map).strip()

            # Clean bullet prefixes again if nested
            restored_sent = BULLET_PREFIX_PATTERN.sub("", restored_sent).strip()

            if restored_sent:
                claims.append(restored_sent)

    return claims

