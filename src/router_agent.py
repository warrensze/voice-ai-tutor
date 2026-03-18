"""Routing logic for selecting a specialist tutor agent."""

import re

HISTORY_KEYWORDS = {
    "history",
    "historical",
    "empire",
    "civilization",
    "war",
    "revolution",
    "timeline",
    "dynasty",
    "world war",
    "ancient",
    "medieval",
    "modern history",
}

CHEMISTRY_KEYWORDS = {
    "chemistry",
    "chemical",
    "atom",
    "molecule",
    "ionic",
    "covalent",
    "molar",
    "molarity",
    "acid",
    "base",
    "ph",
    "reaction",
    "equation",
    "stoichiometry",
    "periodic table",
}

MATH_KEYWORDS = {
    "math",
    "algebra",
    "geometry",
    "calculus",
    "equation",
    "solve",
    "derivative",
    "integral",
    "probability",
    "statistics",
    "matrix",
    "function",
    "theorem",
    "triangle",
    "quadratic",
}

ENGLISH_KEYWORDS = {
    "english",
    "literature",
    "literary",
    "novel",
    "poem",
    "poetry",
    "grammar",
    "vocabulary",
    "essay",
    "thesis",
    "reading comprehension",
    "analyze this passage",
    "theme",
    "character",
    "tone",
}

SUBJECT_ALIAS_PATTERNS = {
    "history": (
        r"\bhistory\b",
        r"\bworld\s+history\b",
        r"\bap\s+world(?:\s+history)?\b",
        r"\bhistorical\b",
    ),
    "chemistry": (
        r"\bchemistry\b",
        r"\bchemical\b",
        r"\bchem\b",
    ),
    "math": (
        r"\bmath\b",
        r"\bmathematics\b",
        r"\balgebra\b",
        r"\bgeometry\b",
        r"\bcalculus\b",
        r"\bstatistics?\b",
    ),
    "english": (
        r"\benglish\b",
        r"\bliterature\b",
        r"\bgrammar\b",
        r"\bwriting\b",
        r"\breading\b",
    ),
}


def _detect_subject_mention(text_lower: str) -> str | None:
    """Detect direct subject mentions using alias patterns."""
    scores = {subject: 0 for subject in SUBJECT_ALIAS_PATTERNS}
    for subject, patterns in SUBJECT_ALIAS_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                scores[subject] += 1

    max_score = max(scores.values())
    if max_score <= 0:
        return None

    # Keep deterministic tie-breaking aligned with existing routing order.
    for subject in ("math", "chemistry", "history", "english"):
        if scores[subject] == max_score:
            return subject

    return None


def _score_keywords(question: str, keywords: set[str]) -> int:
    """Count keyword hits in a normalized question string using word boundaries."""
    text = question.lower()
    count = 0
    for keyword in keywords:
        # Use word boundary matching to handle punctuation properly
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, text):
            count += 1
    return count


def route_subject(question: str) -> str:
    """Route a question to history, chemistry, math, or English specialist."""
    if not question or not question.strip():
        return "english"

    scores = {
        "history": _score_keywords(question, HISTORY_KEYWORDS),
        "chemistry": _score_keywords(question, CHEMISTRY_KEYWORDS),
        "math": _score_keywords(question, MATH_KEYWORDS),
        "english": _score_keywords(question, ENGLISH_KEYWORDS),
    }

    max_score = max(scores.values())
    if max_score <= 0:
        return "english"

    # Use deterministic priority for ties.
    for subject in ("math", "chemistry", "history", "english"):
        if scores[subject] == max_score:
            return subject

    return "english"


def route_subject_sticky(
    question: str, current_subject: str | None = None
) -> tuple[str, bool]:
    """Route a question with 'sticky' subject preference.

    Returns (subject, is_explicit_switch) where:
    - subject: The determined subject
    - is_explicit_switch: True if user explicitly switched topics, False if continuing

    Sticky behavior: Allow switches when user explicitly says they want to change topics.
    Only stick to current subject when signal is weak (0-1 keywords) AND no explicit transition.
    """
    if not question or not question.strip():
        return "english", False

    # Check for explicit transition phrases like "I want to talk about", "discuss", "switch to"
    text_lower = question.lower()
    direct_subject = _detect_subject_mention(text_lower)
    has_explicit = any(
        re.search(pattern, text_lower)
        for pattern in [
            r"\b(want to|let'?s|can we|could we|should we)\s+(talk|discuss|go|switch|change)",
            r"\b(want|wanna|need|like)\s+(to\s+)?(study|learn|do|cover|focus(?:\s+on)?)\b",
            r"\bi\s+(want|need|like)\b",
            r"\b(switch|change|go)\s+(to|back to)\b",
            r"\b(i'?d like to|i'?m interested in)\b",
            r"\b(now)\s+(let'?s|can we)\s+(talk|discuss)",
        ]
    )

    # If user explicitly asks to switch/study and mentions a subject, honor it immediately.
    if has_explicit and direct_subject and direct_subject != current_subject:
        return direct_subject, True

    # One-word or short direct subject mentions should switch quickly.
    # This handles speech transcripts like "history" or "i want history".
    if direct_subject and direct_subject != current_subject:
        word_count = len(text_lower.split())
        if word_count <= 3:
            return direct_subject, True

    scores = {
        "history": _score_keywords(question, HISTORY_KEYWORDS),
        "chemistry": _score_keywords(question, CHEMISTRY_KEYWORDS),
        "math": _score_keywords(question, MATH_KEYWORDS),
        "english": _score_keywords(question, ENGLISH_KEYWORDS),
    }

    max_score = max(scores.values())

    # Find the subject with max score (deterministic on ties)
    determined_subject = "english"
    for subject in ("math", "chemistry", "history", "english"):
        if scores[subject] == max_score:
            determined_subject = subject
            break

    # If no clear signal (score 0) and we have a current subject, stick with it
    if max_score == 0 and current_subject:
        return current_subject, False

    # If explicit transition detected AND a different subject was mentioned, always allow switch
    if has_explicit and determined_subject != current_subject:
        return determined_subject, True

    # If single keyword match (score 1) and we have a current subject, stick with it
    # UNLESS there's explicit transition language
    if max_score == 1 and current_subject and determined_subject != current_subject:
        return current_subject, False

    # If two or more keywords match (score >= 2) and it's different from current, it's explicit
    is_explicit_switch = (
        current_subject and determined_subject != current_subject and max_score >= 2
    )

    # Default to english if no subject found and no current context
    if max_score == 0 and not current_subject:
        determined_subject = "english"

    return determined_subject, is_explicit_switch
