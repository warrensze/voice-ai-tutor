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
